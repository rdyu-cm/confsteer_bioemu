#!/usr/bin/env python3
"""Score the held-out kinase benchmark draws on success@B and write their sidecars. CPU ONLY.

SCOPE: KINASES, DIRECT METRIC. This is the scoring half of
``scripts/kinase_dfg/generate_kinase_benchmark_draws.py`` and a port BY VALUE of
``scripts/domain_motion/measure_success_at_b.py`` onto the kinase-RMSD metric. It reads published
``.pdb`` structures and a reference artifact, measures Angstrom distances, and writes a JSON and a
CSV. **It constructs no sampler, loads no adapter, imports no BioEmu module and touches no GPU** --
a scorer that could generate could be run to a threshold, and a scorer that needed a GPU could not
be pointed at an archive.

WHAT ONE MEASUREMENT IS
------------------------
For one draw of system ``p`` and one reference state ``r``:

    X = N/CA/C/O of the draw, read off its published PDB, restricted to ``p``'s METRIC INDICES
    Y = the deposited chain ``r`` of ``p``, already stored restricted to the same indices
    m = aligned_backbone_rmsd(X, Y)     Angstrom, superposed over exactly the atoms it measures

``m`` is the same kernel the training loss descended
(:func:`~sae_bioemu.kinase_rmsd.objective.aligned_backbone_rmsd`), read here off a file instead of
off a differentiable frame. It is imported rather than reimplemented so the quantity reported and
the quantity descended cannot become two.

success@B: a system SUCCEEDS when the minimum ``m`` over ``B`` of its draws is below ``tau``; the
rate is the UNWEIGHTED fraction of systems that succeed. Draws are never pooled -- 300 draws of one
kinase are 300 draws of one protein -- and the bootstrap resamples SYSTEMS for the same reason.

PER REFERENCE STATE, NEVER POOLED. A kinase-RMSD level NAMES A DEPOSITED, KINCORE-LABELLED CHAIN
(:func:`~sae_bioemu.kinase_rmsd.objective.reference_state_for_level`: ``-1`` is DFGin/BLAminus,
``+1`` is DFGout/BBAminus), so ``dfg_in`` and ``dfg_out`` are two questions about two crystals and
an average over them is a number about neither. Every cell, every CSV row and every emitted
deviation carries its state.

WHY THE FIRST-B CELLS HERE ARE NOT THE HEADLINE ESTIMATOR
----------------------------------------------------------
The cells written below take the FIRST ``B`` draws in ``sample_index`` order, exactly as the
domain-motion original does. At ``B`` equal to the pool depth that is every draw and is exact; below
it, it is ONE deterministic subset, and these runs are 300 draws deep precisely so that ``B`` can be
swept beneath the depth. **That sweep is not implemented here.**
``rescore_expected_success.py`` already does it -- the
expectation over random B-subsets, ConforNets' ``subset_std`` convention, and the nested system
bootstrap -- and it consumes a ``sample_deviations`` mapping. So this file emits a deviation for
EVERY draw in exactly that shape::

    payload["sample_deviations"][<label>] = [
        {"sample_id": <sequence_id>, "reference_state": "dfg_in"|"dfg_out", "rmsd": <Angstrom>, ...}
    ]

and the rescorer is run straight on it with ``--arm <label> --tau 2.0``. Emitting only the first
``B`` would throw away the depth these 18,000 generations are paying for.

TAU IS 2.0 ANGSTROM AND IS RECORDED IN EVERY OUTPUT. See :data:`DEFAULT_TAU_ANGSTROM`.

WHAT IT REFUSES
----------------
* **A system the artifact names and the views do not.** success@B is a mean over SYSTEMS, so a
  missing system is a different denominator. The generator shards BY SYSTEM and writes one
  ``views.shard<k>of<n>/`` per shard, so the whole panel is the CONCATENATION of several roots --
  which is why ``--views-root`` is repeatable, and why the union is checked against the artifact's
  own roster rather than against whatever happened to be passed. Per-shard success does not compose.
* **A ragged pool.** Systems at different depths are named, not truncated silently. ``--truncate-
  pool`` makes the pool square at an explicit depth and is recorded in the payload.
* **A reference/level mismatch.** A view directory is named ``level_<state>`` by the generator, and
  when a ``--run-report`` is supplied its ``views`` block must agree about ``reference_state`` and
  ``reference_entry_index`` (0 is ``dfg_in``), and its artifact fingerprint, metric set and metric
  objective must be this artifact's. A scorer that paired a level with the wrong deposited chain
  would report a complete, finite, INVERTED number.
* **An unreadable or mis-framed structure.** A PDB whose residue letters do not spell the indexed
  sequence, whose residues are not numbered ``0..L-1``, or which is missing a backbone atom, is
  refused by name. Nothing falls back to C-alpha.
* **A draw claimed twice.** Two sources naming one ``(system, state, sample_index)`` is a views root
  passed twice or two structures claiming one draw; both are refused.

THREE PLACES KINASE SEMANTICS FORCED A DIVERGENCE FROM THE DOMAIN-MOTION ORIGINAL
----------------------------------------------------------------------------------
1. **NO SEQUENCE ALIGNMENT, AND NO COVERAGE FLOOR.** The domain-motion scorer places deposited
   chains and generated draws onto a test-case sequence with a Biopython global alignment, and
   refuses a system whose coverage falls under 0.9, because its references are deposited PDBs in
   their own crystallographic numbering. Here the artifact stores both chains ALREADY on the folded
   sequence's own residue frame and ALREADY restricted to ``metric_indices``, and the generated PDB
   is that same sequence numbered ``0..L-1``. So the correspondence is an array index, the coverage
   question was answered offline by the reference builder (and is reported per entry as
   ``resolved_fraction``), and an alignment here would be a second, weaker frame for the same atoms.
2. **STATE NAMES, NOT A CONFORMER INDEX.** The original selects with ``--conformer-index 0|1``
   because a DynDom pair's members are arbitrary. Both chains here carry a Kincore label, the
   generator writes ``views/level_dfg_in`` and ``views/level_dfg_out``, so ``--state`` takes the
   name. ``reference_entry_index`` is DERIVED from
   :data:`~sae_bioemu.kinase_rmsd.objective.REFERENCE_STATES` and reported beside it.
3. **THE ADAPTIVE CELL USES THE ARTIFACT'S OWN s_p.** The original measures the conformer-to-
   conformer separation itself, under the sample protocol, because nothing upstream carries it. Here
   ``s_p`` is in the artifact, re-measured from its own coordinates on every load, and is the
   divisor the training objective used; measuring it again here could only produce a second number
   for one quantity.

Usage::

    PYTHONPATH=src python scripts/kinase_dfg/measure_success_at_b.py \\
        --artifact <data-root>/kinase_rmsd/reference_holdout20/kinase_rmsd_reference.pt \\
        --views-root <data-root>/kinase_dfg/out/holdout20_steered/views.shard0of4 \\
        --views-root <data-root>/kinase_dfg/out/holdout20_steered/views.shard1of4 \\
        --views-root <data-root>/kinase_dfg/out/holdout20_steered/views.shard2of4 \\
        --views-root <data-root>/kinase_dfg/out/holdout20_steered/views.shard3of4 \\
        --run-report <data-root>/kinase_dfg/out/holdout20_steered/kinase_benchmark_draws.shard0of4.json \\
        --label steered --output <data-root>/kinase_dfg/out/holdout20_steered/success_at_b.steered.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sae_bioemu.kinase_rmsd.artifact import (
    KinaseRmsdEntry,
    KinaseRmsdReference,
    load_kinase_rmsd_reference,
)
from sae_bioemu.kinase_rmsd.frames import BACKBONE_ATOMS
from sae_bioemu.kinase_rmsd.objective import (
    KINASE_RMSD_PROPERTY_NAME,
    REFERENCE_STATES,
    aligned_backbone_rmsd,
    reference_state_for_level,
)

#: The two reference states, in the ascending LEVEL order the generator and the objective use.
#: Derived from :data:`~sae_bioemu.kinase_rmsd.objective.REFERENCE_STATES` rather than restated, so
#: the states this scorer answers for and the states a level can name cannot become two sets.
STATES: tuple[str, ...] = tuple(
    reference_state_for_level(level) for level in sorted(REFERENCE_STATES)
)

#: Which member of ``KinaseRmsdEntry.entry_ids`` -- which is ``(dfg_in, dfg_out)`` -- each state is.
#: The generator records the same number per view as ``reference_entry_index``; both are derived
#: from the same ordering above and are cross-checked against each other when a run report is given.
STATE_ENTRY_INDEX: dict[str, int] = {state: index for index, state in enumerate(STATES)}

#: The level that names each state, for the report and for the CSV.
LEVEL_FOR_STATE: dict[str, float] = {
    reference_state_for_level(level): float(level) for level in REFERENCE_STATES
}

#: The success threshold for KINASES, in Angstrom. **2.0, and it is not this property's
#: ``KINASE_RMSD_TAU_ANGSTROM`` of 3.0.** The source is ConforNets' own
#: ``assets/config.yaml`` -> ``benchmarks: kinases: rmsd_threshold: 2.0`` (verified at
#: ConforNets' own ``assets/config.yaml``), which is the
#: threshold the external kinase numbers this benchmark is meant to be comparable with were reported
#: at. ``KINASE_RMSD_TAU_ANGSTROM`` is the domain-motion 3.0 copied by value for the hinge ablation
#: and for this project's own internal reporting; using it here would report a kinase rate against a
#: DynDom threshold and quietly beat a published number by a looser rule. It is a DEFAULT and is
#: written into the JSON, into every CSV row and into every printed line, because tau dominates the
#: number.
DEFAULT_TAU_ANGSTROM = 2.0

#: Percentile bootstrap over SYSTEMS. One system is one protein; resampling draws within a system
#: would treat 300 draws of one kinase as 300 proteins.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260919

#: The generator's own report schema. A file that does not declare it is refused rather than read
#: for keys that happen to be present.
GENERATOR_REPORT_SCHEMA = "kinase_rmsd.benchmark_draws.v1"

OUTPUT_SCHEMA = "kinase_rmsd.success_at_b.v1"

#: What an ``--index`` may be scored against. ``both`` exists for the ADAPTER-OFF control view: one
#: unsteered draw is the matched counterfactual for BOTH levels of its pair, so the generator writes
#: a single ``control/`` view and a baseline rate has to measure each of those draws against each
#: deposited chain. A steered view is never ``both`` -- its level already named one chain.
INDEX_STATE_CHOICES: tuple[str, ...] = (*STATES, "both")

_ATOM_RECORD = re.compile(r"^ATOM  ")

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip


# ------------------------------------------------------------------------------------------------
# Structures
# ------------------------------------------------------------------------------------------------


def read_generated_backbone(path: Path, sequence: str) -> np.ndarray:
    """Return ``[len(sequence), 4, 3]`` Angstrom N/CA/C/O off one BioEmu final-structure PDB.

    BioEmu writes N, CA, C, CB and O per residue from ``get_atom37_from_frames`` -- the same
    construction :mod:`sae_bioemu.kinase_rmsd.frames` reads for the training metric -- numbered
    ``0..L-1`` on one chain, so all four metric atoms are present and no C-alpha fallback is ever
    needed or taken. The C-beta is dropped here exactly as the objective's atom set drops it.

    Everything is CHECKED rather than assumed: the letters must spell the indexed sequence, the
    numbering must be the plain ``0..L-1`` index, and every residue must carry all four atoms. A
    file that fails any of those is refused by name. The alternative is a finite, plausible RMSD
    measured over a permuted or shifted residue frame, which is this project's characteristic silent
    failure.
    """
    try:
        text = path.read_text()
    except OSError as error:
        raise SystemExit(
            f"{path}: the structure is unreadable ({error}). A draw that cannot be opened is not a "
            "draw that scored badly, and dropping it would shrink this system's pool silently"
        ) from error

    keys: list[str] = []
    letters: list[str] = []
    rows: list[dict[str, np.ndarray]] = []
    current: tuple[str, str, str] | None = None
    wanted = set(BACKBONE_ATOMS)
    atoms: dict[str, np.ndarray] = {}

    def flush() -> None:
        if current is None:
            return
        letter = _THREE_TO_ONE.get(current[1])
        if letter is None:
            raise SystemExit(
                f"{path}: residue {current[0]} is {current[1]!r}, which is not one of the twenty "
                "standard residues BioEmu folds; the sequence frame cannot be checked against it"
            )
        keys.append(current[0])
        letters.append(letter)
        rows.append(dict(atoms))

    for line in text.splitlines():
        # ONLY THE FIRST MODEL. These files are single-model today; a multi-model file silently
        # concatenated would give two residues one index and shift the whole metric set.
        if line.startswith("ENDMDL"):
            break
        if not _ATOM_RECORD.match(line):
            continue
        name = line[12:16].strip()
        marker = (f"{line[22:26].strip()}{line[26].strip()}", line[17:20].strip().upper(), line[21])
        if current is None or marker != current:
            flush()
            atoms = {}
            current = marker
        if name in wanted:
            atoms[name] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64
            )
    flush()

    if "".join(letters) != sequence:
        raise SystemExit(
            f"{path}: the residue letters do not spell the indexed sequence "
            f"({len(letters)} residues against {len(sequence)}). metric_indices are positions in "
            "the artifact's sequence, so a different chain would restrict the metric to different "
            "residues and still return a finite number"
        )
    if keys != [str(index) for index in range(len(sequence))]:
        raise SystemExit(
            f"{path}: residues are not numbered 0..L-1 as BioEmu writes them, so a position in the "
            "artifact's sequence frame does not name the residue it appears to"
        )
    missing = [
        (index, name)
        for index, residue in enumerate(rows)
        for name in BACKBONE_ATOMS
        if name not in residue
    ]
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} backbone atoms are absent, first {missing[:5]}. The metric is "
            f"over {list(BACKBONE_ATOMS)} and there is no C-alpha fallback"
        )
    return np.stack(
        [np.stack([residue[name] for name in BACKBONE_ATOMS]) for residue in rows]
    ).astype(np.float64)


def draw_rmsd(backbone: np.ndarray, entry: KinaseRmsdEntry, state: str) -> float:
    """Return the superposed N/CA/C/O RMSD of one draw against one deposited chain, in Angstrom.

    The draw is restricted by the entry's OWN ``metric_indices`` -- the same list the stored chains
    were restricted by at build time -- so the alignment set and the metric set are one set here for
    the same structural reason they are one set in the training path.
    """
    selected = torch.as_tensor(backbone[list(entry.metric_indices)], dtype=torch.float64)
    with torch.no_grad():
        return float(aligned_backbone_rmsd(selected, entry.chain(state).to(torch.float64)))


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Draw:
    """One published structure and the reference states it is to be measured against."""

    sequence_id: str
    sequence: str
    sample_index: int
    pdb_path: Path
    index_path: Path
    states: tuple[str, ...]


def read_sample_index(index_path: Path, states: Sequence[str]) -> list[Draw]:
    """Read one ``final_structures/index.jsonl`` view into draws.

    Paths are resolved against ``index.parent.parent``, which is the shape
    ``generate_kinase_benchmark_draws.write_index_view`` writes and the shape every success@B reader
    in this project already resolves. That is what lets one store serve three reads -- level
    ``dfg_in``, level ``dfg_out`` and the adapter-off control -- without copying a structure.
    """
    if not index_path.is_file():
        raise SystemExit(
            f"{index_path} is not a file. The generator rebuilds its views from the store's own "
            "sidecar after every system, so a shard that has published nothing yet has no index; "
            "scoring the remaining shards alone would be a rate over a different denominator"
        )
    root = index_path.parent.parent
    draws: list[Draw] = []
    for number, line in enumerate(index_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{index_path}:{number} is not readable JSON ({error})") from error
        draws.append(
            Draw(
                sequence_id=str(record["sequence_id"]),
                sequence=str(record["sequence"]),
                sample_index=int(record["sample_index"]),
                pdb_path=root / str(record["pdb_path"]),
                index_path=index_path,
                states=tuple(states),
            )
        )
    return draws


def resolve_views(root: Path, arm: str) -> list[tuple[Path, tuple[str, ...]]]:
    """Map one ``views/`` or ``views.shard<k>of<n>/`` directory to its indices and their states."""
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    if arm == "control":
        return [(root / "control" / "final_structures" / "index.jsonl", STATES)]
    legacy = sorted(
        child.name for child in root.iterdir() if child.name in {"level_m1", "level_p1"}
    )
    if legacy:
        raise SystemExit(
            f"{root} holds {legacy}, which is the DOMAIN-MOTION view spelling. A kinase views root "
            f"names its states ({list(STATES)}) because both deposited chains carry a Kincore "
            "label; this is a domain-motion store and its levels mean a different pair of chains"
        )
    resolved: list[tuple[Path, tuple[str, ...]]] = []
    for state in STATES:
        path = root / f"level_{state}" / "final_structures" / "index.jsonl"
        if not path.is_file():
            raise SystemExit(
                f"{root} carries no level_{state}/final_structures/index.jsonl. Both states are "
                "read from one views root because a rate is reported per state and a root missing "
                "one of them would report a single-state table as though it were the pair"
            )
        resolved.append((path, (state,)))
    return resolved


def check_run_report(report_path: Path, reference: KinaseRmsdReference) -> dict[str, object]:
    """Cross-check one generator run report against this artifact, and return its provenance.

    The FINGERPRINT, the METRIC SET and the METRIC OBJECTIVE are compared because each of them
    changes what a distance MEANS: a different fingerprint is different coordinates, a different
    metric set is a different ~110 residues, and a different objective form is a different
    per-system target the level was ever asked to reach. The views block's ``reference_state`` and
    ``reference_entry_index`` are compared against this file's own derivation of both, because a
    level paired with the wrong deposited chain produces a complete, finite, inverted number.
    """
    payload = json.loads(report_path.read_text())
    if payload.get("schema") != GENERATOR_REPORT_SCHEMA:
        raise SystemExit(
            f"{report_path} declares schema {payload.get('schema')!r}, not "
            f"{GENERATOR_REPORT_SCHEMA!r}; it is not a kinase benchmark-draw run report"
        )
    fingerprint = reference.content_fingerprint()
    for key, expected in (
        ("artifact_content_fingerprint", fingerprint),
        ("artifact_metric_set", reference.metric_set),
        ("artifact_metric_objective", reference.metric_objective),
    ):
        declared = payload.get(key)
        if declared is not None and str(declared) != str(expected):
            raise SystemExit(
                f"{report_path} records {key}={declared!r} and this artifact "
                f"({reference.path}) carries {expected!r}. The draws in that store were generated "
                "against a different reference, so scoring them against this one measures a "
                "distance to a chain the adapter was never aimed at"
            )
    views = payload.get("views")
    if isinstance(views, dict):
        for state in STATES:
            block = views.get(f"level_{state}")
            if not isinstance(block, dict):
                continue
            if str(block.get("reference_state")) != state:
                raise SystemExit(
                    f"{report_path}: view level_{state} declares reference_state "
                    f"{block.get('reference_state')!r}"
                )
            if int(block.get("reference_entry_index", -1)) != STATE_ENTRY_INDEX[state]:
                raise SystemExit(
                    f"{report_path}: view level_{state} declares reference_entry_index "
                    f"{block.get('reference_entry_index')!r}, not {STATE_ENTRY_INDEX[state]}. "
                    "entry_ids is (dfg_in, dfg_out), so a level paired with the other member would "
                    "be scored against the chain it was steered away from"
                )
            level = block.get("level")
            if level is not None and reference_state_for_level(float(level)) != state:
                raise SystemExit(
                    f"{report_path}: view level_{state} was generated at level {level!r}, which "
                    f"names {reference_state_for_level(float(level))!r}"
                )
    return {
        "path": str(report_path),
        "commit": payload.get("commit"),
        "complete": payload.get("complete"),
        "shard": payload.get("shard"),
        "shards": payload.get("shards"),
        "systems": payload.get("systems"),
        "draws_per_system": payload.get("draws_per_system"),
        "levels": payload.get("levels"),
        "adapter": payload.get("adapter"),
        "adapter_checkpoint_identity": payload.get("adapter_checkpoint_identity"),
        "artifact": payload.get("artifact"),
        "artifact_content_fingerprint": payload.get("artifact_content_fingerprint"),
        "restricted_to_sequences": payload.get("restricted_to_sequences"),
    }


# ------------------------------------------------------------------------------------------------
# Coverage
# ------------------------------------------------------------------------------------------------


def pooled_draws(draws: Iterable[Draw]) -> dict[tuple[str, str], dict[int, Draw]]:
    """Collapse every source onto ``{(sequence_id, state): {sample_index: draw}}``, refusing clashes.

    Two sources naming one ``(system, state, sample_index)`` is either a views root passed twice or
    two structures claiming one draw. Both are refused: the first would double a system's apparent
    depth, the second would make "the best of B" depend on which file was read last.
    """
    pools: dict[tuple[str, str], dict[int, Draw]] = {}
    for draw in draws:
        for state in draw.states:
            slot = pools.setdefault((draw.sequence_id, state), {})
            existing = slot.get(draw.sample_index)
            if existing is not None:
                same = existing.pdb_path.resolve() == draw.pdb_path.resolve()
                raise SystemExit(
                    f"{draw.sequence_id} {state} sample_index {draw.sample_index} is claimed by "
                    f"{existing.index_path} and by {draw.index_path}"
                    + (
                        "; both resolve to the same structure, so a source was named twice and the "
                        "pool would be reported at twice its depth"
                        if same
                        else f"; they name {existing.pdb_path} and {draw.pdb_path}, two structures "
                        "for one draw"
                    )
                )
            slot[draw.sample_index] = draw
    return pools


@dataclass(frozen=True)
class Coverage:
    """What the union of the supplied sources covers, and at what depth."""

    required: tuple[str, ...]
    depth: int
    per_state_systems: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "systems_required": list(self.required),
            "n_systems_required": len(self.required),
            "pool_depth": self.depth,
            "systems_per_state": dict(self.per_state_systems),
        }


def require_square_coverage(
    pools: dict[tuple[str, str], dict[int, Draw]],
    required: Sequence[str],
    states: Sequence[str],
    *,
    truncate_pool: int | None,
) -> Coverage:
    """Fail closed unless every required system carries the same number of draws in every state.

    THREE SEPARATE REFUSALS, none of which the other two catch:

    * a system the artifact names and no source carries -- the denominator of a mean over systems;
    * a system some source carries and the artifact does not name -- a draw scored against a
      reference that is not this artifact's;
    * a RAGGED pool -- systems at different depths. This is the ordinary mid-run state of four
      shards resuming independently, and it is exactly when a success@B table is most tempting and
      least comparable: a system with 300 draws and one with 40 are not two systems at one ``B``.
      ``truncate_pool`` squares it at an EXPLICIT depth, which is recorded in the payload; it cannot
      rescue a system that is short of that depth, and does not pretend to.
    """
    present = {system for system, _ in pools}
    unknown = sorted(present - set(required))
    if unknown:
        raise SystemExit(
            f"the supplied views carry rows for {len(unknown)} systems the artifact does not name: "
            f"{unknown[:10]}. Those draws were generated against a different reference, and "
            "scoring them here would measure a distance to another kinase's deposited chains"
        )
    missing = sorted(set(required) - present)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(required)} systems are absent from the union of the supplied "
            f"views: {missing}. The generator shards BY SYSTEM, so a missing system usually means "
            "a shard root was not passed or has published nothing yet. success@B is an unweighted "
            "mean over SYSTEMS -- per-shard success does not compose into panel coverage -- so a "
            "rate over the remaining systems would be a different statistic under the same name. "
            "Pass every views.shard<k>of<n>/ root, or restrict the population deliberately with "
            "--smoke-system (which records the run as incomplete and exits nonzero)"
        )

    if truncate_pool is not None:
        for slot in pools.values():
            for index in sorted(slot)[truncate_pool:]:
                del slot[index]

    depths: dict[int, list[str]] = {}
    for system in required:
        for state in states:
            slot = pools.get((system, state))
            if not slot:
                raise SystemExit(
                    f"{system} carries no draw in state {state!r}. A level names a deposited chain "
                    "and both states are reported, so a system present in one and absent from the "
                    "other cannot be scored"
                )
            depths.setdefault(len(slot), []).append(f"{system}:{state}")
    if len(depths) != 1:
        summary = "; ".join(
            f"{count} draws: {sorted(names)[:6]}{' ...' if len(names) > 6 else ''}"
            for count, names in sorted(depths.items())
        )
        raise SystemExit(
            "THE POOL IS RAGGED -- these (system, state) pools are not all at one depth. "
            f"{summary}. success@B compares the best of B draws across systems, so a panel whose "
            "systems have different depths is a table of different questions. Wait for the shards "
            "to finish, or square the pool explicitly with --truncate-pool "
            f"{min(depths)} (which is recorded in the output)"
        )
    depth = next(iter(depths))
    return Coverage(
        required=tuple(required),
        depth=depth,
        per_state_systems={
            state: sum(1 for system, other in pools if other == state) for state in states
        },
    )


# ------------------------------------------------------------------------------------------------
# The metric
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Deviation:
    """One draw measured against ONE deposited chain."""

    sequence_id: str
    gene: str
    state: str
    sample_index: int
    rmsd_angstrom: float
    metric_residues: int
    separation_angstrom: float
    adaptive_tau_angstrom: float
    entry_id: str
    pdb_path: Path
    index_path: Path


def score_pools(
    pools: dict[tuple[str, str], dict[int, Draw]],
    reference: KinaseRmsdReference,
    required: Sequence[str],
    states: Sequence[str],
) -> list[Deviation]:
    """Measure every draw of every required system against the chain its state names.

    Each published PDB is parsed ONCE per pool entry and released; nothing accumulates a dense
    coordinate array over the panel. The indexed sequence is checked against the artifact's own
    sequence before the file is opened, because a store whose rows name a different construct is a
    different protein and not a badly-scoring draw.
    """
    deviations: list[Deviation] = []
    for system in required:
        entry = reference.require(system)
        normalisation = entry.normalisation
        for state in states:
            chain_id = entry.entry_ids[STATE_ENTRY_INDEX[state]]
            for sample_index in sorted(pools[(system, state)]):
                draw = pools[(system, state)][sample_index]
                if draw.sequence != entry.sequence:
                    raise SystemExit(
                        f"{draw.index_path} indexes {system} as a {len(draw.sequence)}-residue "
                        f"chain and the artifact carries {len(entry.sequence)}; metric_indices are "
                        "positions in the artifact's sequence, so these are two constructs"
                    )
                backbone = read_generated_backbone(draw.pdb_path, entry.sequence)
                deviations.append(
                    Deviation(
                        sequence_id=system,
                        gene=entry.gene,
                        state=state,
                        sample_index=sample_index,
                        rmsd_angstrom=draw_rmsd(backbone, entry, state),
                        metric_residues=len(entry.metric_indices),
                        separation_angstrom=float(entry.separation_angstrom),
                        adaptive_tau_angstrom=float(normalisation.adaptive_tau_angstrom),
                        entry_id=chain_id,
                        pdb_path=draw.pdb_path,
                        index_path=draw.index_path,
                    )
                )
    return deviations


def bootstrap_interval(
    successes: Sequence[bool],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap over SYSTEMS, which are the independent biological units here."""
    flags = np.asarray(list(successes), dtype=float)
    if flags.size == 0:
        return (None, None)
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, flags.size, size=(resamples, flags.size))
    rates = flags[picks].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (float(np.quantile(rates, tail)), float(np.quantile(rates, 1.0 - tail)))


def success_cell(
    deviations: Sequence[Deviation],
    *,
    state: str,
    b: int,
    tau: float | None,
) -> dict[str, object]:
    """One ``(state, tau, B)`` cell: the FIRST ``B`` draws per system, then the unweighted mean.

    ``tau=None`` selects ConforMix's ADAPTIVE per-system threshold, ``0.5 * s_p``, read off the
    artifact. A fixed 2.0 Angstrom on a kinase whose two chains are 1.76 Angstrom apart asks for
    something the crystals themselves barely distinguish, and only the adaptive threshold is
    comparable across a pool whose separations span an order of magnitude.

    The success rule is ``min(rmsd) < tau``, strictly, which is the rule
    ``rescore_expected_success.py`` applies to the deviations emitted beside these cells. The two
    must agree or the headline estimator and its companion table would disagree on ties.
    """
    per_system: dict[str, list[Deviation]] = {}
    for deviation in deviations:
        if deviation.state == state:
            per_system.setdefault(deviation.sequence_id, []).append(deviation)

    systems: list[str] = []
    successes: list[bool] = []
    best: list[float] = []
    thresholds: list[float] = []
    for system in sorted(per_system):
        taken = sorted(per_system[system], key=lambda item: item.sample_index)[:b]
        if not taken:
            continue
        threshold = taken[0].adaptive_tau_angstrom if tau is None else float(tau)
        closest = min(item.rmsd_angstrom for item in taken)
        systems.append(system)
        best.append(closest)
        thresholds.append(threshold)
        successes.append(closest < threshold)
    low, high = bootstrap_interval(successes)
    return {
        "reference_state": state,
        "reference_entry_index": STATE_ENTRY_INDEX[state],
        "level": LEVEL_FOR_STATE[state],
        "tau_angstrom": None if tau is None else float(tau),
        "adaptive_tau": tau is None,
        "b": b,
        "subset_rule": "first B in sample_index order",
        "success_rule": "min(rmsd) < tau",
        "n_systems": len(systems),
        "n_successes": int(sum(successes)),
        "success_rate": (float(np.mean(successes)) if successes else None),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "systems": systems,
        "successes": [bool(value) for value in successes],
        "best_rmsd_angstrom": best,
        "threshold_angstrom": thresholds,
    }


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


_CSV_COLUMNS = (
    "sequence_id",
    "gene",
    "reference_state",
    "reference_entry_index",
    "level",
    "entry_id",
    "sample_index",
    "rmsd_angstrom",
    "tau_angstrom",
    "adaptive_tau_angstrom",
    "separation_angstrom",
    "metric_residues",
    "atoms",
    "draw_within_tau",
    "pdb_path",
    "index_path",
)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="the kinase-RMSD reference the draws were generated against",
    )
    parser.add_argument(
        "--views-root",
        type=Path,
        action="append",
        default=None,
        dest="views_roots",
        help="REPEATABLE. A views/ or views.shard<k>of<n>/ directory. The generator shards BY "
        "SYSTEM, so the whole panel is the concatenation of every shard's root",
    )
    parser.add_argument(
        "--arm",
        choices=("steered", "control"),
        default="steered",
        help="which view of a --views-root to read: the two level_<state>/ directories, or the "
        "adapter-off control/ (whose draws are measured against BOTH deposited chains)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        action="append",
        default=None,
        dest="indices",
        help="REPEATABLE. A final_structures/index.jsonl read directly; needs --index-state",
    )
    parser.add_argument(
        "--index-state",
        choices=INDEX_STATE_CHOICES,
        default=None,
        help="what every --index's draws are measured against. 'both' is the control reading",
    )
    parser.add_argument(
        "--run-report",
        type=Path,
        action="append",
        default=None,
        dest="run_reports",
        help="REPEATABLE. A generator kinase_benchmark_draws.shard<k>of<n>.json, cross-checked "
        "against this artifact and recorded as provenance",
    )
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU_ANGSTROM)
    parser.add_argument(
        "--b",
        type=int,
        action="append",
        default=None,
        dest="b_values",
        help="REPEATABLE. Defaults to the pool depth, where the first-B subset is every draw and "
        "the cell is exact. Below the depth, prefer rescore_expected_success.py on the emitted "
        "sample_deviations, which averages over random B-subsets instead",
    )
    parser.add_argument("--truncate-pool", type=int, default=None)
    parser.add_argument(
        "--smoke-system",
        action="append",
        default=None,
        dest="smoke_systems",
        help="SMOKE ONLY. Restrict the required population to these systems. The run is recorded "
        "as complete=false and exits nonzero, because a rate over a subset is a different "
        "denominator wearing the panel's name",
    )
    parser.add_argument("--output", type=Path, required=True, help="the JSON sidecar")
    parser.add_argument(
        "--draws-csv",
        type=Path,
        default=None,
        help="the per-draw CSV; defaults to <output stem>_draws.csv beside the JSON",
    )
    parser.add_argument("--label", default="steered", help="the arm name in sample_deviations")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915 - one linear measurement
    args = _parse(argv)
    if not args.views_roots and not args.indices:
        raise SystemExit("pass at least one --views-root or --index")
    if args.indices and args.index_state is None:
        raise SystemExit(
            "--index needs --index-state: an index.jsonl does not say which deposited chain its "
            "draws were aimed at, and a level NAMES A CHAIN"
        )
    if not args.tau > 0.0:
        raise SystemExit("--tau is an Angstrom distance and must be positive")
    if args.truncate_pool is not None and args.truncate_pool < 1:
        raise SystemExit("--truncate-pool must be at least 1")

    reference = load_kinase_rmsd_reference(args.artifact)
    required = sorted(reference.entries)
    restricted = None
    if args.smoke_systems:
        restricted = sorted({str(value) for value in args.smoke_systems})
        absent = sorted(set(restricted) - set(required))
        if absent:
            raise SystemExit(f"{args.artifact} does not carry the systems {absent}")
        required = restricted

    reports = [check_run_report(path, reference) for path in (args.run_reports or [])]

    sources: list[tuple[Path, tuple[str, ...]]] = []
    for root in args.views_roots or []:
        sources.extend(resolve_views(root, args.arm))
    if args.indices:
        states = STATES if args.index_state == "both" else (args.index_state,)
        sources.extend((path, states) for path in args.indices)

    draws: list[Draw] = []
    for path, states in sources:
        draws.extend(read_sample_index(path, states))
    if not draws:
        raise SystemExit(
            f"the {len(sources)} supplied sources carry no rows at all; there is nothing to score"
        )

    # The states actually supplied, in the objective's own ascending-level order. A --views-root
    # always supplies both; a bare --index-state dfg_in supplies one, which is a legitimate half of
    # the table and is recorded as such rather than refused -- and rather than silently reported as
    # though the pair had been measured. Coverage is still checked in FULL within every state that
    # was supplied: a half table over half the panel is two different denominators at once.
    states_scored = tuple(state for state in STATES if any(state in item for _, item in sources))

    pools = pooled_draws(draws)
    # Systems the union carries that this run is not scoring are dropped HERE and only under an
    # explicit --smoke-system, so that the coverage gate below sees exactly the population asked for
    # and a missing member of it is still a refusal.
    if restricted is not None:
        pools = {key: slot for key, slot in pools.items() if key[0] in set(required)}
    coverage = require_square_coverage(
        pools, required, states_scored, truncate_pool=args.truncate_pool
    )

    b_values = sorted({int(value) for value in (args.b_values or [coverage.depth])})
    too_deep = [value for value in b_values if value > coverage.depth or value < 1]
    if too_deep:
        raise SystemExit(
            f"--b {too_deep} cannot be taken from a pool {coverage.depth} draws deep; B is part of "
            "the metric and a cell reported at a B the pool does not hold would silently be a cell "
            "at the depth"
        )

    print(
        f"artifact:   {args.artifact} ({len(reference)} systems, metric_set "
        f"{reference.metric_set}, metric_objective {reference.metric_objective})",
        flush=True,
    )
    print(f"property:   {KINASE_RMSD_PROPERTY_NAME}", flush=True)
    print(f"sources:    {len(sources)} index files, {len(draws)} rows", flush=True)
    print(
        f"coverage:   {len(required)} systems x {len(states_scored)} states "
        f"{list(states_scored)} at depth {coverage.depth}",
        flush=True,
    )
    if restricted is not None:
        print(f"SMOKE:      restricted to {restricted}", flush=True)

    deviations = score_pools(pools, reference, required, states_scored)
    print(f"measured:   {len(deviations)} draw-to-chain distances", flush=True)

    cells = [
        success_cell(deviations, state=state, b=b, tau=tau)
        for state in states_scored
        for b in b_values
        for tau in (args.tau, None)
    ]

    sample_deviations = [
        {
            "sample_id": item.sequence_id,
            "reference_state": item.state,
            "rmsd": item.rmsd_angstrom,
            "sample_index": item.sample_index,
        }
        for item in deviations
    ]

    payload: dict[str, object] = {
        "schema": OUTPUT_SCHEMA,
        "label": args.label,
        "property_name": KINASE_RMSD_PROPERTY_NAME,
        "what": (
            "success@B on the held-out kinase_conformer_rmsd panel: the superposed N/CA/C/O RMSD "
            "of each draw against the DEPOSITED CHAIN its level named, over that kinase's own "
            "metric set, at a fixed tau and at ConforMix's adaptive 0.5*s_p. Reported PER "
            "REFERENCE STATE and never pooled, as an unweighted mean over SYSTEMS; draws are not "
            "pooled. The cells take the FIRST B draws -- exact at B = the pool depth. Below it, "
            "feed sample_deviations to rescore_expected_success.py, which averages over random "
            "B-subsets and reports ConforNets' subset_std"
        ),
        # BOTH halves of "complete": every system of the artifact's roster, and both of the
        # reference states. A table missing either is not the reported statistic, and the run exits
        # nonzero so that a pipeline cannot walk past it.
        "complete": restricted is None and states_scored == STATES,
        "restricted_to_systems": restricted,
        "states_scored": list(states_scored),
        "tau_angstrom": float(args.tau),
        "tau_source": (
            "ConforNets assets/config.yaml benchmarks.kinases.rmsd_threshold = 2.0"
            if args.tau == DEFAULT_TAU_ANGSTROM
            else "supplied on the command line"
        ),
        "arm": args.arm,
        "b_values": b_values,
        "truncate_pool": args.truncate_pool,
        "artifact": str(args.artifact),
        "artifact_content_fingerprint": reference.content_fingerprint(),
        "artifact_metric_set": reference.metric_set,
        "artifact_metric_objective": reference.metric_objective,
        "artifact_systems": len(reference),
        "artifact_provenance": dict(reference.provenance),
        "atom_set": list(BACKBONE_ATOMS),
        "reference_states": list(STATES),
        "state_entry_index": dict(STATE_ENTRY_INDEX),
        "state_level": dict(LEVEL_FOR_STATE),
        "sources": [
            {"index": str(path), "states": list(states)} for path, states in sorted(sources)
        ],
        "run_reports": reports,
        "coverage": coverage.as_dict(),
        "systems": {
            system: {
                "gene": reference.require(system).gene,
                "residues": len(reference.require(system).sequence),
                "metric_residues": len(reference.require(system).metric_indices),
                "separation_angstrom": float(reference.require(system).separation_angstrom),
                "adaptive_tau_angstrom": float(
                    reference.require(system).normalisation.adaptive_tau_angstrom
                ),
                "entry_ids": list(reference.require(system).entry_ids),
                "resolved_fraction": list(reference.require(system).resolved_fraction),
                "draws": {state: len(pools[(system, state)]) for state in states_scored},
            }
            for system in required
        },
        "cells": cells,
        # THE INPUT rescore_expected_success.py CONSUMES. One row per draw per state, keyed
        # ``sample_id`` / ``reference_state`` / ``rmsd`` exactly as that script reads them.
        "sample_deviations": {args.label: sample_deviations},
        "rescore_hint": (
            "python rescore_expected_success.py "
            f"--scored-json {args.output} --arm {args.label} --tau {args.tau} "
            "--b 5 --b 32 --b 100 --deviation-form rmsd --out <expected_success.json>"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.with_suffix(args.output.suffix + ".tmp")
    scratch.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    scratch.replace(args.output)

    draws_csv = args.draws_csv or args.output.with_name(f"{args.output.stem}_draws.csv")
    draws_csv.parent.mkdir(parents=True, exist_ok=True)
    with draws_csv.open("w", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_CSV_COLUMNS)
        for item in deviations:
            writer.writerow(
                [
                    item.sequence_id,
                    item.gene,
                    item.state,
                    STATE_ENTRY_INDEX[item.state],
                    LEVEL_FOR_STATE[item.state],
                    item.entry_id,
                    item.sample_index,
                    f"{item.rmsd_angstrom:.6f}",
                    f"{float(args.tau):.6f}",
                    f"{item.adaptive_tau_angstrom:.6f}",
                    f"{item.separation_angstrom:.6f}",
                    item.metric_residues,
                    item.metric_residues * len(BACKBONE_ATOMS),
                    int(item.rmsd_angstrom < float(args.tau)),
                    item.pdb_path,
                    item.index_path,
                ]
            )

    print(f"\nwrote {args.output}")
    print(f"wrote {draws_csv}")
    for cell in cells:
        label = "adaptive 0.5*s_p" if cell["adaptive_tau"] else f"tau={cell['tau_angstrom']}"
        low, high = cell["ci_low"], cell["ci_high"]
        interval = "[n/a, n/a]" if low is None else f"[{low:.3f}, {high:.3f}]"
        print(
            f"  {cell['reference_state']:<8s} B={cell['b']:<4d} {label:>17s}  "
            f"{cell['n_successes']}/{cell['n_systems']}  rate={cell['success_rate']:.3f}  "
            f"ci={interval}"
        )
    # Loud, and NONZERO. A panel statistic is a mean over SYSTEMS at both states, so a restricted
    # population or a single-state read is a different denominator and must not be walked past.
    reasons: list[str] = []
    if restricted is not None:
        reasons.append(f"scored {len(required)} of {len(reference)} systems under --smoke-system")
    if states_scored != STATES:
        reasons.append(f"scored only {list(states_scored)} of the states {list(STATES)}")
    if reasons:
        print(f"\nINCOMPLETE: {'; '.join(reasons)}. This is not a panel rate")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
