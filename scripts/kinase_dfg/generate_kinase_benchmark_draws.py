#!/usr/bin/env python3
"""Generate the STEERED benchmark draws, and their matched controls, for ``kinase_conformer_rmsd``.

SCOPE: KINASES, DIRECT METRIC. GPU. This is the generation half of the held-out benchmark for the
kinase-RMSD arm, and it is a port BY VALUE of ``scripts/domain_motion/generate_benchmark_draws.py``
onto the kinase pool. It computes no metric and reports no verdict: it writes structures and the
sidecars that say what each one is, and a scorer reads them afterwards, on CPU. A generator that
also scored could be run to a threshold.

WHY THE DOMAIN-MOTION SCRIPT IS THE PRECEDENT, AND NOT ``generate_held_out_draws.py``
--------------------------------------------------------------------------------------
``scripts/kinase_dfg/generate_held_out_draws.py`` looks closer -- same programme, same directory --
and is NOT usable here. It is hard-bound to the K-5b pre-registered ten-protein panel through
``kinase_dfg.outcome.PREREGISTERED_PANEL`` and refuses anything else, it requires a
``native_headroom.json`` that covers fourteen of these twenty genes, and it measures ``the progress coordinate``, the
DFG-coordinate property's dimensionless coordinate, rather than an Angstrom distance. Its
``load_intervention_site`` idiom is copied below, and nothing else of it is depended on. The
domain-motion script, by contrast, was built for exactly this shape of run -- success@B benchmark
draws with matched controls over a held-out panel -- against the same two-state direct-metric
construction this property uses.

WHAT ONE UNIT IS
-----------------
One ``(system, draw, level)`` steered draw, plus the UNSTEERED draw of the same system at the SAME
``draw_seed``, generated in the same process from the same noise. The two levels of one draw SHARE
that control -- ``level = -1`` names the Kincore ``DFGin``/``BLAminus`` chain and ``+1`` names the
``DFGout``/``BBAminus`` chain
(:func:`~sae_bioemu.kinase_rmsd.objective.reference_state_for_level`), and a control is a statement
about the noise, not about the reference -- so ``B`` draws cost ``2B`` steered plus ``B`` control
generations per system. That is the training path's own arrangement
(``kinase_rmsd.property._control_measurement``, ``required=True``), not an economy invented here,
and it is what makes a paired primary computable on these draws at all.

WHY THE REGISTRY ``evaluate`` PATH IS NOT USED
-----------------------------------------------
``KINASE_RMSD_PROPERTY.evaluator`` is ``None`` and ``run_workflow.py evaluate`` refuses this
property by name. This entry point needs no checkpoint-evaluation identity at all: it reads an
adapter, generates structures, and writes files. **It is in NO fingerprint group and moves none.**
Nothing under ``src/sae_bioemu/`` is touched by this script's existence.

WHAT IT REFUSES, AND WHY EACH ONE HAS BITTEN SOMETHING
-------------------------------------------------------
* **A non-kinase-RMSD adapter.** A level means a different thing on every property, and
  ``kinase_conformer_rmsd`` is deliberately a DISTINCT name from ``dfg_progress_coordinate``: the
  two are an Angstrom distance and a dimensionless coordinate, measured over different atom sets
  against different references.
* **An adapter with a site channel.** Nothing here can invent the per-residue field it was
  conditioned on; no kinase-RMSD arm has one --
  ``kinase_rmsd.artifact.KinaseRmsdDefinition`` deliberately has no ``arm`` attribute so the
  unconditioned path is taken byte-identically -- and that is asserted rather than assumed.
* **A form mismatch.** The reference's ``metric_objective`` must equal the form the checkpoint
  recorded: it decides the per-system target the LOSS descends, ``s_p`` under the MSE and ``tau_p``
  under the hinge, and therefore what the level a draw is generated at was ever asked to mean.
* **A metric-set mismatch.** THIS REFUSAL HAS NO DOMAIN-MOTION TWIN and is added rather than
  ported. Which residues the metric is read over is this objective's largest single decision:
  ``kinase_rmsd.objective.validated_metric_set`` has no default precisely because the shell
  candidates dilute the ~15-residue activation loop to about 12% of a ~125-residue union, and a
  reference written under ``activation_loop_shell12`` and read as ``activation_loop`` produces
  finite, plausible distances measured over ~110 residues that barely move between DFG-in and
  DFG-out. Two references over two index sets are two objectives, so a benchmark reference built
  under a set the checkpoint never descended is refused by name.
* **An uncached embedding.** ``bioemu.get_embeds`` falls OPEN on a miss and queries ColabFold
  MMseqs2 instead of failing, which on a compute node is a silent hang or a different MSA.
* **A steered draw indistinguishable from its own control.** A store full of controls passes every
  file-count check there is, so every steered member is compared against the C-alpha of its own
  matched control -- read back from the published ``.npz``, so a RESUMED run is checked too.

THE REFERENCE'S CONTENT FINGERPRINT IS DELIBERATELY NOT COMPARED, and this is the domain-motion
rule kept rather than relaxed. The benchmark reference covers the twenty HELD-OUT kinases alone
while the reference bound into the checkpoint covers the whole 61-gene pool the arm was resolved
over, so the two digests differ by construction -- that difference is what makes these twenty held
out -- and requiring equality would refuse the only artifact this run can use. What IS compared is
the pair of decisions that change what a distance MEANS: the objective form and the metric set.

WHAT THE ADAPTER IS CONDITIONED ON IS THE LEVEL, AND NOTHING ELSE. This arm is on the
RELATIVE-SEQUENCE path (``target_kind = relative_sequence`` in the checkpoint's own config), where
``ordinary_workflow._conditioning`` returns ``RequestedLevel(unit.level)``: the bounded
dimensionless dose request ``o``. The per-system constant ``system_constant_target`` builds --
``s_p`` under the MSE, ``tau_p`` under the hinge -- is the LOSS's target and reaches no generated
structure. The domain-motion script that this one is ported from conditioned on that constant in
its first draft and died 8 minutes into a CPU exercise with ``requested_level must lie in [-1, 1]``
after passing every other gate. The conditioning here is built through the shared
``RequestedLevel`` type so the bound is the workflow's own, and both numbers are recorded per
system so a reader can see which one the adapter saw.

THE SITE IS MULTI-SITE, AND THAT IS THE ONE STRUCTURAL DIVERGENCE FROM BOTH PRECEDENTS. The
``multisite8`` arms intervene at EIGHT score calls, so their run report carries the PLURAL
``resolved_timesteps`` and deliberately none of the scalar single-site keys
(``ResolvedInterventionSites.identity``). Both precedents read ``spec["resolved_timestep"]``
unconditionally and would die with a ``KeyError`` while rebuilding the site --- after the adapter
had loaded and before anything generated, but still on a run that had otherwise passed every gate.
:func:`load_intervention_site` reconstructs BOTH shapes, exactly as ``cli.py``'s steering demo
does, and the run report records which shape it read.

RESUME IS AT THE DRAW. A ``(system, draw, level)`` whose steered ``.pdb`` and ``.npz`` are both
published AND whose sidecar row exists is skipped entirely, and a control already on disk is not
regenerated. ``TrainingStructureStore`` appends and fsyncs each row as its batch lands, and the
per-level ``index.jsonl`` views are rebuilt from those rows after every system, so an interrupted
run is readable and the run is its own smoke. That contract is what makes a preemptible queue
admissible for the launcher: a preemption costs at most the one generation in flight.

SHARDING IS BY SYSTEM, and each shard writes its own view directory. The store is safe to share --
the sidecar is append-only, flushed and fsynced per batch, and every structure is published
atomically -- but the derived ``index.jsonl`` views are a pure function of the rows on disk that
every writer rebuilds, so two shards writing ONE view directory would interleave. Under
``--shards > 1`` the views land in ``views.shard<k>of<n>/`` over that shard's own systems only; the
whole-panel view is the concatenation of those files, which is valid because every row's paths are
relative to its own view root and the roots sit at one depth.

Usage::

    python scripts/kinase_dfg/generate_kinase_benchmark_draws.py \\
        --artifact <data-root>/kinase_dfg/out/kinase_holdout20_reference.pt \\
        --adapter  .../bioemu_kinase_conformer_rmsd_no_manifest_exploratory_adapter.pt \\
        --train-run-report .../bioemu_kinase_conformer_rmsd_..._train_run_report.json \\
        --cache-embeds-dir <data-root>/kinase_dfg/out/training_pool/embeds \\
        --cache-so3-dir data/processed/so3_cache \\
        --output-dir <data-root>/kinase_dfg/out/benchmark_draws

``--artifact`` is the kinase-RMSD REFERENCE that
:func:`~sae_bioemu.kinase_rmsd.artifact.load_kinase_rmsd_reference` reads, built offline from the
held-out panel by ``scripts/kinase_rmsd/build_kinase_rmsd_reference.py``. The panel JSON itself is
that builder's input and is not read here: the metric needs the two deposited chains' N/CA/C/O on
the folded sequence's own residue frame, which is a zip reader, an mmCIF parser and a global
alignment, and building it per run is the failure the artifact exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from sae_bioemu.bioemu.cryptic_generation import require_cached_embeddings, sequence_digests
from sae_bioemu.bioemu.extractor import BioEmuRuntimeActivationSampler, SequenceRecord
from sae_bioemu.bioemu.steering import SteeringSite
from sae_bioemu.bioemu.steering_properties import batch_residue_mask
from sae_bioemu.core.config import TraceConfig, seed_everything
from sae_bioemu.kinase_rmsd.artifact import load_kinase_rmsd_reference
from sae_bioemu.kinase_rmsd.objective import (
    KINASE_RMSD_PROPERTY_NAME,
    REFERENCE_STATES,
    reference_state_for_level,
)
from sae_bioemu.kinase_rmsd.property import system_target_constants
from sae_bioemu.steering.adapter_io import adapter_attributes, load_adapter
from sae_bioemu.steering.ordinary_workflow import RequestedLevel
from sae_bioemu.steering.schedule import SteeringUnit, system_constant_target
from sae_bioemu.steering.training_structures import (
    CONTROL_DIRECTORY,
    SIDECAR_NAME,
    SteeredDrawRecorder,
    TrainingStructureStore,
    record_steered_draws,
)

#: The two levels a kinase-RMSD adapter was ever presented with, ascending. Derived from
#: :data:`~sae_bioemu.kinase_rmsd.objective.REFERENCE_STATES` rather than restated, so the set of
#: legal levels here and the set the objective will answer for cannot become two. Each NAMES A
#: DEPOSITED, KINCORE-LABELLED CHAIN rather than a dose, so nothing between them is generated:
#: there is no third crystal for an intermediate level to be measured against, and
#: ``reference_state_for_level`` refuses ``0.0`` by design.
TRAINED_LEVELS: tuple[float, ...] = tuple(sorted(float(level) for level in REFERENCE_STATES))

#: Draws per system. 32 is the count every paired kinase measurement in this programme is reported
#: at -- ``measure_paired_gap.py`` and ``measure_aligned_paired_gap.py`` both draw 32 shared noise
#: seeds per protein -- and it is the domain-motion benchmark's own ``B`` as well, so a steered
#: success@B here is the same statistic at the same ``B`` as the one reported there.
BENCHMARK_DRAWS = 32

#: The split role every row is written under. These twenty kinases are the arm's own declared
#: HOLDOUT role (``split.test`` in the training run report) and appear in no training batch, but
#: the rows written here are a BENCHMARK generation rather than a training run's validation replay,
#: so they carry a role of their own. A reader must ask for it explicitly, which is what stops a
#: benchmark row ever being counted as a validation draw.
BENCHMARK_SPLIT_ROLE = "benchmark"

#: Added once per level ORDINAL to build a steered unit's identity seed. The unit seed is only a
#: STEM -- ``sampler.steer`` is given the draw seed -- so this separates the two levels' files
#: without touching the noise either of them draws. Far outside any plausible
#: ``draw_seed_base + index`` range, so the two namespaces cannot meet.
LEVEL_SEED_STRIDE = 1_000_000

#: Below this maximum absolute C-alpha difference, in Angstrom, a "steered" draw is indistinguishable
#: from its own matched control and the run is measuring the native ensemble under a steered name.
#: Four orders of magnitude below any structural difference: a "these are not the same structure"
#: test, not a "the adapter did something useful" one, and deliberately not a dose check.
STEERED_DIFFERENCE_FLOOR_ANGSTROM = 1.0e-4

RUN_REPORT_SCHEMA = "kinase_rmsd.benchmark_draws.v1"


def level_tag(level: float) -> str:
    """Return the directory-safe name of a level: ``dfg_in`` for ``-1``, ``dfg_out`` for ``+1``.

    The STATE NAME, not a sign mnemonic, and that is a deliberate departure from the domain-motion
    original's ``m1``/``p1``. There the two conformers are an arbitrary member of a DynDom pair and
    only a sign distinguishes them; here both chains carry a Kincore LABEL, so the directory can be
    the answer rather than a code a reader has to look up. It delegates to the objective's own
    mapping, so ``0.0`` is refused here for exactly the reason it is refused there.
    """
    return reference_state_for_level(float(level))


def unit_seed(draw_seed: int, level: float) -> int:
    """Return this ``(draw, level)``'s STEM seed. Unique per level; never the noise seed.

    The ordinal comes from :data:`TRAINED_LEVELS` rather than from the ``--level`` arguments, so a
    run restricted to one level produces the SAME stems as the full run and resumes into it.
    """
    ordinal = TRAINED_LEVELS.index(float(level))
    return int(draw_seed) + (ordinal + 1) * LEVEL_SEED_STRIDE


def chunks(values: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def load_intervention_site(report_path: Path) -> tuple[SteeringSite, dict[str, object]]:
    """Rebuild the TRAINING site from the run report beside the checkpoint, never re-resolve it.

    Re-resolving from ``(timestep, layer)`` recomputes the denoiser's grid and could land on a
    neighbouring score call if any sampler setting differed. An adapter evaluated at a site it was
    not trained at is a different experiment that reports a complete-looking number. Copied by value
    from ``scripts/domain_motion/generate_benchmark_draws.py``, which copied it from
    ``scripts/kinase_dfg/generate_held_out_draws.py``.

    **BOTH RECORD SHAPES, which neither precedent handles.** A multi-site arm's identity carries the
    plural ``resolved_timesteps`` and deliberately NONE of the scalar single-site keys, so that a
    reader cannot mistake it for a single-site record by finding a call index that describes only
    one of the calls actually intervened on. Every kinase-RMSD arm in flight is ``multisite8``, so
    the singular lookup both precedents perform would raise ``KeyError: 'resolved_timestep'`` here.
    ``score_calls`` is already plural in both shapes. This is the same reconstruction ``cli.py``'s
    steering demo performs, and the count is returned to the report so a reader can see how many
    calls the adapter was applied at.
    """
    payload = json.loads(report_path.read_text())
    # A shipped model's sidecar nests the site under ``training``; a raw training report does not.
    spec = payload.get("intervention_site") or (payload.get("training") or {}).get(
        "intervention_site"
    )
    if not isinstance(spec, dict):
        raise SystemExit(
            f"{report_path} carries no intervention_site block, so the site the checkpoint was "
            "trained at is unknown and cannot be reproduced"
        )
    if "resolved_timesteps" in spec:
        expected_timesteps = tuple(float(value) for value in spec["resolved_timesteps"])
    elif "resolved_timestep" in spec:
        expected_timesteps = (float(spec["resolved_timestep"]),)
    else:
        raise SystemExit(
            f"{report_path}'s intervention_site names neither resolved_timesteps nor "
            "resolved_timestep, so the timesteps the selected score calls must be observed at are "
            "unknown. Generating without them would let a wrong call index pass silently"
        )
    site = SteeringSite(
        module_path=str(spec["module_path"]),
        score_calls=tuple(int(value) for value in spec["score_calls"]),
        expected_timesteps=expected_timesteps,
        expected_score_call_total=int(spec["expected_score_call_total"]),
        timestep_tolerance=float(spec["timestep_tolerance"]),
    )
    return site, spec


def checkpoint_arm_block(payload: object) -> dict[str, object]:
    """Return the ``property_arm`` identity block the checkpoint recorded, or an empty mapping."""
    config = payload.get("adapter_config") if isinstance(payload, dict) else None
    arm = config.get("property_arm") if isinstance(config, dict) else None
    return dict(arm) if isinstance(arm, dict) else {}


def read_sidecar_rows(store: TrainingStructureStore) -> dict[tuple[str, int], dict[str, object]]:
    """Every steered-draw row on disk, collapsed on ``(sequence_id, unit_seed)``.

    Collapsed because a resumed ``record_batch`` re-appends a row for a stem it skipped, so the file
    is append-only and may carry a key twice. A line torn by a preemption mid-write is DROPPED and
    its draw replayed, which is cheaper and safer than trusting a truncated record.
    """
    rows: dict[tuple[str, int], dict[str, object]] = {}
    if not store.sidecar_path.exists():
        return rows
    for raw in store.sidecar_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rows[(str(row["sequence_id"]), int(row["unit_seed"]))] = row
    return rows


def published_units(store: TrainingStructureStore) -> set[tuple[str, int]]:
    """Every ``(sequence_id, unit_seed)`` whose row exists AND whose two files are both on disk.

    Read from the SIDECAR rather than from the pass directories, because the sidecar is what names a
    row's level and because a regenerated unit lands in ``pass_01``: the row records where its own
    structure went. A row whose files are absent is NOT counted as done -- that is the torn state a
    preemption leaves -- so the draw is replayed.
    """
    done: set[tuple[str, int]] = set()
    for key, row in read_sidecar_rows(store).items():
        pdb = store.root / str(row["steered_pdb"])
        npz = store.root / str(row["steered_npz"])
        if pdb.is_file() and npz.is_file():
            done.add(key)
    return done


def control_paths(
    store: TrainingStructureStore, sequence_id: str, draw_seed: int
) -> tuple[Path, Path]:
    pdb, npz = store.control_entry_paths(sequence_id, int(draw_seed))
    return store.root / pdb, store.root / npz


def control_published(store: TrainingStructureStore, sequence_id: str, draw_seed: int) -> bool:
    pdb, npz = control_paths(store, sequence_id, draw_seed)
    return pdb.is_file() and npz.is_file()


def control_positions_nm(
    store: TrainingStructureStore, sequence_id: str, draw_seed: int, sequence: str
) -> torch.Tensor:
    """Read one published control's C-alpha positions back, in nanometres.

    Read from DISK rather than kept from the generating call, so the steered-versus-control check
    runs identically on a fresh draw and on a resumed one. A gate a resume silently skips is a gate
    that reports success when it never ran.
    """
    _, npz = control_paths(store, sequence_id, draw_seed)
    with np.load(npz, allow_pickle=True) as payload:
        stored = torch.from_numpy(payload["pos"]).to(torch.float64)
        stored_sequence = str(payload["sequence"])
    if stored_sequence != sequence:
        raise SystemExit(
            f"{npz} carries a different sequence from the one this run folds for {sequence_id}; "
            "the control and the steered draw would not be the same protein"
        )
    if tuple(stored.shape) != (len(sequence), 3):
        raise SystemExit(
            f"{npz} holds {tuple(stored.shape)} positions for a {len(sequence)}-residue chain"
        )
    return stored


def write_index_view(destination: Path, store_root: Path, rows: Sequence[dict[str, object]]) -> int:
    """Write one ``final_structures/index.jsonl`` naming a chosen subset of persisted draws.

    The shared success@B readers -- ``cryptic_steering.success_at_b.read_sample_index`` and its
    domain-motion twin -- resolve each ``pdb_path`` against ``index.parent.parent``. This writes
    exactly that shape over an ARBITRARY subset, which is what lets ONE store serve three reads --
    level -1, level +1 and the adapter-off control -- without copying a structure or generating one
    twice. There is no kinase-RMSD scorer yet; writing the views in the shared shape is what lets
    one be pointed at this store rather than taught a fourth layout.

    Paths are RELATIVE, from the view's own root to the real file, so a view stays valid wherever the
    run directory is moved to as a whole. Nothing is symlinked: a symlink farm is a second thing that
    can rot, and it would hide which structure a row actually names.

    ``sample_index`` is the DRAW index ``0..B-1``, not the unit seed. A success@B reader takes the
    first ``B`` draws in ``sample_index`` order, so the two levels and the control must be numbered
    on ONE ladder or "the best of 32" would mean a different 32 on each.

    The scratch file carries this PROCESS's pid. One shard writes one view directory, so the name
    cannot collide today; it is pid-qualified anyway because a shared scratch name is how two
    writers come to interleave into one file and publish the result atomically.
    """
    root = destination.parent.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    written = 0
    with scratch.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (str(item["sequence_id"]), int(item["draw"]))):
            target = store_root / str(row["pdb"])
            if not target.is_file():
                raise SystemExit(
                    f"{destination}: {target} is named by a sidecar row but is not on disk. A view "
                    "that promised a structure it does not have would fail inside the scorer, "
                    "after the GPU time was already spent"
                )
            handle.write(
                json.dumps(
                    {
                        "sequence_id": str(row["sequence_id"]),
                        "sequence": str(row["sequence"]),
                        "sample_index": int(row["draw"]),
                        "pdb_path": os.path.relpath(target, root),
                        "npz_path": os.path.relpath(store_root / str(row["npz"]), root),
                        "pos_units": "nanometers",
                        "pdb_units": "angstrom_centered",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    scratch.replace(destination)
    return written


def build_views(
    store: TrainingStructureStore,
    views_root: Path,
    sequences: dict[str, str],
    visible: Sequence[str],
    levels: Sequence[float],
    draw_seed_base: int,
) -> dict[str, dict[str, object]]:
    """Rebuild every per-level and the control ``index.jsonl`` from the store's own sidecar.

    DERIVED, never accumulated: a view is a pure function of the rows on disk, so an interrupted or
    resumed run's views describe exactly what it has rather than what it once appended.

    ``sequences`` is the WHOLE reference's map, so a row this process did not write is still
    resolvable. ``visible`` is which systems this writer OWNS: the whole panel for an unsharded run,
    so a smoke that names one system does not silently drop the other nineteen's existing rows out
    of the views, and this shard's systems alone when ``--shards > 1``, because two shards rebuilding
    one view directory from one shared sidecar would race on it.
    """
    rows = read_sidecar_rows(store)
    unknown = sorted({str(row["sequence_id"]) for row in rows.values()} - set(sequences))
    if unknown:
        raise SystemExit(
            f"the store at {store.root} carries rows for systems the reference does not name: "
            f"{unknown[:10]}. A view built over them would score a draw against a reference that "
            "is not this artifact's"
        )
    owned = set(visible)
    mine = [row for row in rows.values() if str(row["sequence_id"]) in owned]

    views: dict[str, dict[str, object]] = {}
    for level in levels:
        tag = level_tag(level)
        selected = [
            {
                "sequence_id": row["sequence_id"],
                "sequence": sequences[str(row["sequence_id"])],
                "draw": int(row["draw_seed"]) - draw_seed_base,
                "pdb": row["steered_pdb"],
                "npz": row["steered_npz"],
            }
            for row in mine
            if float(row["level"]) == float(level)
        ]
        destination = views_root / f"level_{tag}" / "final_structures" / "index.jsonl"
        views[f"level_{tag}"] = {
            "index": str(destination),
            "draws": write_index_view(destination, store.root, selected),
            "level": float(level),
            "reference_state": tag,
            # Which member of ``KinaseRmsdEntry.entry_ids`` this arm was aimed at: that tuple is
            # ``(dfg_in, dfg_out)``, so level -1 is index 0. Recorded rather than left implicit,
            # because a scorer that paired a level with the wrong deposited chain would report a
            # complete, finite, inverted number.
            "reference_entry_index": 0 if tag == "dfg_in" else 1,
        }

    # ONE row per (sequence, draw_seed): the two levels SHARE a control, so collapsing on that key
    # is what stops the adapter-off view counting every control twice and reporting B = 64 where 32
    # structures exist.
    control_rows = {
        (str(row["sequence_id"]), int(row["draw_seed"])): {
            "sequence_id": row["sequence_id"],
            "sequence": sequences[str(row["sequence_id"])],
            "draw": int(row["draw_seed"]) - draw_seed_base,
            "pdb": row["control_pdb"],
            "npz": row["control_npz"],
        }
        for row in mine
        if bool(row["control_present"])
    }
    destination = views_root / "control" / "final_structures" / "index.jsonl"
    views["control"] = {
        "index": str(destination),
        "draws": write_index_view(destination, store.root, list(control_rows.values())),
        "level": None,
        "reference_state": None,
        "reference_entry_index": None,
    }
    return views


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="the kinase-RMSD REFERENCE over the held-out panel, from "
        "scripts/kinase_rmsd/build_kinase_rmsd_reference.py",
    )
    parser.add_argument("--adapter", type=Path, required=True, help="the trained .pt checkpoint")
    parser.add_argument(
        "--train-run-report",
        type=Path,
        required=True,
        help="the training run report the intervention site is read from, never re-resolved",
    )
    parser.add_argument("--cache-embeds-dir", type=Path, required=True)
    parser.add_argument("--cache-so3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=BENCHMARK_DRAWS)
    parser.add_argument("--draw-seed-base", type=int, default=20260919)
    parser.add_argument(
        "--level",
        action="append",
        type=float,
        default=None,
        dest="levels",
        help=f"repeatable; defaults to both of {list(TRAINED_LEVELS)}",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="this shard's index. Sharding is BY SYSTEM: the store is shared and safe to share, "
        "and each shard writes its own views.shard<k>of<n>/ directory",
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        dest="sequences",
        help=(
            "restrict to these benchmark systems. SMOKE ONLY: a panel statistic is a mean over "
            "SYSTEMS, so a restricted run is recorded as incomplete in its own report and its rate "
            "would be over a different denominator"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=8,
        help=(
            "batches between torch.cuda.empty_cache(); the steered path retains a graph from the "
            "first intervened score call onward, and a multisite8 arm intervenes at eight"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915 - one linear measurement
    args = _parse(argv)
    levels = tuple(float(value) for value in (args.levels or TRAINED_LEVELS))
    unknown = [value for value in levels if value not in TRAINED_LEVELS]
    if unknown:
        raise SystemExit(
            f"--level must name a deposited chain, one of {list(TRAINED_LEVELS)}; received "
            f"{unknown}. -1 is the Kincore DFGin/BLAminus chain and +1 the DFGout/BBAminus chain; "
            "the metric is an unsigned distance to a deposited structure and there is no third "
            "structure for an intermediate level to be measured against"
        )
    if len(set(levels)) != len(levels):
        raise SystemExit("--level was repeated; each level is generated once per draw")
    if args.draws < 1 or args.batch_size < 1:
        raise SystemExit("--draws and --batch-size must be positive")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise SystemExit("--shards must be positive and --shard must lie in [0, --shards)")

    artifact = load_kinase_rmsd_reference(args.artifact)
    all_sequences = {name: entry.sequence for name, entry in artifact.entries.items()}
    wanted = sorted(artifact.entries)
    if args.sequences:
        requested = [str(value) for value in args.sequences]
        absent = sorted(set(requested) - set(wanted))
        if absent:
            raise SystemExit(f"{args.artifact} does not carry the systems {absent}")
        wanted = [name for name in wanted if name in set(requested)]
    # Strided, not blocked, so that a shard's systems are spread across the length distribution
    # rather than being this panel's shortest or longest run.
    sharded = wanted[args.shard :: args.shards] if args.shards > 1 else list(wanted)
    complete_panel = len(sharded) == len(artifact.entries)
    records = {
        name: SequenceRecord(sequence_id=name, sequence=all_sequences[name]) for name in sharded
    }
    if not sharded:
        raise SystemExit(
            f"shard {args.shard} of {args.shards} names no systems out of {len(wanted)}; a shard "
            "that generates nothing would report success without producing a draw"
        )

    adapter, payload = load_adapter(args.adapter)
    attributes = adapter_attributes(payload)
    if str(attributes["property_name"]) != KINASE_RMSD_PROPERTY_NAME:
        raise SystemExit(
            f"{args.adapter} is a {attributes['property_name']!r} adapter, not "
            f"{KINASE_RMSD_PROPERTY_NAME!r}; a level means a different thing on every property, "
            "and 'dfg_progress_coordinate' in particular is a dimensionless coordinate rather "
            "than an Angstrom distance"
        )
    if attributes.get("site_conditioning") is not None:
        raise SystemExit(
            f"{args.adapter} declares site_conditioning={attributes['site_conditioning']!r}. No "
            "kinase-RMSD arm has a site channel -- KinaseRmsdDefinition carries no `arm` attribute "
            "precisely so the unconditioned path is taken byte-identically -- so this checkpoint "
            "is not one this script can generate from without inventing the per-residue field it "
            "was trained on"
        )
    arm_block = checkpoint_arm_block(payload)
    trained_form = arm_block.get("metric_objective")
    if trained_form is not None and str(trained_form) != artifact.metric_objective:
        raise SystemExit(
            f"{args.adapter} was trained under metric_objective {trained_form!r} and "
            f"{args.artifact} declares {artifact.metric_objective!r}. The form decides the "
            "per-system target the LOSS descends -- s_p under the MSE, tau_p under the hinge -- "
            "and the level that selects a deposited chain means a different distance under each"
        )
    trained_set = arm_block.get("metric_set")
    if trained_set is not None and str(trained_set) != artifact.metric_set:
        raise SystemExit(
            f"{args.adapter} was trained over metric_set {trained_set!r} and {args.artifact} "
            f"declares {artifact.metric_set!r}. The metric set names WHICH RESIDUES the distance "
            "and its per-system normalisation s_p are read over: the shell sets dilute the "
            "activation loop to about 12% of the union, so a benchmark measured over one set "
            "against an adapter trained on another is a finite, plausible number measured over "
            "~110 residues that barely move between DFG-in and DFG-out"
        )
    # THE PER-SYSTEM CONSTANT REACHES NO GENERATED STRUCTURE. The loss's target is
    # ``system_constant_target(...)`` in Angstrom, and conditioning the adapter on it is what the
    # first draft of the domain-motion original did: the adapter refused it outright with
    # ``requested_level must lie in [-1, 1]``, 8 minutes into a run that had otherwise passed every
    # gate above. This arm records ``target_kind = relative_sequence``, so
    # ``ordinary_workflow._conditioning`` returns ``RequestedLevel(unit.level)`` -- the bounded
    # dimensionless dose request ``o``, and nothing else. So the conditioning is the LEVEL.
    #
    # The table is still built, and it is built HERE rather than in the report block, because
    # ``system_target_constants`` re-derives both numbers from the reference's own coordinates and
    # refuses a non-positive one: it is a gate on the reference that costs nothing and would catch
    # a kinase whose s_p was zero before 1920 generations were spent on it. It is RECORDED as
    # provenance and read by nothing downstream.
    constants = system_target_constants(artifact)
    loss_targets = {
        name: {level: system_constant_target(constants.require(name), level) for level in levels}
        for name in sharded
    }
    # Through the shared type, so the bound is the workflow's own and a future change to what the
    # adapter is conditioned on is a change this line stops compiling against rather than a silent
    # divergence between training and this path.
    conditioning_for = {level: RequestedLevel(float(level)).value for level in levels}

    site, site_spec = load_intervention_site(args.train_run_report)
    train_report = json.loads(args.train_run_report.read_text())
    device = torch.device(args.device)
    adapter.to(device)
    adapter.eval()

    require_cached_embeddings(
        args.cache_embeds_dir, {name: records[name].sequence for name in sharded}
    )
    print(f"embedding cache:   all {len(sharded)} sequences covered", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store_root = args.output_dir / "store"
    seed_everything(args.draw_seed_base)
    trace = TraceConfig(cache_so3_dir=str(args.cache_so3_dir), save_final_structures=True)
    sampler = BioEmuRuntimeActivationSampler(
        trace,
        cache_embeds_dir=str(args.cache_embeds_dir),
        # The sampler's OWN writer only ever sees the adapter-off path: ``steer`` never persists a
        # structure. So this directory holds the matched CONTROLS, exactly as the training path
        # arranges it, and the steered half is written by the store's recorder below.
        structures_dir=str(store_root / CONTROL_DIRECTORY),
        batch_size=1,
        device=args.device,
    )
    sampler.assert_runtime_assets_unchanged()
    store = TrainingStructureStore(store_root, property_name=KINASE_RMSD_PROPERTY_NAME)
    recorder = SteeredDrawRecorder(sampler, store)
    views_root = args.output_dir / (
        "views" if args.shards == 1 else f"views.shard{args.shard}of{args.shards}"
    )

    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "python": platform.python_version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "precision": sampler.precision.mode,
        "denoiser_type": trace.denoiser_type,
        "model_name": trace.model_name,
        "runtime_assets": sampler.runtime_asset_provenance,
        "requested_device": str(args.device),
        # get_device_name reports the GPU the job was ALLOCATED, not the device the score model ran
        # on, and those differ exactly when the run is silently on CPU.
        "score_model_device": "not yet generated",
    }
    print(f"environment:       {json.dumps(environment, sort_keys=True)}", flush=True)
    print(
        f"site:              {site.module_path} at {len(site.score_calls)} score calls "
        f"{list(site.score_calls)}",
        flush=True,
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    draws = [args.draw_seed_base + index for index in range(args.draws)]
    # Named before the loop, not inside it, so the closing lines name a file even when the loop
    # raised before its first system completed.
    report_path = args.output_dir / f"kinase_benchmark_draws.shard{args.shard}of{args.shards}.json"
    done = published_units(store)
    if done:
        print(f"resume:            {len(done)} steered draws already recorded", flush=True)

    per_system: list[dict[str, object]] = []
    started_all = time.monotonic()
    batches = 0
    generations = 0
    for position, name in enumerate(sharded, start=1):
        entry = artifact.require(name)
        record = records[name]
        started = time.monotonic()
        controls_generated = 0
        steered_generated = 0
        deviations: list[float] = []

        # THE CONTROLS FIRST, for the whole system. Their files are what ``control_present`` on
        # every steered row asserts, so generating them afterwards would write rows that claim a
        # counterfactual before it exists.
        pending_controls = [seed for seed in draws if not control_published(store, name, seed)]
        for chunk in chunks(pending_controls, args.batch_size):
            members = [record] * len(chunk)
            sampler.generate_unsteered(
                members if len(chunk) > 1 else record,
                seed=chunk if len(chunk) > 1 else chunk[0],
            )
            controls_generated += len(chunk)
            generations += len(chunk)
        missing = [seed for seed in draws if not control_published(store, name, seed)]
        if missing:
            raise SystemExit(
                f"{name}: {len(missing)} controls were generated but not published to "
                f"{store.root / CONTROL_DIRECTORY}; the paired primary is undefined without them "
                "and the unpaired threshold metric is not a fallback -- success@tau is unwinnable "
                "at the measured alignment on this programme's pools"
            )

        # THE STEERED DRAWS, one level at a time. The batch dimension is over DRAW SEEDS and never
        # over levels. Batching the two levels of one draw would be the natural shape and it is
        # WRONG on the shared path as well as unnecessary here: ``_steering_members`` checks
        # ``(sequence, seed, conditioning)`` for uniqueness, and two members of one sequence at one
        # seed are distinguished only by their conditioning -- which is the level, so a two-level
        # batch is legal but produces no saving, while a two-level batch under a per-system
        # CONSTANT conditioning (the shape the domain-motion original first had) is two identical
        # triples and is refused. Over draw seeds every member differs in the one field that
        # decides the noise.
        for level in levels:
            conditioning = float(conditioning_for[level])
            pending = [seed for seed in draws if (name, unit_seed(seed, level)) not in done]
            for chunk in chunks(pending, args.batch_size):
                members = [record] * len(chunk)
                units = tuple(
                    SteeringUnit(
                        split=BENCHMARK_SPLIT_ROLE,
                        sequence_id=name,
                        cycle=0,
                        visit=seed - args.draw_seed_base,
                        level=float(level),
                        seed=unit_seed(seed, level),
                        draw_seed=seed,
                    )
                    for seed in chunk
                )
                target = torch.tensor(
                    [conditioning] * len(chunk), dtype=torch.float32, device=device
                )
                mask = batch_residue_mask(members, device=device)
                with record_steered_draws(recorder, units, (conditioning,) * len(chunk)):
                    sample = recorder.steer(
                        members if len(chunk) > 1 else record,
                        seed=chunk if len(chunk) > 1 else chunk[0],
                        site=site,
                        adapter=adapter,
                        target=target,
                        residue_mask=mask,
                        site_field=None,
                    )
                steered = sample.coordinates.detach().to("cpu").to(torch.float64)
                del sample
                length = len(record.sequence)
                for index, seed in enumerate(chunk):
                    reference = control_positions_nm(store, name, seed, record.sequence)
                    deviation = float((steered[index, :length] - reference).abs().max() * 10.0)
                    deviations.append(deviation)
                    if deviation <= STEERED_DIFFERENCE_FLOOR_ANGSTROM:
                        raise SystemExit(
                            f"{name} draw_seed {seed} level {level}: the steered structure is "
                            f"indistinguishable from its own matched control, max |delta| = "
                            f"{deviation:.3e} Angstrom. Either the adapter did nothing at all or a "
                            "control was generated under a steered name, and both make every "
                            "number downstream a measurement of the native ensemble"
                        )
                del steered
                batches += 1
                steered_generated += len(chunk)
                generations += len(chunk)
                if (
                    args.empty_cache_every > 0
                    and batches % args.empty_cache_every == 0
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

        sampler.assert_runtime_assets_unchanged()
        environment["score_model_device"] = str(next(sampler.score_model.parameters()).device)
        elapsed = time.monotonic() - started
        produced = controls_generated + steered_generated
        published = published_units(store)
        per_system.append(
            {
                "sequence_id": name,
                "gene": entry.gene,
                "residue_count": len(record.sequence),
                "separation_angstrom": float(entry.separation_angstrom),
                "metric_residues": len(entry.metric_indices),
                # Per-chain fraction of the A-loop span that survived the
                # all-four-atoms-in-both-chains restriction, carried so a kinase whose loop is half
                # disordered is a reported fact rather than a short index list nobody looked at.
                "resolved_fraction": list(entry.resolved_fraction),
                "entry_ids": list(entry.entry_ids),
                # What the LOSS's target would be for this system, in Angstrom, under the
                # reference's own form. Provenance: it reaches no generated structure here, and
                # ``adapter_conditioning`` below is what the adapter actually saw.
                "loss_target_constant_angstrom": {
                    str(level): float(loss_targets[name][level]) for level in levels
                },
                "adapter_conditioning": {
                    str(level): float(conditioning_for[level]) for level in levels
                },
                "draws_requested": len(draws),
                "controls_generated": controls_generated,
                "controls_published": sum(
                    1 for seed in draws if control_published(store, name, seed)
                ),
                "steered_generated": steered_generated,
                "steered_requested": len(draws) * len(levels),
                "steered_published": sum(1 for key in published if key[0] == name),
                "max_steered_minus_control_angstrom": max(deviations) if deviations else None,
                "min_steered_minus_control_angstrom": min(deviations) if deviations else None,
                "seconds": elapsed,
                "seconds_per_generation": elapsed / max(1, produced),
            }
        )
        print(
            f"  [{position}/{len(sharded)}] {name:14s} L={len(record.sequence):4d}  "
            f"{controls_generated:3d} control + {steered_generated:3d} steered in {elapsed:7.1f}s "
            f"({elapsed / max(1, produced):.2f} s/generation)",
            flush=True,
        )

        # Published as each system completes, not at the end. A derived summary that only exists
        # after the last system is a summary a preemption destroys.
        views = build_views(store, views_root, all_sequences, sharded, levels, args.draw_seed_base)
        report = {
            "schema": RUN_REPORT_SCHEMA,
            "what": (
                "steered BioEmu draws for the held-out kinase_conformer_rmsd benchmark systems, "
                "with the matched unsteered control of each draw at the SAME draw_seed. Structures "
                "only: no metric is computed here"
            ),
            "commit": commit,
            "complete": complete_panel and len(per_system) == len(sharded),
            "restricted_to_sequences": list(args.sequences) if args.sequences else None,
            "shard": args.shard,
            "shards": args.shards,
            "systems": list(sharded),
            "artifact": str(args.artifact),
            "artifact_content_fingerprint": artifact.content_fingerprint(),
            "artifact_metric_objective": artifact.metric_objective,
            "artifact_metric_set": artifact.metric_set,
            "artifact_systems": len(artifact),
            "artifact_provenance": dict(artifact.provenance),
            "adapter": str(args.adapter),
            "adapter_attributes": attributes,
            "adapter_checkpoint_identity": payload.get("checkpoint_identity_fingerprint"),
            "adapter_epoch": payload.get("epoch"),
            "adapter_unit_cursor": payload.get("unit_cursor"),
            "adapter_complete": payload.get("complete"),
            # NO VALIDATION HISTORY, and that is the arm's declared configuration rather than a
            # missing file: this property runs ``model_selection: none`` because the pool is 41
            # train / 0 validation / 20 holdout, so there are no ``.best-N`` sidecars to look for
            # and the checkpoint here is the terminal one.
            "adapter_model_selection": train_report.get(
                "model_selection", (train_report.get("training") or {}).get("model_selection")
            ),
            # The TRAINING reference's identity, carried beside this run's. They differ by
            # construction -- this one is the twenty held-out genes, that one the whole pool -- and
            # a reader has to be able to see both to know that they should.
            "adapter_trained_arm": arm_block,
            "train_run_report": str(args.train_run_report),
            "train_split": train_report.get("split"),
            "intervention_site": site_spec,
            "intervention_site_shape": ("multi" if "resolved_timesteps" in site_spec else "single"),
            "intervention_score_calls": len(site.score_calls),
            "draws_per_system": len(draws),
            "draw_seed_base": args.draw_seed_base,
            "draw_seeds": [draws[0], draws[-1]],
            "levels": list(levels),
            "level_states": {str(level): reference_state_for_level(level) for level in levels},
            "level_seed_stride": LEVEL_SEED_STRIDE,
            "split_role": BENCHMARK_SPLIT_ROLE,
            "batch_size": args.batch_size,
            "control_kind": "matched_unsteered_draw",
            "steered_difference_floor_angstrom": STEERED_DIFFERENCE_FLOOR_ANGSTROM,
            "store": store.identity(),
            "sidecar": str(store.root / SIDECAR_NAME),
            "views": views,
            "warm_up": sampler.warm_up,
            "environment": environment,
            "sequence_digests": {
                name: sequence_digests([records[name].sequence])[0] for name in sharded
            },
            "systems_detail": per_system,
            "generations_this_run": generations,
            "seconds_total": time.monotonic() - started_all,
        }
        scratch = report_path.with_suffix(".json.tmp")
        scratch.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        scratch.replace(report_path)

    print(f"\nwrote {report_path}")
    print(f"store {store.root}")
    short = [
        str(item["sequence_id"])
        for item in per_system
        if int(item["steered_published"]) != int(item["steered_requested"])
        or int(item["controls_published"]) != int(item["draws_requested"])
    ]
    if short:
        # Loud, and NONZERO. A panel statistic is a mean over SYSTEMS, so a system short of its
        # draws is a different denominator; per-shard success does not compose into panel coverage.
        print(f"INCOMPLETE: {len(short)} systems short of their draws: {short}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
