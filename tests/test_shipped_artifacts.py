"""Contract for what a fresh clone can actually run.

An audit of the working repository found three defects that had survived a merge untouched: both
shipped example configs pointed ``project_config`` at a file that does not exist, the README told
people to install a ``bioemu`` extra that was never declared, and it documented a checkpoint path
no config produces. All three were invisible because **nothing tested the shipped artifacts**.

These tests are cheap, need no GPU and no network. They check the three things a reviewer's clone
depends on: every shipped model matches its sidecar, every shipped model loads and runs, and every
path the release resolves is inside the repository.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


#: Directories a reader creates by running the release; never part of what shipped.
_LOCAL = {".venv", "venv", "outputs", "examples", "__pycache__", ".pytest_cache", ".ruff_cache",
          "build", "dist", ".git"}  # fmt: skip


def shipped_files() -> set[str]:
    """What this copy of the release contains, as repository-relative paths.

    From git where there is one. A copy downloaded as an archive has no git metadata -- the anonymous
    mirror serves exactly that -- and there, what is present IS what shipped, less anything a reader
    has since created by running it.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
        return {line for line in out.split("\n") if line}
    except (OSError, subprocess.CalledProcessError):
        return {
            str(path.relative_to(REPO))
            for path in REPO.rglob("*")
            if path.is_file()
            and not (_LOCAL & set(path.relative_to(REPO).parts))
            and "data/benchmarks/structures" not in str(path.relative_to(REPO))
        }


#: The base SAE, the radius-of-gyration adapter at k=1, and the six single-direction benchmark
#: adapters. Each ships with a ``.json`` sidecar recording its bytes, digest and provenance.
SHIPPED_MODELS = (
    "sae_k32_time_layer_m1",
    "rg_adapter_cold_v2",
    "cryptic_uncond_m1",
    "cryptic_uncond_p1",
    "domain_motion_uncond_m1",
    "domain_motion_uncond_p1",
    "kinase_uncond_m1",
    "kinase_uncond_p1",
)

#: The six benchmark adapters, which carry a trained level and a generator report beside them.
BENCHMARK_ADAPTERS = SHIPPED_MODELS[2:]

BENCHMARK_DATA = (
    "data/benchmarks/pools/cryptic_benchmark.json",
    "data/benchmarks/pools/domain_motion.json",
    "data/benchmarks/pools/kinase.json",
    "data/benchmarks/references/domain_motion_benchmark.pt",
    "data/benchmarks/references/domain_motion_testsplit.pt",
    "data/benchmarks/references/cryptic_testsplit.pt",
    "data/benchmarks/references/kinase.pt",
    "data/benchmarks/cryptic/metric_ranges.json",
    "data/benchmarks/cryptic/pairing.csv",
    "data/benchmarks/cryptic/sequences.csv",
)


@pytest.mark.parametrize("name", SHIPPED_MODELS)
def test_shipped_model_matches_its_provenance_sidecar(name: str) -> None:
    """A sidecar that disagrees with its checkpoint is worse than no sidecar."""
    checkpoint = REPO / "models" / f"{name}.pt"
    sidecar = json.loads((REPO / "models" / f"{name}.json").read_text())
    payload = checkpoint.read_bytes()
    assert len(payload) == sidecar["bytes"]
    assert hashlib.sha256(payload).hexdigest() == sidecar["sha256"]


@pytest.mark.parametrize("name", BENCHMARK_ADAPTERS)
def test_benchmark_adapter_is_single_direction_and_agrees_with_its_sidecar(name: str) -> None:
    """The arm's trained level decides which reference it may be scored against.

    ``run_benchmark.py`` reads the level from the checkpoint and the injection site from the
    sidecar beside it. If those two disagree the run would steer at one dose and measure another,
    so the equality is asserted here rather than trusted.
    """
    torch = pytest.importorskip("torch")
    checkpoint = torch.load(REPO / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    sidecar = json.loads((REPO / "models" / f"{name}.json").read_text())
    config = checkpoint["adapter_config"]

    levels = config["target_levels"]
    assert len(levels) == 1, f"{name} is not single-direction: {levels}"
    assert [sidecar["trained_level"]] == levels

    site = config["intervention_site"]
    site = json.loads(site) if isinstance(site, str) else site
    assert sidecar["training"]["intervention_site"] == site
    assert len(site["score_calls"]) == 8


@pytest.mark.parametrize("name", SHIPPED_MODELS)
def test_shipped_checkpoint_carries_no_optimizer_state(name: str) -> None:
    """Optimizer state exists to continue training, which the release does not do."""
    torch = pytest.importorskip("torch")
    payload = torch.load(REPO / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    for key in ("optimizer_state", "optimizer_state_dict"):
        assert not payload.get(key), f"{name} still carries {key}"


def test_shipped_sae_loads_and_reconstructs() -> None:
    """The stripped SAE must still be a working TopK SAE, not merely a file that opens."""
    torch = pytest.importorskip("torch")
    from sae_bioemu.core.model import TimeConditionedTopKSAE

    model = TimeConditionedTopKSAE.load_checkpoint(REPO / "models/sae_k32_time_layer_m1.pt").eval()
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(4, model.config.input_dim, generator=generator)
    timesteps = torch.rand(4, generator=generator)
    with torch.no_grad():
        output = model(batch, timesteps)
    # TopK sparsity is architectural: exactly k latents are active, never "about" k.
    active = (output.latents != 0).sum(dim=-1)
    assert active.tolist() == [model.config.top_k] * 4
    assert output.reconstruction.shape == batch.shape


@pytest.mark.parametrize("relative", BENCHMARK_DATA)
def test_benchmark_data_ships(relative: str) -> None:
    assert (REPO / relative).is_file(), f"{relative} is missing from the release"


def test_benchmark_data_matches_its_manifest() -> None:
    """The manifest is what a reviewer checks a download against."""
    manifest = json.loads((REPO / "data/benchmarks/MANIFEST.json").read_text())
    for relative, digest in manifest["sha256"].items():
        path = REPO / "data/benchmarks" / relative
        assert path.is_file(), f"{relative} is in the manifest but not in the tree"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative


def test_every_released_property_resolves() -> None:
    """Two descriptors are imported by NAME at runtime, so a pruned tree can lose them silently.

    ``pocket_conformer_rmsd`` and ``radius_of_gyration`` reach their descriptors through
    ``importlib``, which no static import graph sees. Both were in fact dropped once while
    building this release and only a resolution check found it.
    """
    pytest.importorskip("torch")
    from sae_bioemu.steering.property_registry import resolve

    for name in (
        "pocket_conformer_rmsd",
        "domain_motion_backbone_rmsd",
        "kinase_conformer_rmsd",
        "radius_of_gyration",
    ):
        assert resolve(name).name == name


def test_the_command_line_offers_exactly_the_shipped_commands() -> None:
    pytest.importorskip("torch")
    import typer.main

    from sae_bioemu.cli import app

    command = typer.main.get_command(app)
    # GPU commands only. The CPU self-checks were cut: loading a checkpoint on synthetic noise
    # demonstrates nothing about the model, and the release ships only what produces a result.
    assert sorted(command.commands) == ["sae-encode", "steer-apply"]


def test_run_benchmark_knows_every_shipped_adapter() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_benchmark", REPO / "scripts/run_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert sorted(module.ARMS) == sorted(BENCHMARK_ADAPTERS)
    for name in module.ARMS:
        assert (REPO / "models" / f"{name}.pt").is_file()
        assert (REPO / "models" / f"{name}.json").is_file()


@pytest.mark.parametrize("name", SHIPPED_MODELS[1:])
def test_every_adapter_builds_its_site_through_the_function_steer_apply_calls(name: str) -> None:
    """``steer-apply --adapter <name>`` must be able to build the site for ALL seven adapters.

    It calls the SAME function the command calls, rather than re-implementing it: the previous
    version of this test rebuilt the site by hand without the two guard fields, passed, and the
    command died with ``KeyError`` on every conformational adapter. The benchmark generators read
    the same sidecar block, so the command and the benchmark steer at the same calls.
    """
    pytest.importorskip("torch")
    pytest.importorskip("bioemu")
    from sae_bioemu.bioemu.steering import steering_site_from_record
    from sae_bioemu.steering.adapter_io import load_adapter

    _adapter, payload = load_adapter(REPO / "models" / f"{name}.pt")
    spec = json.loads((REPO / "models" / f"{name}.json").read_text())["training"][
        "intervention_site"
    ]
    site = steering_site_from_record(spec)
    assert len(site.score_calls) == len(site.expected_timesteps)
    assert site.expected_score_call_total is not None

    if name in BENCHMARK_ADAPTERS:  # the Rg adapter is trained at several levels
        levels = payload["adapter_config"]["target_levels"]
        assert len(levels) == 1, f"{name} is documented as single-direction"


def test_reference_scoring_is_n_ca_c_o_in_angstrom_and_pairs_residues_by_number() -> None:
    """``--reference`` on real coordinates, in the atom set and units the benchmarks use.

    A domain-motion system's two deposited conformers are scored exactly as a user would get them
    from ``export_system.py``. Three things are checked at once, each of which has been wrong here:
    a conformer against ITSELF must read ~0 (the units bug read 29.3 A, because the reference was
    angstrom and the arm nanometres, and a rigid superposition cannot absorb a factor of ten); the
    OTHER conformer must reproduce the artifact's own recorded ``separation_angstrom``, which is
    what proves this is the same N/CA/C/O metric the benchmark reports; and a renumbered file must
    be refused rather than silently pairing unrelated residues.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("mdtraj")
    import subprocess
    import sys

    import typer

    from sae_bioemu.cli import _score_against_reference, _value_columns

    system = "A0A075Q0W3"
    subprocess.run(
        [sys.executable, str(REPO / "scripts/export_system.py"), "--benchmark", "domain_motion",
         "--system", system, "--out-dir", str(tmp := REPO / "outputs/_test_export")],
        check=True, capture_output=True,
    )  # fmt: skip
    entry = torch.load(
        REPO / "data/benchmarks/references/domain_motion_benchmark.pt",
        map_location="cpu",
        weights_only=False,
    )["entries"][system]
    sequence = str(entry["sequence"])
    first, second = entry["first"].double(), entry["second"].double()
    reference = tmp / f"{system}_first.pdb"

    values = _score_against_reference(reference, REPO, sequence, first, second)

    # 1e-3 A, not zero: the reference goes out through a PDB and back through mdtraj's float32
    # nanometres, so a self-comparison lands near 2e-6 A. The bug this guards against read 29.3 A.
    assert values["unsteered"] < 1e-3, values
    recorded = float(entry["separation_angstrom"])
    assert abs(values["steered"] - recorded) < 1e-4, (
        f"scoring the other conformer must reproduce the artifact's own separation "
        f"{recorded}, not {values['steered']}"
    )
    # And the command must SAY which atom set the two numbers are, since it prints them bare.
    assert "N/CA/C/O" in _value_columns(reference=True, control=True)

    shifted = reference.read_text().replace(" A   ", " A  1", 3)
    (tmp / "shifted.pdb").write_text(shifted)
    with pytest.raises(typer.BadParameter):
        _score_against_reference(tmp / "shifted.pdb", REPO, sequence, first, second)


def test_every_flag_and_path_the_readme_shows_actually_exists() -> None:
    """The README's commands must match the code, because a reviewer runs them verbatim.

    Twice now a flag was documented that the parser did not have, and once the install command
    could not build a wheel: prose drifts from argparse silently and nothing else catches it. This
    reads each command line out of the fenced blocks and checks its flags against the real parser,
    and checks that every repository path the README names is tracked.
    """
    import re
    import subprocess
    import sys

    from sae_bioemu.cli import app

    readme = (REPO / "README.md").read_text()
    lines: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", readme, re.S):
        # One logical command per entry: join backslash continuations, drop comments.
        for line in block.replace("\\\n", " ").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                lines.append(line)

    # DERIVED from the app, not hand-copied. A hand-written set goes stale the moment a flag is
    # renamed -- the README would still name the old one and this would still pass.
    import typer.main

    command = typer.main.get_command(app)
    known = {
        name: {
            option
            for parameter in sub.params
            for option in parameter.opts + parameter.secondary_opts
            if option.startswith("--")
        }
        for name, sub in command.commands.items()  # type: ignore[attr-defined]
    }
    assert set(known) == {"sae-encode", "steer-apply"}, sorted(known)
    for command, allowed in known.items():
        for line in lines:
            if f"confsteer {command}" not in line:
                continue
            unknown = set(re.findall(r"--[a-z][a-z-]+", line)) - allowed
            assert not unknown, f"README shows {sorted(unknown)} for `{command}`, which has none"

    scripts = {m for line in lines for m in re.findall(r"scripts/[A-Za-z0-9_/]+\.py", line)}
    for script in sorted(scripts):
        assert (REPO / script).is_file(), f"README runs {script}, which does not ship"
        helped = subprocess.run(
            [sys.executable, str(REPO / script), "--help"],
            capture_output=True, text=True, cwd=REPO,
        )  # fmt: skip
        assert helped.returncode == 0, helped.stderr[-400:]
        for line in lines:
            if script not in line:
                continue
            for flag in sorted(set(re.findall(r"--[a-z][a-z-]+", line))):
                # WORD BOUNDARY: a plain substring test passes `--system` because `--systems`
                # appears in the help, which is how a real flag went undocumented-but-"verified".
                assert re.search(
                    rf"{re.escape(flag)}\b", helped.stdout
                ), f"README passes {flag} to {script}, which rejects it"

    tracked = shipped_files()
    for path in sorted(set(re.findall(r"\b(?:data|models|configs)/[A-Za-z0-9_/.-]+", readme))):
        assert path in tracked or any(
            name.startswith(path.rstrip("/") + "/") for name in tracked
        ), f"README names {path}, which does not ship"


def test_domain_motion_is_scored_on_the_canonical_conformer_not_the_deposited_one() -> None:
    """The canonical cells must select each system's OWN nearer/further conformer.

    A DynDom pair's "conformer 0" is whichever structure was deposited first; the arms were trained
    on the CANONICAL order, level -1 toward whichever conformer the unsteered ensemble already sits
    nearer. They disagree in 17 of 22 systems, so a fixed deposition index scores most systems
    against the conformer the arm was not aimed at and still returns a plausible rate -- on the real
    stores the m1 arm reads 0.5909 that way, which is the published value of the OTHER arm.

    This drives the real aggregation with synthetic draws rather than grepping for label strings. An
    earlier version did grep, and would have passed with the two canonical mappings SWAPPED -- which
    is precisely the defect, so it guarded nothing.
    """
    pytest.importorskip("numpy")
    from sae_bioemu.domain_motion.success_at_b import success_at_b

    labels = json.loads(
        (REPO / "data/benchmarks/rosters/domain_motion_canonical_labels.json").read_text()
    )
    assert len(labels) == 22
    nearer = {str(row["system"]): int(str(row["nearer"]).rsplit("_", 1)[1]) for row in labels}
    assert sum(1 for index in nearer.values() if index == 1) == 17, (
        "the canonical and deposited orders are expected to disagree in 17 of 22 systems; if this "
        "changed, every domain-motion number must be re-read"
    )

    from sae_bioemu.domain_motion.success_at_b import DrawDeviation

    def _draw(system: str, index: int, rmsd: float) -> DrawDeviation:
        """One real DrawDeviation, so the aggregation's own ordering and fields are exercised."""
        return DrawDeviation(
            system=system,
            conformer=f"conformer_{index}",
            conformer_index=index,
            sample_id=f"{system}-0",
            sample_index=0,
            rmsd_angstrom=rmsd,
            residues=10,
            atoms=40,
        )

    class _Score:
        """One system whose draws sit AT the canonical conformer and far from the other."""

        def __init__(self, system: str, canonical: int) -> None:
            self.system = system
            self.separation = None
            self._canonical = canonical

        def draws(self, conformer_index: int) -> list[DrawDeviation]:
            close = conformer_index == self._canonical
            return [_draw(self.system, conformer_index, 0.1 if close else 99.0)]

    scores = [_Score(system, index) for system, index in sorted(nearer.items())]

    # Scored on each system's canonical conformer, every system succeeds.
    canonical = success_at_b(scores, conformer_index=nearer, b=1, tau=3.0, resamples=10)
    assert canonical.rate == 1.0, "the canonical mapping must reach each system's own conformer"

    # Scored on the FLIPPED mapping, none does. This is what the swap defect would produce.
    flipped = {system: 1 - index for system, index in nearer.items()}
    assert success_at_b(scores, conformer_index=flipped, b=1, tau=3.0, resamples=10).rate == 0.0

    # A fixed deposition index reaches only the systems whose two orders happen to agree.
    agreeing = sum(1 for index in nearer.values() if index == 0)
    fixed = success_at_b(scores, conformer_index=0, b=1, tau=3.0, resamples=10)
    assert (
        sum(fixed.successes) == agreeing == 5
    ), f"deposition index 0 should reach the {agreeing} agreeing systems, not {sum(fixed.successes)}"

    # And the SCORER must attach those mappings to the right names. Swapping them leaves both
    # labels in the file and every arm scored against the conformer it was steered away from, so
    # this asserts the association itself rather than the presence of the words.
    import sys

    sys.path.insert(0, str(REPO / "scripts/domain_motion"))
    from measure_success_at_b import canonical_choices

    named = dict(canonical_choices(nearer))
    assert named["canonical_nearer"] == nearer, (
        "canonical_nearer must score each system's OWN nearer conformer; it is currently mapped to "
        "the other one, which reports every arm against the conformer it was steered away from"
    )
    assert named["canonical_further"] == {s: 1 - i for s, i in nearer.items()}

    # And the launcher must select the arm's cell BY NAME, not by position in the cell list.
    launcher = (REPO / "scripts/run_benchmark.py").read_text()
    assert '"canonical_nearer" if level < 0 else "canonical_further"' in launcher
    assert "--canonical-labels" in launcher


def test_several_proteins_land_in_one_latent_file_tagged_by_sequence(tmp_path: Path) -> None:
    """``--sequences N`` keeps every protein's activations, in ONE file, told apart by id.

    The obvious failure is silent: encoding N proteins and writing a file that holds only the
    last one's rows, or holds all of them with no way to say which protein a row came from.
    Sequence is the statistical unit for everything downstream, so a row that has lost its
    sequence_id cannot be grouped afterwards and the file is worth little.
    """
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    import types

    from sae_bioemu.bioemu.extractor import ActivationRecordList
    from sae_bioemu.cli import _write_latents
    from sae_bioemu.core.types import ActivationRecord

    sink = ActivationRecordList()
    lengths = {"P1": 5, "P2": 3}
    for sequence_id, length in lengths.items():
        sink.append(
            ActivationRecord(
                activations=torch.zeros(length, 512),
                timesteps=torch.full((length,), 0.5),
                sequence_ids=(sequence_id,) * length,
                residue_indices=torch.arange(length),
            )
        )
    record = sink.concat()
    assert record.activations.shape[0] == sum(lengths.values())

    latents = torch.zeros(record.activations.shape[0], 16)
    latents[:, 0], latents[:, 1] = 1.0, 0.5
    model = types.SimpleNamespace(config=types.SimpleNamespace(top_k=2, n_latents=16))
    _write_latents(tmp_path, record, latents, model)

    # No allow_pickle: an object-dtype id column would not load for a reader who omits it.
    payload = numpy.load(tmp_path / "sae_latents.npz")
    written = [str(value) for value in payload["sequence_id"]]
    assert {name: written.count(name) for name in lengths} == lengths
    assert payload["latent_values"].shape[0] == sum(lengths.values())
    # Residue indices restart per protein, so they only locate a row together with its id.
    assert payload["residue_index"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]


def test_the_cli_actually_runs_as_a_module() -> None:
    """``python -m sae_bioemu.cli <command>`` must RUN the command, not import and exit 0.

    Without a ``__main__`` guard the module defines its commands, runs nothing and exits 0. That is
    worse than a crash: a wrapper that checks the exit code reads the silence as success. It is why
    a GPU job reported four generations it had never run, in 17 seconds, with every exit code 0.

    So this asserts on OUTPUT, not on the exit code: an unknown command must be REJECTED, and a
    known one must reach its own argument checking rather than returning quietly.
    """
    import subprocess
    import sys

    helped = subprocess.run(
        [sys.executable, "-m", "sae_bioemu.cli", "--help"],
        capture_output=True, text=True, cwd=REPO, env=_module_env(),
    )  # fmt: skip
    assert "sae-encode" in helped.stdout and "steer-apply" in helped.stdout, helped.stderr[-400:]

    rejected = subprocess.run(
        [sys.executable, "-m", "sae_bioemu.cli", "no-such-command"],
        capture_output=True, text=True, cwd=REPO, env=_module_env(),
    )  # fmt: skip
    assert rejected.returncode != 0, "an unknown command must not exit 0"
    assert "no-such-command" in (rejected.stdout + rejected.stderr)


def _module_env() -> dict[str, str]:
    """``PYTHONPATH`` pointing at this checkout, so the test exercises THIS tree's CLI."""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    return env


def test_the_cryptic_benchmark_references_match_the_scorer() -> None:
    """The shipped cryptic BENCHMARK references must be what the benchmark scores against.

    This panel's references are the DEPOSITED apo and holo chains, in numbering the benchmark's own
    ranges do not use. They were resolved once, through the SCORER's letter-matched alignment, into
    the shipped artifact -- so the export needs no download and no extra command, like the other
    four sets.

    Checked by reproducing a number, not by inspection: apo-to-holo over each system's metric
    region must equal the separation the scorer records for it. If the alignment, the atom set, the
    metric positions or the states were wrong, these would not agree.
    """
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")

    from sae_bioemu.steering.aligned_rmsd import aligned_rmsd

    entries = torch.load(
        REPO / "data/benchmarks/references/cryptic_benchmark.pt",
        map_location="cpu",
        weights_only=False,
    )["entries"]
    assert len(entries) == 32, (
        f"the cryptic panel the paper reports is 32 systems, not {len(entries)}: the pool holds 34, "
        "Q16539 has no recoverable metric ranges and Q02763 no measurable apo/holo separation"
    )

    #: The scorer's own recorded apo->holo RMSD over the metric region, angstrom.
    recorded = {
        "O74933": 2.38762150743482,
        "P00766": 2.068090048823736,
        "P56817": 0.36054207686489226,
    }
    for system, expected in recorded.items():
        entry = entries[system]
        apo = torch.as_tensor(numpy.asarray(entry["apo"], dtype=float)).reshape(-1, 3)
        holo = torch.as_tensor(numpy.asarray(entry["holo"], dtype=float)).reshape(-1, 3)
        measured = float(aligned_rmsd(holo, apo).reshape(()))
        assert abs(measured - expected) < 1e-3, f"{system}: {measured} vs the scorer's {expected}"
        assert len(entry["metric_indices"]) == apo.shape[0] // 4 == holo.shape[0] // 4
        assert max(entry["metric_indices"]) < len(str(entry["sequence"]))


def test_neither_command_asks_the_sampler_to_save_structures_it_cannot_write() -> None:
    """Both commands must hand the sampler a trace with ``save_final_structures`` OFF.

    The shipped config sets it true, which is right for a training run that has a structure store.
    These commands pass ``structures_dir=None``, and the sampler refuses that pairing outright
    rather than generating every structure and dropping it -- so each command has to turn the flag
    off for itself, or it dies before the first denoising step.

    This is checked on the SOURCE, deliberately. Behaviourally it is only reachable where bioemu's
    full runtime is installed: everywhere else the sampler raises a missing-dependency error first,
    which made an earlier behavioural version of this test pass with the fix reverted. A structural
    check is worth less than a behavioural one, and much more than one that cannot fail.
    """
    import ast

    source = (REPO / "src/sae_bioemu/cli.py").read_text()
    tree = ast.parse(source)
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "BioEmuRuntimeActivationSampler"
    ]
    assert len(constructions) == 2, f"expected one sampler per command, found {len(constructions)}"
    for call in constructions:
        assert call.args, "the trace is the sampler's first positional argument"
        trace = call.args[0]
        assert isinstance(trace, ast.Call) and getattr(trace.func, "id", "") == "replace", (
            "the trace must be passed through dataclasses.replace(...) so this command can turn "
            "save_final_structures off; passing project.trace straight through dies at setup"
        )
        disabled = [
            keyword
            for keyword in trace.keywords
            if keyword.arg == "save_final_structures"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
        ]
        assert disabled, "replace(...) must set save_final_structures=False"
        # And a DEVICE. BioEmu's model loads on the host and the sampler moves it only when told
        # where to; without this keyword both commands generated on the CPU, correctly and about
        # sixty times slower, on a GPU node that sat idle.
        # A bare ``device=`` would pass with ``device=None``, which is the CPU default again.
        devices = [keyword.value for keyword in call.keywords if keyword.arg == "device"]
        assert (
            devices
            and all(
                isinstance(value, ast.Call) and getattr(value.func, "id", "") == "_resolve_device"
                for value in devices
            )
        ), "every sampler the CLI builds must be given device=_resolve_device(...), or BioEmu runs on the CPU"


def test_sequences_zero_means_every_protein_not_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--sequences 0`` means ALL, for both commands, as the README says.

    Slicing ``[:0]`` took NONE and then reported "contained no sequences", blaming the FASTA for the
    flag. Checked without generating anything: the sampler is replaced by one that stops the command
    the moment it would build the model. An earlier version let the command run on, which was
    harmless where BioEmu's runtime was incomplete and, in a COMPLETE install, started embedding and
    generating a thousand proteins on whatever machine ran the suite -- a test suite a reviewer
    cannot finish.
    """
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    import sae_bioemu.cli as cli

    class _Stop(Exception):
        pass

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise _Stop("the model would be built here; the selection is already decided")

    monkeypatch.setattr(cli, "BioEmuRuntimeActivationSampler", _refuse)
    fasta = REPO / "data/splits/uniref50_interpro_train_1000.fasta"
    result = CliRunner().invoke(cli.app, ["sae-encode", "--fasta", str(fasta), "--sequences", "0"])
    combined = f"{result.output or ''}\n{result.exception or ''}"
    assert isinstance(result.exception, _Stop), f"stopped somewhere else:\n{combined}"
    assert "contained no sequences" not in combined, combined
    counted = [line for line in (result.output or "").splitlines() if line.startswith("encoding")]
    assert counted, combined
    assert int(counted[0].split()[1]) == 1000, f"--sequences 0 should read all 1000: {counted[0]}"


#: The panel each benchmark REPORTS on, which is not always its pool. A rate is an unweighted mean
#: over systems, so the denominator is part of the number: scoring 33 cryptic systems where the
#: paper scored 32 is a different statistic wearing the same name.
REPORTED_PANELS = {
    "cryptic_benchmark.pt": (32, "cryptic pockets"),
    "domain_motion_benchmark.pt": (22, "domain motions"),
    "kinase.pt": (20, "kinase DFG flip"),
}


@pytest.mark.parametrize("artifact", sorted(REPORTED_PANELS))
def test_each_shipped_panel_is_the_size_the_paper_reports(artifact: str) -> None:
    """Every benchmark artifact must hold exactly the systems its reported rate is averaged over.

    The cryptic pool holds 34 and the reported panel is 32: Q16539 has no recoverable metric ranges
    upstream, and Q02763's apo/holo alignment resolves 18 of 42 positions, below the coverage floor,
    so the scorer measures no separation for it. An artifact carrying either would let a rerun
    average over a denominator the paper never used, and the rate would look like a discrepancy in
    the adapters rather than in the population.
    """
    torch = pytest.importorskip("torch")

    expected, what = REPORTED_PANELS[artifact]
    entries = torch.load(
        REPO / "data/benchmarks/references" / artifact, map_location="cpu", weights_only=False
    )["entries"]
    assert (
        len(entries) == expected
    ), f"{what}: reported panel is {expected}, artifact has {len(entries)}"


@pytest.mark.parametrize("name", SHIPPED_MODELS[1:])
def test_every_adapters_property_can_be_persisted(name: str) -> None:
    """The structure writer must know the units of every shipped adapter's property.

    It refuses an undeclared name rather than stamping a default, which is the right call -- a wrong
    unit on a persisted property is silent downstream. But only the radius-of-gyration property was
    declared, so the held-out evaluation, which persists both arms under the adapter's OWN property
    name, generated every structure and then refused to write it.
    """
    torch = pytest.importorskip("torch")

    from sae_bioemu.steering.evaluation_structures import property_units

    payload = torch.load(REPO / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    property_name = payload["adapter_config"]["property_name"]
    assert (
        property_units(property_name) == "angstrom"
    ), f"{name} steers {property_name}, a superposed N/CA/C/O deviation in angstrom"


def test_a_reference_directory_reaches_the_scorer_as_one_file_per_protein() -> None:
    """``--reference <dir>`` must hand the SCORER a file, never the directory.

    The per-protein lookup and the scoring call are twenty lines apart, and the command printed the
    right file for each protein while passing the directory on to be opened -- so it resolved
    correctly, announced correctly, and then died with "no reference structure at <dir>" after
    generating both arms. Checked on the source because reaching it needs a GPU.
    """
    import ast

    source = (REPO / "src/sae_bioemu/cli.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_score_against_reference"
    ]
    assert calls, "steer-apply must score against the reference"
    for call in calls:
        first = call.args[0]
        assert getattr(first, "id", "") == "reference_for_record", (
            "the first argument must be the per-protein file resolved for THIS record, not the "
            f"--reference argument itself (got {ast.dump(first)[:60]})"
        )


def test_every_file_the_release_needs_is_actually_TRACKED() -> None:
    """Present in the working tree is not shipped. Everything needed must be in `git ls-files`.

    A reference artifact was added, verified, manifested and documented while `.gitignore`'s `*.pt`
    rule kept it untracked. Every test passed, because every test reads the working tree -- and a
    clone would have had a broken `--benchmark cryptic` and two failing tests. `git ls-files` is the
    only statement about what ships, so it is what this asserts against.
    """

    tracked = shipped_files()
    manifest = json.loads((REPO / "data/benchmarks/MANIFEST.json").read_text())
    needed = {f"data/benchmarks/{name}" for name in manifest["sha256"]}
    needed |= {f"models/{name}.pt" for name in SHIPPED_MODELS}
    needed |= {f"models/{name}.json" for name in SHIPPED_MODELS}
    missing = sorted(name for name in needed if name not in tracked and not name.endswith("/"))
    assert not missing, (
        "these are needed but not tracked, so they exist here and not in a clone:\n  "
        + "\n  ".join(missing)
    )


def test_an_unset_seed_is_fresh_and_a_given_seed_is_kept() -> None:
    """Without --seed every run samples fresh; with it a run is reproducible.

    A fixed default looked reproducible and broke the obvious use -- a loop over the same command
    returned the identical structure every time. BioEmu's own sampler takes a fresh seed when none
    is given, and so does this.
    """
    pytest.importorskip("torch")
    from sae_bioemu.cli import _resolve_seed

    assert _resolve_seed(7) == 7
    assert _resolve_seed(0) == 0, "an explicit 0 is a seed, not 'unset'"
    first, second = _resolve_seed(None), _resolve_seed(None)
    assert first != second, "two runs without --seed must not share a seed"


def test_steer_apply_sequences_zero_means_every_protein(monkeypatch: pytest.MonkeyPatch) -> None:
    """``steer-apply --sequences 0`` selects every protein, as ``sae-encode`` does."""
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    import sae_bioemu.cli as cli

    class _Stop(Exception):
        pass

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise _Stop("the model would be built here; the selection is already decided")

    monkeypatch.setattr(cli, "BioEmuRuntimeActivationSampler", _refuse)
    fasta = REPO / "data/splits/uniref50_interpro_train_1000.fasta"
    result = CliRunner().invoke(cli.app, ["steer-apply", "--fasta", str(fasta), "--sequences", "0"])
    combined = f"{result.output or ''}\n{result.exception or ''}"
    assert isinstance(result.exception, _Stop), f"stopped somewhere else:\n{combined}"
    counted = [line for line in (result.output or "").splitlines() if " proteins x " in line]
    assert counted, combined
    assert counted[0].split()[1] == "1000", counted[0]


def test_a_reference_directory_is_matched_by_id_at_a_separator(tmp_path: Path) -> None:
    """One file per protein, named by id; a prefix id must not claim a longer id's file."""
    import types

    import typer

    from sae_bioemu.cli import _resolve_references

    def chosen(*names: str) -> list:
        return [types.SimpleNamespace(sequence_id=name) for name in names]

    for name in ("P0205.pdb", "P02051_apo.pdb", "Q9-holo.cif"):
        (tmp_path / name).write_text("")
    found = _resolve_references(tmp_path, tmp_path, chosen("P0205", "P02051", "Q9", "absent"))
    assert found == {
        "P0205": tmp_path / "P0205.pdb",
        "P02051": tmp_path / "P02051_apo.pdb",
        "Q9": tmp_path / "Q9-holo.cif",
        "absent": None,
    }
    # A single protein may be given its reference directly, as a file.
    assert _resolve_references(tmp_path / "Q9-holo.cif", tmp_path, chosen("Q9")) == {
        "Q9": tmp_path / "Q9-holo.cif"
    }
    with pytest.raises(typer.BadParameter, match="one structure"):
        _resolve_references(tmp_path / "Q9-holo.cif", tmp_path, chosen("Q9", "P0205"))
    (tmp_path / "Q9_apo.pdb").write_text("")
    with pytest.raises(typer.BadParameter, match="2 structures for Q9"):
        _resolve_references(tmp_path, tmp_path, chosen("Q9"))
    with pytest.raises(typer.BadParameter, match="holds no structure"):
        _resolve_references(tmp_path, tmp_path, chosen("absent"))


def _run_benchmark_module() -> object:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_benchmark", REPO / "scripts/run_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_coverage_gate_reads_the_state_it_is_asked_about(tmp_path: Path) -> None:
    """The gate must judge the requested state's cell, not whichever state the sidecar lists."""
    confirm_coverage = _run_benchmark_module().confirm_coverage

    def cell(state: str, systems: int, criterion: str = "plain") -> dict:
        return {
            "reference_state": state,
            "criterion": criterion,
            "deviation_form": "rmsd",
            "n_systems": systems,
            "success_rate": 0.5,
            "b_requested": 100,
            "tau_angstrom": 1.5,
        }

    sidecar = tmp_path / "success_at_b.json"
    sidecar.write_text(
        json.dumps(
            {"success_table": [cell("holo", 31), cell("apo", 32), cell("apo", 5, "specific")]}
        )
    )
    confirm_coverage(sidecar, 32, "apo", 100, arm="cryptic_uncond_m1", reference="apo", panel=32)
    with pytest.raises(SystemExit, match="scored 31 systems"):
        confirm_coverage(sidecar, 32, "holo", 100)
    with pytest.raises(SystemExit, match="B=100"):
        confirm_coverage(sidecar, 32, "apo", 10)
    with pytest.raises(SystemExit, match="no scored cell"):
        confirm_coverage(sidecar, 32, "canonical_nearer", 100)
    # And the state each arm is gated on is the state it was trained toward.
    source = (REPO / "scripts/run_benchmark.py").read_text()
    assert '"apo" if level < 0 else "holo"' in source
    assert '"canonical_nearer" if level < 0 else "canonical_further"' in source


def test_no_tracked_file_is_also_ignored() -> None:
    """A tracked file that an ignore rule matches ships today and silently stops shipping when
    re-added. ``examples/`` once matched ``configs/examples/`` this way."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-i", "-c", "--exclude-standard"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert not listed, f"tracked but ignored: {listed}"


def test_a_shallow_test_split_run_is_scored_at_its_own_depth() -> None:
    """``--split test --draws 2`` must score, not skip every system for having under five draws.

    Five holds at the reported depth, where it is part of the published denominator; below it the
    minimum drops to the run's own depth.
    """
    import ast

    tree = ast.parse((REPO / "scripts/run_benchmark.py").read_text())
    scorer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and any(
            isinstance(item, ast.Constant) and str(item.value).endswith("testsplit.py")
            for item in node.elts
        )
    ]
    assert len(scorer_calls) == 2, "one paired-shift scorer per test-split benchmark"
    for call in scorer_calls:
        items = [ast.unparse(item) for item in call.elts]
        assert "'--min-draws'" in items, items
        assert (
            items[items.index("'--min-draws'") + 1] == "str(min(MIN_PAIRED_DRAWS, arguments.draws))"
        )


def test_the_paired_shift_scorer_refuses_when_nothing_is_scorable(tmp_path: Path) -> None:
    """An empty result is a refusal with a reason, not an IndexError that reads as partial."""
    import sys

    pytest.importorskip("torch")
    (tmp_path / "steered_draws.jsonl").write_text("")
    result = subprocess.run(
        [sys.executable, "scripts/community/score_dm_testsplit.py", "--store", str(tmp_path)]
        + ["--reference-artifact", "data/benchmarks/references/domain_motion_testsplit.pt"]
        + ["--level", "-1", "--tau", "3.0", "--out-json", str(tmp_path / "s.json")]
        + ["--out-csv", str(tmp_path / "s.csv")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "REFUSING" in result.stderr, result.stderr[-800:]
    assert "IndexError" not in result.stderr


@pytest.mark.parametrize(
    "entry", ["import sae_bioemu.cli", "import runpy; runpy.run_path('scripts/run_benchmark.py')"]
)
def test_a_notebook_matplotlib_backend_does_not_break_the_entry_points(entry: str) -> None:
    """Colab exports an inline backend the project environment lacks; matplotlib then refuses to
    import, and BioEmu's AlphaFold2 imports it via TensorFlow/Keras before any embedding runs."""
    import os
    import sys

    pytest.importorskip("matplotlib")
    env = dict(os.environ, MPLBACKEND="module://not_an_installed_backend.inline")
    code = f"{entry}\nimport matplotlib.pyplot\nprint('ok')"
    if "run_benchmark" in entry:
        # run_path executes main() only under __main__; the module-level repair is what matters.
        code = code.replace(
            "runpy.run_path('scripts/run_benchmark.py')",
            "runpy.run_path('scripts/run_benchmark.py', run_name='not_main')",
        )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-1500:]


def test_a_structure_file_name_says_which_protein_seed_and_arm_it_is() -> None:
    """Both arms of a draw must land beside each other under names that cannot collide.

    The id comes from a FASTA header, which may carry a separator or a space: unsanitised, one
    protein's structures are written into some other directory, or under a name whose second word
    is lost by every tool that splits on whitespace.
    """
    pytest.importorskip("torch")
    from sae_bioemu.cli import _structure_filename

    assert _structure_filename("abl1_kd", 11, "on") == "abl1_kd_seed11_on.pdb"
    assert _structure_filename("abl1_kd", 11, "off") == "abl1_kd_seed11_off.pdb"
    assert _structure_filename("../etc/x y", 3, "on") == ".._etc_x_y_seed3_on.pdb"
    assert "/" not in _structure_filename("a/b", 1, "on")


def test_an_arm_is_written_as_a_backbone_pdb_in_angstrom(tmp_path: Path) -> None:
    """``steer-apply`` writes PDBs itself, so the units and the atom set are its responsibility.

    BioEmu's coordinates are nanometres and a PDB is Angstrom. A factor of ten writes a file that
    opens and renders and measures nothing, which is exactly the defect ``--reference`` scoring hit
    once already -- so the round trip is asserted against the input rather than eyeballed.
    """
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    mdtraj = pytest.importorskip("mdtraj")
    pytest.importorskip("bioemu")

    from sae_bioemu.cli import _structure_filename, _write_backbone_pdb

    sequence = "ACDEFGHIKL"
    generator = torch.Generator().manual_seed(0)
    # Padded like the sampler's own output: [members, residues, 3] nanometres, and the frames beside
    # it, which is what makes the written file backbone-complete rather than C-alpha only.
    positions = torch.randn(1, len(sequence), 3, generator=generator) * 0.5
    orientations = torch.eye(3).expand(1, len(sequence), 3, 3).contiguous()
    path = tmp_path / _structure_filename("demo", 7, "on")
    _write_backbone_pdb(path, positions, orientations, sequence)

    assert [item.name for item in tmp_path.iterdir()] == [path.name], "a temporary file was left"
    frame = mdtraj.load(str(path))
    assert frame.topology.n_residues == len(sequence)
    assert {"N", "CA", "C", "O"} <= {atom.name for atom in frame.topology.atoms}
    # mdtraj reads the file's Angstrom back into nanometres, so the C-alphas must return the input.
    alphas = [atom.index for atom in frame.topology.atoms if atom.name == "CA"]
    centred = (positions[0] - positions[0].mean(dim=0)).numpy()
    assert numpy.abs(frame.xyz[0][alphas] - centred).max() < 1e-3


def test_a_stage_logs_its_output_and_prints_the_tail_only_when_it_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run`` must keep a child's output out of the terminal and in a file -- until it fails.

    A benchmark run is several children, each of which streams tables and progress for hours; what
    the run concluded scrolls away. The output still has to be somewhere, and a failure has to say
    what happened rather than only an exit code.
    """
    import sys

    run = _run_benchmark_module().run
    log = tmp_path / "logs" / "stage.log"

    run([sys.executable, "-c", "print('quiet')"], "scoring", log)
    printed = capsys.readouterr().out
    assert printed.splitlines() == ["scoring... done"], printed
    assert "quiet" in log.read_text()

    failing = [sys.executable, "-c", "print('x'); raise SystemExit(3)"]
    with pytest.raises(SystemExit, match="exited 3"):
        run(failing, "generating 3 systems x 8 draws", log, show_log=True)
    printed = capsys.readouterr().out
    assert "(log: " in printed and "FAILED" in printed, printed
    assert "x" in printed.splitlines()[-1], printed
    assert "x" in log.read_text()


def test_a_relative_path_is_read_from_where_the_user_is(tmp_path: Path, monkeypatch) -> None:
    """``--fasta my.fasta`` means the file in the working directory; defaults still resolve."""
    from sae_bioemu.cli import _input_path, _output_path

    (tmp_path / "mine.fasta").write_text(">a\nAAA\n")
    monkeypatch.chdir(tmp_path)
    assert _input_path(Path("mine.fasta"), REPO) == tmp_path / "mine.fasta"
    assert (
        _input_path(Path("models/kinase_uncond_p1.pt"), REPO) == REPO / "models/kinase_uncond_p1.pt"
    )
    assert _output_path(Path("outputs/x")) == tmp_path / "outputs/x"


def test_the_benchmark_generators_read_the_site_from_the_shipped_sidecar() -> None:
    """``run_benchmark.py`` hands the generators ``models/<name>.json``; no report file ships."""
    import importlib.util

    assert not list((REPO / "models").glob("*_report.json"))
    spec = importlib.util.spec_from_file_location(
        "kinase_generate", REPO / "scripts/kinase_dfg/generate_kinase_benchmark_draws.py"
    )
    module = importlib.util.module_from_spec(spec)
    pytest.importorskip("torch")
    try:
        spec.loader.exec_module(module)
    except ImportError as error:  # the generator needs BioEmu's optional runtime
        pytest.skip(str(error))
    site, _spec = module.load_intervention_site(REPO / "models/kinase_uncond_p1.json")
    assert len(site.score_calls) == 8
    assert 'adapter.with_suffix(".json")' in (REPO / "scripts/run_benchmark.py").read_text()
