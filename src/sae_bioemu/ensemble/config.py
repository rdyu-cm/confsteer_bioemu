from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import yaml

_ARCHIVE_DEPENDENCIES = (
    Path("src/sae_bioemu/core/config.py"),
    Path("src/sae_bioemu/bioemu/extractor.py"),
    Path("src/sae_bioemu/bioemu/shared_evaluation.py"),
)
_ANALYSIS_DEPENDENCY_FILENAMES = (
    "properties.py",
    "representations.py",
    "inference.py",
    "timestep_inference.py",
    "reporting.py",
    "prior.py",
    "export.py",
)
_PROPERTY_SCIENTIFIC_DEPENDENCIES = (
    Path("src/sae_bioemu/analysis/general.py"),
    Path("src/sae_bioemu/ensemble/properties.py"),
)
_SHARED_ANALYSIS_SOURCE_DEPENDENCIES = (
    Path("src/sae_bioemu/analysis/general.py"),
    Path("src/sae_bioemu/bioemu/evaluation_archive.py"),
    Path("src/sae_bioemu/bioemu/evaluation_archive_composite.py"),
    Path("src/sae_bioemu/bioemu/shared_training.py"),
    Path("src/sae_bioemu/core/checkpoint_io.py"),
    Path("src/sae_bioemu/core/model.py"),
    Path("src/sae_bioemu/core/store.py"),
    Path("src/sae_bioemu/core/types.py"),
    Path("src/sae_bioemu/ensemble/config.py"),
)
_SHARED_ANALYSIS_SNAPSHOT_RELATIVE_PATH = Path("provenance") / "analysis_source_snapshot.json"
AnalysisDesign = Literal["independent_supplemental_128", "shared_original_32"]


@dataclass(frozen=True)
class ExpandedEnsembleConfig:
    original_archive_root: Path
    supplemental_archive_root: Path | None
    original_structure_root: Path
    supplemental_structure_root: Path | None
    input_fasta: Path
    ablation_configs: tuple[Path, ...]
    checkpoint_root: Path
    output_root: Path
    export_root: Path
    predictor_sample_start: int = 0
    predictor_sample_count: int = 32
    supplemental_sample_start: int = 32
    supplemental_sample_count: int = 128
    recorded_score_calls: int = 10
    permutations: int = 10_000
    bootstrap_replicates: int = 2_000
    random_seed: int = 20260728
    execution_scope: Literal["production", "smoke"] = "production"
    analysis_design: AnalysisDesign = "independent_supplemental_128"

    @classmethod
    def load(cls, path: str | Path) -> ExpandedEnsembleConfig:
        with Path(path).open() as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError("expanded ensemble config must be a mapping")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown keys in expanded ensemble config: {sorted(unknown)}")

        def resolved(value: object) -> Path:
            candidate = Path(str(value)).expanduser()
            return candidate if candidate.is_absolute() else Path.cwd() / candidate

        analysis_design = str(payload.get("analysis_design", "independent_supplemental_128"))
        if analysis_design not in {"independent_supplemental_128", "shared_original_32"}:
            raise ValueError("analysis_design is invalid")

        def optional_resolved(key: str) -> Path | None:
            value = payload.get(key)
            if value is None:
                return None
            return resolved(value)

        try:
            config = cls(
                original_archive_root=resolved(payload["original_archive_root"]),
                supplemental_archive_root=optional_resolved("supplemental_archive_root"),
                original_structure_root=resolved(payload["original_structure_root"]),
                supplemental_structure_root=optional_resolved("supplemental_structure_root"),
                input_fasta=resolved(payload["input_fasta"]),
                ablation_configs=tuple(resolved(item) for item in payload["ablation_configs"]),
                checkpoint_root=resolved(payload["checkpoint_root"]),
                output_root=resolved(payload["output_root"]),
                export_root=resolved(payload["export_root"]),
                predictor_sample_start=int(payload.get("predictor_sample_start", 0)),
                predictor_sample_count=int(payload.get("predictor_sample_count", 32)),
                supplemental_sample_start=int(payload.get("supplemental_sample_start", 32)),
                supplemental_sample_count=int(payload.get("supplemental_sample_count", 128)),
                recorded_score_calls=int(payload.get("recorded_score_calls", 10)),
                permutations=int(payload.get("permutations", 10_000)),
                bootstrap_replicates=int(payload.get("bootstrap_replicates", 2_000)),
                random_seed=int(payload.get("random_seed", 20260728)),
                execution_scope=str(payload.get("execution_scope", "production")),
                analysis_design=analysis_design,  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise ValueError(f"missing expanded ensemble config key: {error.args[0]}") from error
        config.validate_paths()
        return config

    def checkpoint_path(self, ablation_config: Path) -> Path:
        return self.checkpoint_root / ablation_config.stem / "sae.pt"

    @property
    def predictor_sample_indices(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.predictor_sample_start,
                self.predictor_sample_start + self.predictor_sample_count,
            )
        )

    @property
    def supplemental_sample_indices(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.supplemental_sample_start,
                self.supplemental_sample_start + self.supplemental_sample_count,
            )
        )

    @property
    def property_sample_indices(self) -> tuple[int, ...]:
        return self.supplemental_sample_indices

    @property
    def property_archive_root(self) -> Path:
        if self.analysis_design == "shared_original_32":
            return self.original_archive_root
        if self.supplemental_archive_root is None:  # pragma: no cover - validate_paths guards this.
            raise ValueError("independent analysis requires supplemental_archive_root")
        return self.supplemental_archive_root

    @property
    def property_structure_root(self) -> Path:
        if self.analysis_design == "shared_original_32":
            return self.original_structure_root
        if (
            self.supplemental_structure_root is None
        ):  # pragma: no cover - validate_paths guards this.
            raise ValueError("independent analysis requires supplemental_structure_root")
        return self.supplemental_structure_root

    @property
    def property_sampling_scope(self) -> AnalysisDesign:
        return self.analysis_design

    @property
    def convergence_target_counts(self) -> tuple[int, ...]:
        if self.analysis_design == "shared_original_32":
            return (32,)
        return (32, 64, 96, 128)

    @property
    def uses_supplemental_archive(self) -> bool:
        return self.analysis_design == "independent_supplemental_128"

    @property
    def shared_analysis_snapshot_path(self) -> Path:
        return self.output_root / _SHARED_ANALYSIS_SNAPSHOT_RELATIVE_PATH

    def validate_paths(self) -> None:
        if self.analysis_design not in {"independent_supplemental_128", "shared_original_32"}:
            raise ValueError("analysis_design is invalid")
        if self.uses_supplemental_archive and (
            self.supplemental_archive_root is None or self.supplemental_structure_root is None
        ):
            raise ValueError("independent analysis requires supplemental roots")
        roots = tuple(
            _canonical_path(path)
            for path in (
                self.original_archive_root,
                self.original_structure_root,
                *(
                    (self.supplemental_archive_root, self.supplemental_structure_root)
                    if self.uses_supplemental_archive
                    else ()
                ),
                self.checkpoint_root,
                self.output_root,
                self.export_root,
            )
            if path is not None
        )
        for index, left in enumerate(roots):
            for right in roots[index + 1 :]:
                if left == right or left in right.parents or right in left.parents:
                    raise ValueError(f"configured roots overlap: {left} and {right}")
        if self.execution_scope not in {"production", "smoke"}:
            raise ValueError("execution_scope must be production or smoke")
        if self.predictor_sample_indices != tuple(range(32)):
            raise ValueError("predictor samples must be exactly 0..31")
        if self.analysis_design == "shared_original_32":
            if self.property_sample_indices != tuple(range(32)):
                raise ValueError("shared property samples must be exactly 0..31")
        else:
            expected_supplemental = (
                tuple(range(32, 160)) if self.execution_scope == "production" else (32, 33)
            )
            if self.supplemental_sample_indices != expected_supplemental:
                if self.execution_scope == "smoke":
                    raise ValueError("smoke supplemental samples must be exactly 32 and 33")
                raise ValueError("supplemental samples must be exactly 32..159")
        if self.recorded_score_calls != 10:
            raise ValueError("recorded_score_calls must be exactly 10")

    def fingerprint(self, stage: Literal["archive", "analysis"]) -> str:
        if stage not in {"archive", "analysis"}:
            raise ValueError("stage must be 'archive' or 'analysis'")
        checkpoint_hashes = _checkpoint_hashes(self) if stage == "analysis" else None
        return self._fingerprint(stage, checkpoint_hashes=checkpoint_hashes)

    def analysis_fingerprint(self, *, checkpoint_hashes: dict[str, str]) -> str:
        """Reconstruct the analysis identity from previously attested checkpoint hashes."""

        expected_paths = {str(self.checkpoint_path(path)) for path in self.ablation_configs}
        if set(checkpoint_hashes) != expected_paths or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in checkpoint_hashes.values()
        ):
            raise ValueError("attested checkpoint hash inventory is invalid")
        return self._fingerprint("analysis", checkpoint_hashes=checkpoint_hashes)

    def _fingerprint(
        self,
        stage: Literal["archive", "analysis"],
        *,
        checkpoint_hashes: dict[str, str] | None,
    ) -> str:
        if self.uses_supplemental_archive:
            archive_paths = {
                "original_archive_root": str(_canonical_path(self.original_archive_root)),
                "supplemental_archive_root": str(_canonical_path(self.property_archive_root)),
                "original_structure_root": str(_canonical_path(self.original_structure_root)),
                "supplemental_structure_root": str(_canonical_path(self.property_structure_root)),
            }
            seed_scope: dict[str, Any] = {
                "predictor": self.predictor_sample_indices,
                "supplemental": self.supplemental_sample_indices,
                "recorded_score_calls": self.recorded_score_calls,
            }
        else:
            archive_paths = {
                "original_archive_root": str(_canonical_path(self.original_archive_root)),
                "original_structure_root": str(_canonical_path(self.original_structure_root)),
            }
            seed_scope = {
                "predictor": self.predictor_sample_indices,
                "property": self.property_sample_indices,
                "deliberate_overlap": tuple(
                    sample
                    for sample in self.predictor_sample_indices
                    if sample in self.property_sample_indices
                ),
                "recorded_score_calls": self.recorded_score_calls,
            }
        payload: dict[str, Any] = {
            "schema_version": 2,
            "analysis_contract": "expanded_ensemble_without_local_delta_g_v1",
            "stage": stage,
            "archive_paths": archive_paths,
            "seed_scope": seed_scope,
            "execution_scope": self.execution_scope,
            "fasta_sha256": _sha256_file(self.input_fasta),
            "ablation_configs": {str(path): _sha256_file(path) for path in self.ablation_configs},
            "extraction_dependencies": _hash_paths(_ARCHIVE_DEPENDENCIES),
        }
        if not self.uses_supplemental_archive:
            payload["analysis_design"] = self.analysis_design
        if stage == "analysis":
            payload.update(
                analysis_paths={
                    "checkpoint_root": str(_canonical_path(self.checkpoint_root)),
                    "output_root": str(_canonical_path(self.output_root)),
                    "export_root": str(_canonical_path(self.export_root)),
                },
                checkpoint_hashes=checkpoint_hashes,
                analysis_dependencies=_hash_existing_paths(_analysis_dependency_paths()),
                statistical_parameters={
                    "permutations": self.permutations,
                    "bootstrap_replicates": self.bootstrap_replicates,
                    "random_seed": self.random_seed,
                },
            )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _resolved_pair(left: Path, right: Path) -> tuple[Path, Path]:
    return _canonical_path(left), _canonical_path(right)


def _canonical_path(path: Path) -> Path:
    path = path.expanduser()
    absolute = path if path.is_absolute() else Path.cwd() / path
    suffix: list[str] = []
    while not absolute.exists() and not absolute.is_symlink():
        suffix.append(absolute.name)
        absolute = absolute.parent
    return absolute.resolve(strict=False).joinpath(*reversed(suffix))


def _hash_paths(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): _sha256_file(path) for path in paths}


def _analysis_dependency_paths() -> tuple[Path, ...]:
    package_root = Path(__file__).parent
    return tuple(package_root / filename for filename in _ANALYSIS_DEPENDENCY_FILENAMES)


def analysis_dependency_fingerprint() -> str:
    """Hash the exact property-analysis source dependency set."""

    payload = _hash_existing_paths(_analysis_dependency_paths())
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def property_scientific_dependency_fingerprint() -> str:
    """Hash code that defines Task 5 property values, excluding its consumers."""

    return _property_scientific_dependency_fingerprint(Path(__file__).parents[3])


def _property_scientific_dependency_fingerprint(repository_root: Path) -> str:
    payload = {
        path.as_posix(): _sha256_file(repository_root / path)
        for path in _PROPERTY_SCIENTIFIC_DEPENDENCIES
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def shared_analysis_source_preflight_fingerprint(
    config: ExpandedEnsembleConfig,
    config_path: str | Path,
    repository_commit: str,
    config_bytes: bytes | None = None,
) -> str:
    """Hash compact shared inputs before the exhaustive scheduled snapshot."""

    if config.analysis_design != "shared_original_32":
        raise ValueError("analysis-source preflight requires analysis_design=shared_original_32")
    if len(repository_commit) != 40 or any(
        character not in "0123456789abcdef" for character in repository_commit
    ):
        raise ValueError("analysis-source preflight requires an exact repository commit")
    resolved_config = _canonical_path(Path(config_path))
    if config_bytes is None:
        config_bytes = _read_regular_file(resolved_config)
    archive_manifest = config.original_archive_root / "dataset_manifest.json"
    structure_candidates = (
        config.original_structure_root / "index.jsonl",
        config.original_structure_root / "final_structures" / "index.jsonl",
    )
    structure_indexes = tuple(path for path in structure_candidates if path.is_file())
    if len(structure_indexes) != 1:
        raise ValueError("shared analysis requires one canonical structure index")
    payload = {
        "schema_version": 1,
        "analysis_design": config.analysis_design,
        "repository_commit": repository_commit,
        "config": {
            "path": str(resolved_config),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "input_fasta": {
            "path": str(_canonical_path(config.input_fasta)),
            "sha256": hashlib.sha256(_read_regular_file(config.input_fasta)).hexdigest(),
        },
        "archive_manifest": {
            "path": str(_canonical_path(archive_manifest)),
            "sha256": hashlib.sha256(_read_regular_file(archive_manifest)).hexdigest(),
        },
        "structure_index": {
            "path": str(_canonical_path(structure_indexes[0])),
            "sha256": hashlib.sha256(_read_regular_file(structure_indexes[0])).hexdigest(),
        },
        "checkpoints": _checkpoint_hashes(config),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def shared_analysis_source_snapshot_bytes(
    config: ExpandedEnsembleConfig,
    config_path: str | Path,
    config_bytes: bytes | None = None,
) -> bytes:
    """Capture the immutable source identity required by shared-archive analysis."""

    if config.analysis_design != "shared_original_32":
        raise ValueError("analysis-source snapshots require analysis_design=shared_original_32")
    config.validate_paths()
    resolved_config = _canonical_path(Path(config_path))
    if config_bytes is None:
        config_bytes = _read_regular_file(resolved_config)
    source_paths = (
        *_ARCHIVE_DEPENDENCIES,
        *_analysis_dependency_paths(),
        *_SHARED_ANALYSIS_SOURCE_DEPENDENCIES,
    )
    payload = {
        "schema_version": 1,
        "analysis_design": config.analysis_design,
        "original_archive": _source_tree_identity(config.original_archive_root),
        "original_structures": _source_tree_identity(config.original_structure_root),
        "input_fasta": {
            "path": str(_canonical_path(config.input_fasta)),
            "sha256": hashlib.sha256(_read_regular_file(config.input_fasta)).hexdigest(),
        },
        "checkpoints": _checkpoint_hashes(config),
        "config": {
            "path": str(resolved_config),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "analysis_sources": _hash_regular_paths(source_paths),
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _source_tree_identity(root: Path) -> dict[str, object]:
    root_mode = root.lstat().st_mode
    if stat.S_ISLNK(root_mode):
        raise ValueError(f"shared analysis source root is a symlink: {root}")
    canonical_root = _canonical_path(root)
    root_mode = canonical_root.lstat().st_mode
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ValueError(f"shared analysis source root is missing: {canonical_root}")
    files: dict[str, str] = {}
    for path in sorted(canonical_root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"shared analysis source contains symlink: {path}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"shared analysis source contains non-regular file: {path}")
        resolved = path.resolve()
        if resolved != canonical_root and canonical_root not in resolved.parents:
            raise ValueError(f"shared analysis source member escapes root: {path}")
        files[path.relative_to(canonical_root).as_posix()] = _sha256_file(path)
    serialized_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(canonical_root),
        "identity": hashlib.sha256(serialized_files).hexdigest(),
        "files": files,
    }


def _read_regular_file(path: Path) -> bytes:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"shared analysis source is not a regular file: {path}")
    return path.read_bytes()


def _hash_regular_paths(paths: tuple[Path, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        hashes[str(path)] = hashlib.sha256(_read_regular_file(path)).hexdigest()
    return hashes


def _hash_existing_paths(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): _sha256_file(path) for path in paths if path.is_file()}


def _checkpoint_hashes(config: ExpandedEnsembleConfig) -> dict[str, str]:
    checkpoints: dict[str, str] = {}
    for ablation_config in config.ablation_configs:
        checkpoint = config.checkpoint_path(ablation_config)
        checkpoints[str(checkpoint)] = hashlib.sha256(_read_regular_file(checkpoint)).hexdigest()
    return checkpoints


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
