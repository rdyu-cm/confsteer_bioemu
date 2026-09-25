"""Per-sequence property reference statistics for relative-level steering targets.

The relative-level target ``t = mean_seq + level * k * sigma_seq`` (design decisions
D3/D4) needs one ``(mean, sigma)`` pair per sequence. Those statistics are measured
once from cached adapter-off ensembles and then declared in configuration; this module
loads that declaration and fails closed on anything ambiguous.

Two on-disk formats are accepted, both keyed by ``sequence_id``:

``.csv``
    The per-sequence sidecar that :mod:`sae_bioemu.steering.rg_calibration` already
    writes (``<stem>_sequences.csv``). Its ``store_label``, ``sequence_id``,
    ``sample_count``, ``mean_radius_of_gyration_angstrom`` and
    ``sd_radius_of_gyration_angstrom`` columns are consumed directly, so the radius
    of gyration is never recomputed here or defined twice.

``.json``
    ``{"schema_version": 1, "property_name": ..., "units": ...,
    "sequences": {"<id>": {"mean": ..., "sigma": ..., "sample_count": ...,
    "store_label": ...}}}``, for properties without a calibration CSV.

Per D5 the train and validation sequences take their statistics from the 32-sample
train-FASTA ensemble while test sequences take theirs from their own 128-sample
eval-FASTA ensemble. That asymmetry is deliberate, so ``store_label`` and
``sample_count`` are retained per sequence and a split whose sequences do not all
share one ``store_label`` is reported loudly rather than silently accepted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

__all__ = [
    "ANGSTROM_PROPERTY_UNITS",
    "DEFAULT_SIGMA_ELIGIBILITY",
    "DIMENSIONLESS_PROPERTY_UNITS",
    "MINIMUM_SAMPLE_COUNT",
    "STEERABLE_PROPERTY_UNITS",
    "PropertyReferenceTable",
    "SequencePropertyReference",
    "SigmaEligibility",
    "csv_statistic_columns",
    "default_sigma_eligibility_applies",
    "load_property_reference_table",
]

#: ``property_name -> (mean column, standard-deviation column, units)`` for the CSV form.
_CSV_STATISTIC_COLUMNS: dict[str, tuple[str, str, str]] = {
    "radius_of_gyration": (
        "mean_radius_of_gyration_angstrom",
        "sd_radius_of_gyration_angstrom",
        "angstrom",
    ),
}

#: An sd from fewer than two samples is not an estimate of anything.
MINIMUM_SAMPLE_COUNT = 2

_JSON_KEYS = frozenset({"schema_version", "property_name", "units", "sequences"})
_JSON_SEQUENCE_KEYS = frozenset({"mean", "sigma", "sample_count", "store_label"})
_MAXIMUM_REPORTED_IDS = 10


def _is_finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def csv_statistic_columns(property_name: str) -> tuple[str, str, str]:
    """Return ``(mean column, standard-deviation column, units)`` for a property's CSV.

    Split construction reads the same calibration sidecar this module consumes, so the
    column names are declared once here rather than restated wherever the sidecar is
    parsed.
    """
    columns = _CSV_STATISTIC_COLUMNS.get(property_name)
    if columns is None:
        raise ValueError(
            f"no CSV column mapping is declared for property {property_name!r}; add one to "
            "sae_bioemu.steering.property_reference or supply a JSON reference table"
        )
    return columns


@dataclass(frozen=True)
class SequencePropertyReference:
    """One sequence's adapter-off ensemble mean and within-ensemble spread."""

    sequence_id: str
    mean: float
    sigma: float
    sample_count: int
    store_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("property reference sequence_id must be a non-empty string")
        if not isinstance(self.store_label, str) or not self.store_label:
            raise ValueError(
                f"property reference store_label for {self.sequence_id!r} must be non-empty"
            )
        if not _is_finite(self.mean):
            raise ValueError(f"property reference mean for {self.sequence_id!r} must be finite")
        if not _is_finite(self.sigma) or self.sigma <= 0:
            raise ValueError(
                f"property reference sigma for {self.sequence_id!r} must be strictly positive "
                f"and finite, found {self.sigma!r}"
            )
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or self.sample_count < MINIMUM_SAMPLE_COUNT
        ):
            raise ValueError(
                f"property reference sample_count for {self.sequence_id!r} must be an integer "
                f"of at least {MINIMUM_SAMPLE_COUNT}, found {self.sample_count!r}"
            )

    def payload(self) -> dict[str, object]:
        """Return the canonical primitive record used for content fingerprinting."""
        return {
            "mean": float(self.mean),
            "sigma": float(self.sigma),
            "sample_count": int(self.sample_count),
            "store_label": self.store_label,
        }


@dataclass(frozen=True)
class SigmaEligibility:
    """The inclusive within-ensemble spread band a sequence must fall in (D12).

    Measured Rg spread on this dataset covers nearly three orders of magnitude, so a
    uniform ``k * sigma_seq`` dose is not meaningful at either extreme: below the band a
    one-sigma request is smaller than Rg meaningfully resolves while the sigma-normalized
    loss explodes on tiny absolute deviations, and above it a single ensemble mean is a
    poor summary of a disordered ensemble.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        for name in ("lower", "upper"):
            if not _is_finite(getattr(self, name)):
                raise ValueError(f"sigma eligibility {name} bound must be finite")
        if self.lower <= 0:
            raise ValueError("sigma eligibility lower bound must be positive")
        if self.upper <= self.lower:
            raise ValueError("sigma eligibility upper bound must exceed the lower bound")

    def contains(self, sigma: float) -> bool:
        """Return whether one sequence's spread lies in the inclusive band."""
        return self.lower <= float(sigma) <= self.upper

    def identity(self) -> dict[str, float]:
        """Return the primitive record bound into the checkpoint identity."""
        return {"lower": float(self.lower), "upper": float(self.upper)}


#: D12's approved default band, in the property's own units (Angstrom for Rg).
DEFAULT_SIGMA_ELIGIBILITY = SigmaEligibility(0.5, 8.0)

ANGSTROM_PROPERTY_UNITS = "angstrom"
DIMENSIONLESS_PROPERTY_UNITS = "dimensionless"

#: The units every steerable property's sigma is measured in. This exists so the eligibility
#: default can be keyed on a property CLASS rather than on one property's NAME.
#:
#: :data:`DEFAULT_SIGMA_ELIGIBILITY` is a band on the radius of gyration's own sigma, in
#: Angstrom. Applying it to a progress coordinate compares a dimensionless ratio against a
#: length: on the committed kinase panel, whose sigma spans 0.027-0.205, the 0.5-8.0 band
#: excludes every protein, and it does so by numerical accident rather than by any check. The
#: guard was previously written as ``property_name == "pocket_progress_coordinate"``, which
#: covered the one instance that existed and not the next one -- ``dfg_progress_coordinate``
#: is the same kind of quantity and would have taken the Angstrom default silently.
#:
#: The two progress coordinates are spelled out rather than imported. ``cryptic_steering.arm``
#: and ``kinase_dfg.objective`` both import steering modules, so importing them back would be a
#: cycle; ``cryptic_steering.calibration`` duplicates the same string for the same reason. A
#: test pins these keys against the canonical constants so the copies cannot drift.
STEERABLE_PROPERTY_UNITS: Mapping[str, str] = {
    "radius_of_gyration": ANGSTROM_PROPERTY_UNITS,
    "pocket_progress_coordinate": DIMENSIONLESS_PROPERTY_UNITS,
    "dfg_progress_coordinate": DIMENSIONLESS_PROPERTY_UNITS,
}


def default_sigma_eligibility_applies(property_name: str) -> bool:
    """Return whether :data:`DEFAULT_SIGMA_ELIGIBILITY` is a band in this property's own units.

    True only for a property registered as Angstrom-valued. An unregistered property is
    False, not True: a new steerable property must state its band or state its units, and
    "not in the table" must not be the way it inherits a band from the radius of gyration.
    """
    return STEERABLE_PROPERTY_UNITS.get(property_name) == ANGSTROM_PROPERTY_UNITS


@dataclass(frozen=True)
class PropertyReferenceTable:
    """A validated, content-fingerprinted per-sequence reference table."""

    path: Path
    property_name: str
    units: str
    source_format: str
    file_sha256: str
    content_fingerprint: str
    entries: Mapping[str, SequencePropertyReference]

    def __contains__(self, sequence_id: object) -> bool:
        return sequence_id in self.entries

    def statistics(self, sequence_id: str) -> SequencePropertyReference:
        """Return one sequence's statistics, failing closed when it is absent."""
        try:
            return self.entries[sequence_id]
        except KeyError as error:
            raise KeyError(
                f"property reference table {self.path} has no entry for sequence "
                f"{sequence_id!r}"
            ) from error

    def require_coverage(self, sequence_ids: Iterable[str], *, context: str) -> None:
        """Fail closed unless every requested sequence has a reference entry."""
        requested: list[str] = []
        for sequence_id in sequence_ids:
            if sequence_id not in requested:
                requested.append(sequence_id)
        missing = tuple(sequence_id for sequence_id in requested if sequence_id not in self.entries)
        if missing:
            raise ValueError(
                f"property reference table {self.path} is missing {len(missing)} of "
                f"{len(requested)} {context} sequences; relative-level targets cannot be "
                "constructed without a per-sequence mean and sigma. First missing sequence "
                f"IDs: {', '.join(missing[:_MAXIMUM_REPORTED_IDS])}"
            )

    def eligible_sequence_ids(self, eligibility: SigmaEligibility) -> tuple[str, ...]:
        """Return the sequence IDs whose spread lies inside the eligibility band.

        This is the helper split construction should use: filtering belongs where the
        split is chosen, never inside a workflow that would then silently prune it.
        """
        if not isinstance(eligibility, SigmaEligibility):
            raise TypeError("eligibility must be a SigmaEligibility")
        return tuple(
            sequence_id
            for sequence_id, entry in self.entries.items()
            if eligibility.contains(entry.sigma)
        )

    def ineligible_sequence_ids(self, eligibility: SigmaEligibility) -> tuple[str, ...]:
        """Return the sequence IDs whose spread falls outside the eligibility band."""
        eligible = set(self.eligible_sequence_ids(eligibility))
        return tuple(sequence_id for sequence_id in self.entries if sequence_id not in eligible)

    def require_eligibility(
        self,
        sequence_ids: Iterable[str],
        eligibility: SigmaEligibility,
        *,
        context: str,
    ) -> None:
        """Fail closed when any requested sequence lies outside the eligibility band.

        The operator chooses the split, so an ineligible member is an operator error and
        is reported rather than dropped: silently pruning it would change the evaluated
        population without saying so.
        """
        if not isinstance(eligibility, SigmaEligibility):
            raise TypeError("eligibility must be a SigmaEligibility")
        requested: list[str] = []
        for sequence_id in sequence_ids:
            if sequence_id not in requested:
                requested.append(sequence_id)
        offenders = tuple(
            (sequence_id, self.entries[sequence_id].sigma)
            for sequence_id in requested
            if sequence_id in self.entries
            and not eligibility.contains(self.entries[sequence_id].sigma)
        )
        if offenders:
            listed = ", ".join(
                f"{sequence_id} (sigma={sigma:g})"
                for sequence_id, sigma in offenders[:_MAXIMUM_REPORTED_IDS]
            )
            raise ValueError(
                f"{len(offenders)} of {len(requested)} {context} sequences fall outside the "
                f"configured sigma eligibility band [{eligibility.lower:g}, "
                f"{eligibility.upper:g}]: a uniform relative dose is not meaningful there "
                "(D12). Rebuild the split from eligible_sequence_ids rather than dropping "
                f"them silently. First offending sequence IDs: {listed}"
            )

    def eligibility_summary(
        self, eligibility: SigmaEligibility, splits: Mapping[str, Sequence[str]] | None = None
    ) -> dict[str, object]:
        """Return the applied bounds with eligible/ineligible counts for the run report."""
        eligible = set(self.eligible_sequence_ids(eligibility))
        summary: dict[str, object] = {
            "bounds": eligibility.identity(),
            "table_sequence_count": len(self.entries),
            "table_eligible_count": len(eligible),
            "table_ineligible_count": len(self.entries) - len(eligible),
        }
        if splits is not None:
            summary["by_split"] = {
                split: {
                    "sequence_count": len(splits[split]),
                    "eligible_count": sum(
                        1 for sequence_id in splits[split] if sequence_id in eligible
                    ),
                    "ineligible_count": sum(
                        1
                        for sequence_id in splits[split]
                        if sequence_id in self.entries and sequence_id not in eligible
                    ),
                }
                for split in sorted(splits)
            }
        return summary

    def store_label_warnings(self, splits: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
        """Return one message per split whose sequences do not share a single store label.

        D5 expects one store per split: train and validation from the 32-sample train
        ensemble, test from the 128-sample eval ensemble. A mixed split is reported, not
        rejected, because the statistics themselves are still individually valid.
        """
        messages: list[str] = []
        inventory = self.split_inventory(splits)
        for split in sorted(inventory):
            labels = inventory[split]
            if len(labels) > 1:
                detail = ", ".join(
                    f"{label}={record['sequence_count']} sequences "
                    f"(sample_count {record['sample_counts']})"
                    for label, record in sorted(labels.items())
                )
                messages.append(
                    f"property reference split {split!r} mixes {len(labels)} store labels: "
                    f"{detail}. D5 expects one ensemble per split; check the reference "
                    "table before trusting the reported per-split sampling error."
                )
        return tuple(messages)

    def split_inventory(
        self, splits: Mapping[str, Sequence[str]]
    ) -> dict[str, dict[str, dict[str, object]]]:
        """Return an auditable per-split store-label and sample-count inventory."""
        inventory: dict[str, dict[str, dict[str, object]]] = {}
        for split in sorted(splits):
            entries = tuple(
                entry
                for entry in (self.entries.get(sequence_id) for sequence_id in splits[split])
                if entry is not None
            )
            labels = sorted({entry.store_label for entry in entries})
            inventory[split] = {
                label: {
                    "sequence_count": sum(1 for entry in entries if entry.store_label == label),
                    "sample_counts": sorted(
                        {entry.sample_count for entry in entries if entry.store_label == label}
                    ),
                }
                for label in labels
            }
        return inventory

    def identity(self) -> dict[str, object]:
        """Return the content-bound identity block for the training checkpoint.

        The filesystem path is deliberately excluded: the same reviewed table copied to
        another checkout must produce the same checkpoint identity, while any change to
        its contents must not.
        """
        return {
            "property_name": self.property_name,
            "units": self.units,
            "source_format": self.source_format,
            "sequence_count": len(self.entries),
            "file_sha256": self.file_sha256,
            "content_fingerprint": self.content_fingerprint,
        }


def load_property_reference_table(
    path: str | Path, *, property_name: str
) -> PropertyReferenceTable:
    """Load and validate a per-sequence property reference table."""
    if not isinstance(property_name, str) or not property_name:
        raise ValueError("property_name must be a non-empty string")
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"property reference table is unreadable: {resolved}") from error
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        entries, units = _parse_csv(raw, resolved, property_name)
    elif suffix == ".json":
        entries, units = _parse_json(raw, resolved, property_name)
    else:
        raise ValueError(
            f"property reference table must be a .csv or .json file, found {resolved.name!r}"
        )
    if not entries:
        raise ValueError(f"property reference table {resolved} contains no sequences")
    ordered = {sequence_id: entries[sequence_id] for sequence_id in sorted(entries)}
    content = {
        "property_name": property_name,
        "units": units,
        "sequences": {sequence_id: entry.payload() for sequence_id, entry in ordered.items()},
    }
    return PropertyReferenceTable(
        path=resolved,
        property_name=property_name,
        units=units,
        source_format=suffix.lstrip("."),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        content_fingerprint=hashlib.sha256(_canonical_json_bytes(content)).hexdigest(),
        entries=ordered,
    )


def _parse_csv(
    raw: bytes, path: Path, property_name: str
) -> tuple[dict[str, SequencePropertyReference], str]:
    columns = _CSV_STATISTIC_COLUMNS.get(property_name)
    if columns is None:
        raise ValueError(
            f"no CSV column mapping is declared for property {property_name!r}; add one to "
            "sae_bioemu.steering.property_reference or supply a JSON reference table"
        )
    mean_column, sigma_column, units = columns
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"property reference table {path} is not UTF-8") from error
    reader = csv.DictReader(text.splitlines())
    required = {"sequence_id", "sample_count", "store_label", mean_column, sigma_column}
    header = set(reader.fieldnames or ())
    missing_columns = sorted(required - header)
    if missing_columns:
        raise ValueError(
            f"property reference CSV {path} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )
    entries: dict[str, SequencePropertyReference] = {}
    for line_number, row in enumerate(reader, start=2):
        sequence_id = (row.get("sequence_id") or "").strip()
        if not sequence_id:
            raise ValueError(f"{path}:{line_number}: sequence_id is required")
        if sequence_id in entries:
            raise ValueError(
                f"{path}:{line_number}: duplicate property reference row for sequence "
                f"{sequence_id!r}; one sequence must have exactly one reference ensemble"
            )
        entries[sequence_id] = SequencePropertyReference(
            sequence_id=sequence_id,
            mean=_float(row.get(mean_column), f"{path}:{line_number}: {mean_column}"),
            sigma=_float(row.get(sigma_column), f"{path}:{line_number}: {sigma_column}"),
            sample_count=_integer(row.get("sample_count"), f"{path}:{line_number}: sample_count"),
            store_label=(row.get("store_label") or "").strip(),
        )
    return entries, units


def _parse_json(
    raw: bytes, path: Path, property_name: str
) -> tuple[dict[str, SequencePropertyReference], str]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"property reference table {path} is not valid JSON") from error
    if not isinstance(payload, Mapping) or frozenset(payload) != _JSON_KEYS:
        raise ValueError(
            f"property reference JSON {path} must contain exactly "
            f"{', '.join(sorted(_JSON_KEYS))}"
        )
    if payload["schema_version"] != 1:
        raise ValueError(f"unsupported property reference schema version in {path}")
    if payload["property_name"] != property_name:
        raise ValueError(
            f"property reference JSON {path} declares property "
            f"{payload['property_name']!r} but the run steers {property_name!r}"
        )
    units = payload["units"]
    if not isinstance(units, str) or not units:
        raise ValueError(f"property reference JSON {path} requires non-empty units")
    expected = _CSV_STATISTIC_COLUMNS.get(property_name)
    if expected is not None and units != expected[2]:
        raise ValueError(
            f"property reference JSON {path} declares units {units!r} but "
            f"{property_name!r} is measured in {expected[2]!r}"
        )
    sequences = payload["sequences"]
    if not isinstance(sequences, Mapping):
        raise ValueError(f"property reference JSON {path} sequences must be an object")
    entries: dict[str, SequencePropertyReference] = {}
    for sequence_id in sequences:
        record = sequences[sequence_id]
        if not isinstance(record, Mapping) or frozenset(record) != _JSON_SEQUENCE_KEYS:
            raise ValueError(
                f"property reference JSON {path} entry {sequence_id!r} must contain exactly "
                f"{', '.join(sorted(_JSON_SEQUENCE_KEYS))}"
            )
        entries[str(sequence_id)] = SequencePropertyReference(
            sequence_id=str(sequence_id),
            mean=_float(record["mean"], f"{path}: {sequence_id} mean"),
            sigma=_float(record["sigma"], f"{path}: {sequence_id} sigma"),
            sample_count=_integer(record["sample_count"], f"{path}: {sequence_id} sample_count"),
            store_label=record["store_label"] if isinstance(record["store_label"], str) else "",
        )
    return entries, units


def _float(value: object, name: str) -> float:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise ValueError(f"{name} must be a finite number") from error
    if not _is_finite(value):
        raise ValueError(f"{name} must be a finite number, found {value!r}")
    return float(value)  # type: ignore[arg-type]


def _integer(value: object, name: str) -> int:
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, found {value!r}")
    return int(value)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
