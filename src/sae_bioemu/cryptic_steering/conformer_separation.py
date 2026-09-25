"""The two admission gates a cryptic system must clear, and the strata they sort a pool into.

TWO GATES, AND NEITHER IMPLIES THE OTHER
-----------------------------------------

GATE 1, ``s > tau``. The two DEPOSITED conformers are distinguishable at the threshold the metric
is graded at. Without it the task is not well posed: apo and holo are one structure to the scorer.

GATE 2, ``m_unsteered > tau``. The UNSTEERED BioEmu draw does not already land inside ``tau`` of
holo. Without it the METRIC does not discriminate: the arm is scored on systems it gets for free.

These are different quantities and gating one does not gate the other, which is measured rather
than argued. The domain-motion programme gates on ``s`` alone
(:mod:`sae_bioemu.domain_motion.separation`, keeping 418 of 1476 at ``s > 3.0``) and three of the
six systems on its own D-94 panel were still inside ``tau`` unsteered -- ``NUCLE1R1`` at
``s = 17.100`` and ``m_unsteered = 1.079``. A BioEmu draw does not start on the other conformer, so
a large crystallographic separation says nothing about where the sampler actually sits. On this
pool, over the graded region:

    s <= tau = 1.5  (crystallographic, 1920 systems)     1191/1920   62.0%
    m_unsteered <= tau = 1.5  (unsteered draw)             138/367   37.6%
    m_unsteered <= tau = 1.0                                43/367   11.7%

and on the 367 systems where both are measured the two gates overlap without subsuming: at
``tau = 1.5``, 110 fail both, 104 fail gate 1 only and **28 fail gate 2 only**. Gate 2 is the one
:mod:`sae_bioemu.domain_motion.gap_closed` added as ``admitted_at_tau``, and this module follows
that precedent so the two programmes agree on what a scorable system is: the rule is BASELINE-ONLY
and fixed before any adapter runs, so it cannot select for systems an arm happens to do well on.

Sections below take the two gates in turn.

GATE 1: THE DEFECT IT REMOVES
------------------------------

The cryptic arm grades a draw by its superposed N/CA/C/O deviation from a deposited conformer, and
calls a system a success when some draw lands within ``tau`` of holo. That is the field's
instrument and :mod:`~sae_bioemu.cryptic_steering.success_at_b` implements it. It presumes
something nobody had measured on this pool: that apo and holo are far enough apart for "within
``tau`` of holo" to be a statement about holo.

Measured over the 1920 systems of the D-38 metric reference, on exactly the region the arm grades
(``pocket_indices | context_indices``, restricted to residues modelled in both crystals) and
through exactly the kernel it grades with:

    apo-holo RMSD over the graded region:  median 1.2553 A
      <= tau = 1.0 :  643/1920 (33.5%)
      <= tau = 1.5 : 1191/1920 (62.0%)

For 62% of the pool at the BioEmu ``tau`` the two crystal conformers are closer to each other than
the success threshold. **Sitting at apo already scores as a holo success.** An adapter cannot learn
a distinction its data does not contain, and a success rate computed over that pool is mostly
measuring the baseline that would be obtained by not steering at all.

GATE 1: THE RULE
----------------

A system is admitted only when ``s > tau``, where ``s`` is its aligned backbone deviation between
the two deposited conformers over its own metric region. ``tau`` is supplied by the caller and has
no default here, for the reason it has none anywhere in this project: ``tau`` and ``B`` dominate
the number, and :data:`~sae_bioemu.cryptic_steering.direct_metric.BENCHMARK_TAU_ANGSTROM` carries
two values of record rather than one. The realised cut is recorded on the census so an artifact
states the number the pool was actually cut at rather than the name of a parameter.

GATE 1: WHY ``tau`` AND NOT ``2*tau``, WHICH IS THE RIGOROUS CUT
-----------------------------------------------------------------

This is the non-obvious part and it is the reason the rule is written here rather than assumed.

The ``tau``-ball around apo and the ``tau``-ball around holo are disjoint **iff** ``s > 2*tau``, by
the triangle inequality: if ``d(x, apo) <= tau`` and ``d(x, holo) <= tau`` then ``s <= 2*tau``. So
``s > 2*tau`` is the only cut above which a holo success is *provably* not also an apo success.
The rigorous admission rule is therefore ``s > 2*tau``, and it is not the one used.

The reason is measured rather than argued. Taking the cut up moves the pool away from the
phenomenon under study. Pocket enrichment -- the share of the apo->holo change that falls inside
the pocket residues, relative to the pocket's share of the graded region -- INVERTS with the cut:

    cut s>   kept    med enrichment   enrich>1    top-decile share
    0.0      1920        1.129          54.5%          0.657
    1.5       729        0.954          47.2%          0.669
    3.0       180        0.635          25.6%          0.664

The top-decile share is flat at ~0.66 at every cut, so the apo->holo change stays equally
CONCENTRATED as the cut rises -- it simply stops being concentrated in the pocket. Systems with
large apo->holo motion are domain and loop rearrangements, not pocket openings. Cutting at
``2*tau`` selects against cryptic pocket opening, the thing the programme exists to steer, and
leaves 117 train / 14 validation systems: a pool too small to train on and pointed at the wrong
phenomenon.

The rigour of ``2*tau`` is therefore taken from the SCORER instead, as the specificity criterion
``within tau of holo AND NOT within tau of apo`` (see
:func:`~sae_bioemu.cryptic_steering.success_at_b.specific_success_at_b`). That criterion subsumes
the harder cut: above ``s > 2*tau`` every holo success satisfies it automatically, so ``2*tau`` is
the special case in which plain and specific success coincide. Below it the criterion is what
restores specificity, per sample, on the systems the ``tau`` cut admits and the ``2*tau`` cut would
have discarded.

GATE 1: THE BAND IS A STRATUM, NEVER A FILTER
----------------------------------------------

Admitted systems therefore divide in two, and every result carries the division:

* :data:`STRATUM_CLEAN` -- ``s > 2*tau``. The two ``tau``-balls are disjoint; a holo success cannot
  also be an apo success, so plain and specific success are the same number by construction.
* :data:`STRATUM_AMBIGUOUS` -- ``tau < s <= 2*tau``. Admitted, and a draw inside both balls exists
  geometrically. Plain and specific success may differ here, and the gap between them is exactly
  the mass this stratum contributes.

and the refused systems are :data:`STRATUM_DEGENERATE` -- ``s <= tau``, where apo is itself a holo
success. :data:`STRATUM_UNKNOWN` exists because a system whose ``s`` could not be measured is not
evidence of anything and must not be quietly counted as clean.

The band is reported and never filtered. Dropping it would be the ``2*tau`` cut wearing a
stratification's clothes, and section 2 above is the measurement of what that costs.

GATE 2: THE BASELINE WITNESS, AND WHY ONE DRAW IS ENOUGH TO REFUSE
-------------------------------------------------------------------

``m_unsteered`` needs an adapter-off draw, and this project does not generate one to answer this
question. It reads the arm's OWN training trace: a checkpoint's ``validation_record``
``training_records`` carry ``metric_deviation_angstrom`` beside the ``residual_fraction`` the
adapter actually delivered on that unit, and the rows at the very start of a run come from an
adapter at initialisation, which injects essentially nothing. Those draws ARE BioEmu untouched.
:data:`AMBIENT_RESIDUAL_FRACTION_CEILING` is the threshold, copied by value from D-93's
``AMBIENT_DOSE_CEILING`` so the two analyses mean the same thing by "ambient".

**The ceiling is load-bearing and its effect is measured rather than assumed.** On the
``cryptic_lg20`` holo rows it retains, ``residual_fraction`` runs 0.0 / 8.9e-05 / **0.0045** /
0.0144 / 0.0310 / 0.0499 at min / p25 / median / p75 / p90 / max, with 124 of 367 rows above 0.01:
these draws carry a small realised dose, not literally none. D-93 is the evidence that reading them
as ambient is sound -- it measured this same population's realised alignment at ``|cos| = 0.154``
against BioEmu's untouched 0.173, i.e. at the floor and distinguishable from a trained arm's 0.214
-- and the ceiling's own sensitivity says the same thing from the other side. Tightening it moves
COVERAGE and not the answer: at ``tau = 1.5`` the share of witnessed systems that gate 2 refuses is
35% at a ceiling of 1e-4 (33 of 94), 36% at 1e-3 (53 of 147), 37% at 1e-2 (90 of 243) and 38% at
the 0.05 of record (138 of 367). The ambient population is homogeneous with respect to the question
being asked of it, so the wider ceiling buys coverage at no measurable bias.

**The witness is the MINIMUM of a sequence's ambient draws, not their median**, and that is the one
place this module departs from :mod:`sae_bioemu.domain_motion.gap_closed`, which takes a median.
The reason is that the two are reducing different statistics: ``fraction_of_gap_closed`` is a
central tendency, so its baseline is one too, while ``success@B`` is a MAXIMUM over ``B`` draws --
a system succeeds when ANY draw lands inside ``tau``. The quantity that makes a system a free pass
is therefore its BEST unsteered draw, and one draw inside ``tau`` is a witness that the unsteered
ensemble already succeeds rather than an estimate that it typically does.

That makes the gate one-sided in the safe direction, twice over:

* more ambient draws can only find MORE systems already inside ``tau``, never fewer, so a thin
  trace under-detects degeneracy and never invents it; and
* the journalled deviation is
  :data:`~sae_bioemu.cryptic_steering.direct_metric.TRAINED_DEVIATION_FORM`, the ROOT-mean-square,
  while the benchmark grades the arithmetic MEAN, which by Jensen is never larger. A draw whose RMS
  is inside ``tau`` is inside ``tau`` in the graded form too.

So a refusal under gate 2 is a fact about the system, and an admission under it is only the absence
of a witness. :data:`STRATUM_BASELINE_UNMEASURED` exists to keep those two apart, and is carried on
every admitted system rather than collapsed into "passed".

GATE 2: WHAT ITS COVERAGE ACTUALLY IS ON THIS POOL, MEASURED
-------------------------------------------------------------

Measured on the ``cryptic_lg20`` trace: 5710 training records over the 1306 train-role sequences,
of which **740 are ambient and they cover 373 sequences, 367 of them at the holo level, with
exactly one ambient draw each**. Every one is in the parent split's ``train`` role; validation and
test have no ambient row at all, because the run journals only training units.

That is 367 of 1920 pool systems, 19.1%. The gate is applied anyway, and the reason it is safe to
apply at that coverage is the one-sidedness above: it deletes 28 systems (at ``tau = 1.5``) that
gate 1 admits and a witnessed unsteered draw already succeeds on, and it leaves every unwitnessed
system in. It is NOT a certificate that the survivors are baseline-separated, and no artifact this
module writes says that it is: the survivors carry
:data:`STRATUM_BASELINE_PASSED` or :data:`STRATUM_BASELINE_UNMEASURED` and the counts are reported
separately. A future arm that generates matched adapter-off controls for every scheduled protein
can promote the stratum into a full gate; until then, "81% of the admitted pool is unverified on
gate 2" is a sentence every consumer of this pool has to read.

WHERE THIS FILE SITS IN THE FINGERPRINT SCHEME
-----------------------------------------------

In no code group, deliberately, exactly like :mod:`~sae_bioemu.cryptic_steering.split`. This module
decides which proteins a split record CONTAINS; the record is then an identity-bearing artifact
whose own fingerprint rides ``schedule_seed_fingerprint``, so a change to this rule reaches a
checkpoint through the record it produced and not through a code digest. Nothing training or
evaluation executes imports it at module scope.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sae_bioemu.cryptic_steering.direct_metric import CrypticMetricReference

__all__ = [
    "AMBIENT_RESIDUAL_FRACTION_CEILING",
    "BASELINE_STRATA",
    "HOLO_LEVEL",
    "SEPARATION_STRATA",
    "STRATUM_AMBIGUOUS",
    "STRATUM_BASELINE_DEGENERATE",
    "STRATUM_BASELINE_PASSED",
    "STRATUM_BASELINE_UNMEASURED",
    "STRATUM_CLEAN",
    "STRATUM_DEGENERATE",
    "STRATUM_UNKNOWN",
    "AdmissionCensus",
    "BaselineCensus",
    "BaselineWitnesses",
    "SeparationCensus",
    "admission_census",
    "baseline_census",
    "baseline_stratum",
    "baseline_witnesses",
    "checked_tau",
    "is_admitted",
    "pool_separations",
    "separation_census",
    "separation_stratum",
]

#: ``s <= tau``: the two deposited conformers are closer to each other than the success threshold,
#: so the UNSTEERED apo structure already scores as a holo success. Refused by the admission rule.
STRATUM_DEGENERATE = "degenerate"

#: ``tau < s <= 2*tau``: admitted, and the two ``tau``-balls still overlap, so a draw can satisfy
#: both references at once. Plain and specific success may differ here. Never filtered.
STRATUM_AMBIGUOUS = "ambiguous"

#: ``s > 2*tau``: the two ``tau``-balls are disjoint, so a holo success is provably not an apo
#: success and the plain and specific criteria coincide.
STRATUM_CLEAN = "clean"

#: No ``s`` could be measured for this system. Reported as its own stratum rather than folded into
#: any of the other three: an unmeasured separation is not evidence that the pair is separated.
STRATUM_UNKNOWN = "unknown"

#: Reporting order: worst to best, then the unmeasured.
SEPARATION_STRATA: tuple[str, ...] = (
    STRATUM_DEGENERATE,
    STRATUM_AMBIGUOUS,
    STRATUM_CLEAN,
    STRATUM_UNKNOWN,
)

#: GATE 2. ``m_unsteered <= tau``: an adapter-off draw was WITNESSED inside ``tau`` of holo, so the
#: unsteered ensemble already succeeds under ``success@B``. Refused.
STRATUM_BASELINE_DEGENERATE = "baseline_degenerate"

#: GATE 2. Every witnessed adapter-off draw sits outside ``tau``. Admitted -- but read the module
#: docstring: with one draw per sequence this is the absence of a witness, not a certificate.
STRATUM_BASELINE_PASSED = "baseline_passed"

#: GATE 2. No adapter-off draw exists for this system, so gate 2 was never evaluated on it. Carried
#: as its own stratum on every admitted system: 81% of this pool is in it, and an artifact that
#: folded it into :data:`STRATUM_BASELINE_PASSED` would claim a gate it never applied.
STRATUM_BASELINE_UNMEASURED = "baseline_unmeasured"

#: Reporting order: refused, admitted, then the unevaluated.
BASELINE_STRATA: tuple[str, ...] = (
    STRATUM_BASELINE_DEGENERATE,
    STRATUM_BASELINE_PASSED,
    STRATUM_BASELINE_UNMEASURED,
)

#: The direct-metric level naming the HOLO conformer. Gate 2 asks whether the unsteered draw
#: already succeeds, and ``success@B`` on this programme is success at reaching HOLO, so the apo
#: level's rows measure a different distance and are not the baseline this gate is about. Copied by
#: value rather than imported from ``direct_metric`` for the reason that module copies
#: ``MINIMUM_SEPARATION_ANGSTROM``: a pool derivation must not pull a training-only source into its
#: import graph.
HOLO_LEVEL = 1.0

#: A training record whose realised ``residual_fraction`` is at or below this is read as an
#: ADAPTER-OFF draw. The value is D-93's ``AMBIENT_DOSE_CEILING``
#: (``scripts/steering/measure_realised_alignment.py``), copied so that "ambient" means one thing
#: across the two analyses. It is load-bearing -- the retained rows run up to 0.0499 -- and the
#: module docstring carries both the realised distribution and the sensitivity sweep that shows
#: tightening it moves coverage rather than the answer.
AMBIENT_RESIDUAL_FRACTION_CEILING = 0.05


def checked_tau(tau_angstrom: object) -> float:
    """Return ``tau`` as a positive finite float, or refuse.

    There is deliberately no default. ``tau`` is an EVALUATION parameter with two values of record
    in this project, and a rule that silently adopted one of them would make a pool built at 1.0 A
    and a pool built at 1.5 A indistinguishable in every artifact that did not happen to print it.
    """
    if isinstance(tau_angstrom, bool) or not isinstance(tau_angstrom, Real):
        raise TypeError(
            f"tau_angstrom must be a real number of Angstrom; got {type(tau_angstrom).__name__}"
        )
    value = float(tau_angstrom)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"tau_angstrom must be finite and positive; got {value!r}")
    return value


def _checked_separation(separation_angstrom: object, *, name: str) -> float:
    """Return one measured ``s`` as a finite non-negative float, or refuse."""
    if isinstance(separation_angstrom, bool) or not isinstance(separation_angstrom, Real):
        raise TypeError(
            f"{name}: separation must be a real number of Angstrom; got "
            f"{type(separation_angstrom).__name__}"
        )
    value = float(separation_angstrom)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name}: separation must be finite and non-negative; got {value!r}")
    return value


def is_admitted(separation_angstrom: object, *, tau_angstrom: float) -> bool:
    """Whether a system clears the admission rule: ``s > tau``, strictly.

    Strict, and the strictness carries meaning rather than being a boundary convention: at
    ``s == tau`` an apo-shaped draw sits exactly on the success threshold and the system supplies
    no margin at all.
    """
    return _checked_separation(separation_angstrom, name="system") > checked_tau(tau_angstrom)


def separation_stratum(separation_angstrom: object | None, *, tau_angstrom: float) -> str:
    """Return which of :data:`SEPARATION_STRATA` a measured ``s`` falls in.

    ``None`` -- an ``s`` that could not be measured -- returns :data:`STRATUM_UNKNOWN` rather than
    raising, because the systems this happens to are real (five of the benchmark's own) and must
    reach a report as a named population rather than as a missing row.
    """
    tau = checked_tau(tau_angstrom)
    if separation_angstrom is None:
        return STRATUM_UNKNOWN
    value = _checked_separation(separation_angstrom, name="system")
    if value <= tau:
        return STRATUM_DEGENERATE
    if value <= 2.0 * tau:
        return STRATUM_AMBIGUOUS
    return STRATUM_CLEAN


@dataclass(frozen=True)
class SeparationCensus:
    """One population sorted by the admission rule at one ``tau``, with the realised cut recorded.

    ``separations`` maps every system considered to its measured ``s``, or to ``None`` when none
    could be measured. Membership is the census: a system absent from it was never considered, and
    a system present with ``None`` was considered and could not be measured. The two are different
    facts and are not merged.
    """

    tau_angstrom: float
    #: The Angstrom value the pool was actually cut at. Equal to ``tau_angstrom`` by construction
    #: -- see the module docstring on why the cut is not ``2*tau`` -- and written out separately so
    #: an artifact states a number rather than the name of a parameter.
    realised_cut_angstrom: float
    #: The boundary of the ambiguous band, ``2*tau``. Not a cut; the band above ``tau`` and at or
    #: below this value is admitted and stratified.
    ambiguous_band_upper_angstrom: float
    separations: dict[str, float | None]

    def stratum(self, system: str) -> str:
        """Return the stratum of one system of this census."""
        return separation_stratum(self.separations[system], tau_angstrom=self.tau_angstrom)

    def members(self, stratum: str) -> tuple[str, ...]:
        """Return, sorted, the systems in one stratum."""
        if stratum not in SEPARATION_STRATA:
            raise ValueError(f"stratum must be one of {SEPARATION_STRATA}; got {stratum!r}")
        return tuple(sorted(name for name in self.separations if self.stratum(name) == stratum))

    @property
    def counts(self) -> dict[str, int]:
        """Systems per stratum, every stratum present even at zero."""
        counts = dict.fromkeys(SEPARATION_STRATA, 0)
        for name in self.separations:
            counts[self.stratum(name)] += 1
        return counts

    @property
    def admitted(self) -> tuple[str, ...]:
        """The systems the rule admits: the clean and the ambiguous, together and sorted.

        The ambiguous band is IN. It is separated by :meth:`members` for reporting, and a caller
        that wants only the clean systems is applying the ``2*tau`` cut and must say so by name.
        """
        return tuple(
            sorted(set(self.members(STRATUM_CLEAN)) | set(self.members(STRATUM_AMBIGUOUS)))
        )

    @property
    def refused(self) -> tuple[str, ...]:
        """The systems the rule refuses: the degenerate ones, and the unmeasurable ones.

        An unmeasurable ``s`` is refused rather than admitted on the benefit of the doubt. The
        objective divides by ``s``; a system without one has no place in a pool defined by it.
        """
        return tuple(
            sorted(set(self.members(STRATUM_DEGENERATE)) | set(self.members(STRATUM_UNKNOWN)))
        )

    def quantile(self, fraction: float) -> float:
        """Return a quantile of the MEASURED separations, unmeasurable systems excluded.

        Excluded rather than imputed, and the count of measured systems is on the census beside it,
        so a reader can see what the quantile is a quantile of.
        """
        values = sorted(value for value in self.separations.values() if value is not None)
        if not values:
            return float("nan")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        position = fraction * (len(values) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return values[low]
        return values[low] + (values[high] - values[low]) * (position - low)

    def at_or_below(self, threshold: float) -> int:
        """How many MEASURED separations sit at or below ``threshold``."""
        return sum(
            1 for value in self.separations.values() if value is not None and value <= threshold
        )

    def as_dict(self) -> dict[str, Any]:
        """The block an artifact carries. Counts, the realised cut, and the membership."""
        measured = [value for value in self.separations.values() if value is not None]
        return {
            "tau_angstrom": self.tau_angstrom,
            "realised_cut_angstrom": self.realised_cut_angstrom,
            "ambiguous_band_upper_angstrom": self.ambiguous_band_upper_angstrom,
            "rule": "admitted iff s > tau, where s is the aligned N/CA/C/O apo-holo deviation",
            "systems_considered": len(self.separations),
            "systems_measured": len(measured),
            "separation_angstrom_median": self.quantile(0.5) if measured else None,
            "separation_angstrom_min": min(measured) if measured else None,
            "separation_angstrom_max": max(measured) if measured else None,
            "counts": self.counts,
            "n_admitted": len(self.admitted),
            "n_refused": len(self.refused),
            "strata": {stratum: list(self.members(stratum)) for stratum in SEPARATION_STRATA},
        }


def separation_census(
    separations: Mapping[str, float | None], *, tau_angstrom: float
) -> SeparationCensus:
    """Sort a population of measured separations into the strata at one ``tau``."""
    tau = checked_tau(tau_angstrom)
    table: dict[str, float | None] = {}
    for name, value in separations.items():
        key = str(name)
        if key in table:
            raise ValueError(f"{key!r} appears twice in the separation table")
        table[key] = None if value is None else _checked_separation(value, name=key)
    return SeparationCensus(
        tau_angstrom=tau,
        realised_cut_angstrom=tau,
        ambiguous_band_upper_angstrom=2.0 * tau,
        separations=table,
    )


def baseline_stratum(witness_angstrom: object | None, *, tau_angstrom: float) -> str:
    """Return which of :data:`BASELINE_STRATA` a system's baseline witness puts it in."""
    tau = checked_tau(tau_angstrom)
    if witness_angstrom is None:
        return STRATUM_BASELINE_UNMEASURED
    value = _checked_separation(witness_angstrom, name="system")
    return STRATUM_BASELINE_DEGENERATE if value <= tau else STRATUM_BASELINE_PASSED


@dataclass(frozen=True)
class BaselineWitnesses:
    """Per-sequence adapter-off witnesses, with the coverage that produced them.

    ``witness`` maps a sequence to the SMALLEST adapter-off deviation seen for it -- its best
    unsteered draw, which is the quantity ``success@B`` reads. ``draws`` records how many ambient
    rows each sequence contributed, because "one draw" and "forty draws" support very different
    statements about a system that was NOT witnessed inside ``tau``.
    """

    witness: dict[str, float]
    draws: dict[str, int]
    level: float
    residual_fraction_ceiling: float
    #: Total training rows read, ambient rows kept, and sequences named by the whole trace. The
    #: last is the denominator coverage is quoted against when the trace is the only source.
    rows_read: int
    rows_ambient: int
    sequences_in_trace: int
    #: The largest ``residual_fraction`` any kept row carried, so a reader can see how far the
    #: retained draws sit below the ceiling rather than taking the ceiling's word for it.
    max_kept_residual_fraction: float

    def as_dict(self) -> dict[str, Any]:
        counts = sorted(self.draws.values())
        return {
            "level": self.level,
            "reference_state": "holo" if self.level == HOLO_LEVEL else "apo",
            "residual_fraction_ceiling": self.residual_fraction_ceiling,
            "max_kept_residual_fraction": self.max_kept_residual_fraction,
            "rows_read": self.rows_read,
            "rows_ambient": self.rows_ambient,
            "sequences_in_trace": self.sequences_in_trace,
            "sequences_witnessed": len(self.witness),
            "draws_per_sequence_min": counts[0] if counts else 0,
            "draws_per_sequence_max": counts[-1] if counts else 0,
            "witness_is": (
                "the MINIMUM adapter-off deviation per sequence: success@B is a maximum over "
                "draws, so the best unsteered draw is what makes a system a free pass"
            ),
        }


def baseline_witnesses(
    training_records: Sequence[Mapping[str, Any]],
    *,
    level: float = HOLO_LEVEL,
    residual_fraction_ceiling: float = AMBIENT_RESIDUAL_FRACTION_CEILING,
    deviation_key: str = "metric_deviation_angstrom",
) -> BaselineWitnesses:
    """Extract per-sequence adapter-off witnesses from a checkpoint's training records.

    A row qualifies when its ``unit.level`` names ``level``, its ``residual_fraction`` is present
    and at or below ``residual_fraction_ceiling``, and its ``deviation_key`` is a finite number. A
    row whose ``residual_fraction`` is ABSENT is skipped rather than treated as zero: a missing
    dose is not an adapter-off draw, and reading it as one would manufacture witnesses.

    Nothing here interprets a checkpoint file. The caller loads the trace and hands in the list, so
    this stays testable without torch and without a five-megabyte artifact.
    """
    ceiling = float(residual_fraction_ceiling)
    if not math.isfinite(ceiling) or ceiling < 0.0:
        raise ValueError(
            f"residual_fraction_ceiling must be finite and non-negative; got {ceiling!r}"
        )
    witness: dict[str, float] = {}
    draws: dict[str, int] = {}
    sequences: set[str] = set()
    ambient = 0
    max_kept = 0.0
    for row in training_records:
        unit = row.get("unit")
        if not isinstance(unit, Mapping):
            raise ValueError(
                "every training record must carry a `unit` mapping naming its sequence"
            )
        sequence_id = str(unit["sequence_id"])
        sequences.add(sequence_id)
        if float(unit["level"]) != float(level):
            continue
        fraction = row.get("residual_fraction")
        if fraction is None or not isinstance(fraction, Real) or isinstance(fraction, bool):
            continue
        if not math.isfinite(float(fraction)) or float(fraction) > ceiling:
            continue
        deviation = row.get(deviation_key)
        if deviation is None or isinstance(deviation, bool) or not isinstance(deviation, Real):
            continue
        value = float(deviation)
        if not math.isfinite(value) or value < 0.0:
            continue
        ambient += 1
        max_kept = max(max_kept, float(fraction))
        draws[sequence_id] = draws.get(sequence_id, 0) + 1
        previous = witness.get(sequence_id)
        if previous is None or value < previous:
            witness[sequence_id] = value
    return BaselineWitnesses(
        witness=witness,
        draws=draws,
        level=float(level),
        residual_fraction_ceiling=ceiling,
        rows_read=len(training_records),
        rows_ambient=ambient,
        sequences_in_trace=len(sequences),
        max_kept_residual_fraction=max_kept,
    )


@dataclass(frozen=True)
class BaselineCensus:
    """A population sorted by gate 2 at one ``tau``. Unwitnessed systems are their own stratum."""

    tau_angstrom: float
    realised_cut_angstrom: float
    #: system -> its best adapter-off deviation, or ``None`` when no adapter-off draw exists.
    witnesses: dict[str, float | None]
    #: How many ambient draws each witnessed system contributed.
    draws: dict[str, int]

    def stratum(self, system: str) -> str:
        return baseline_stratum(self.witnesses[system], tau_angstrom=self.tau_angstrom)

    def members(self, stratum: str) -> tuple[str, ...]:
        if stratum not in BASELINE_STRATA:
            raise ValueError(f"stratum must be one of {BASELINE_STRATA}; got {stratum!r}")
        return tuple(sorted(name for name in self.witnesses if self.stratum(name) == stratum))

    @property
    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(BASELINE_STRATA, 0)
        for name in self.witnesses:
            counts[self.stratum(name)] += 1
        return counts

    @property
    def refused(self) -> tuple[str, ...]:
        """The systems gate 2 refuses. ONLY the witnessed-degenerate ones.

        An unwitnessed system is NOT refused: gate 2 never ran on it. That is the asymmetry the
        module docstring is about, and it is why this property is not the complement of
        :attr:`admitted`'s underlying rule but the degenerate stratum exactly.
        """
        return self.members(STRATUM_BASELINE_DEGENERATE)

    @property
    def coverage(self) -> str:
        """ "N of M systems carry an adapter-off draw", for every output."""
        witnessed = sum(1 for value in self.witnesses.values() if value is not None)
        return f"{witnessed} of {len(self.witnesses)} systems carry an adapter-off draw"

    def as_dict(self) -> dict[str, Any]:
        measured = [value for value in self.witnesses.values() if value is not None]
        return {
            "tau_angstrom": self.tau_angstrom,
            "realised_cut_angstrom": self.realised_cut_angstrom,
            "rule": (
                "refused iff a WITNESSED adapter-off draw lands at or within tau of holo; a system "
                "with no adapter-off draw is not refused and not certified"
            ),
            "systems_considered": len(self.witnesses),
            "systems_witnessed": len(measured),
            "coverage": self.coverage,
            "coverage_fraction": (
                len(measured) / len(self.witnesses) if self.witnesses else float("nan")
            ),
            "witness_angstrom_median": (sorted(measured)[len(measured) // 2] if measured else None),
            "counts": self.counts,
            "strata": {stratum: list(self.members(stratum)) for stratum in BASELINE_STRATA},
        }


def baseline_census(
    systems: Sequence[str],
    witnesses: Mapping[str, float],
    *,
    tau_angstrom: float,
    draws: Mapping[str, int] | None = None,
) -> BaselineCensus:
    """Sort ``systems`` by gate 2, leaving those with no witness in their own stratum.

    ``systems`` is the population, ``witnesses`` whatever subset of it has an adapter-off draw. A
    witness naming a system outside the population raises rather than being ignored: a trace built
    for a different pool cannot gate this one, and an intersection taken quietly hides that.
    """
    tau = checked_tau(tau_angstrom)
    population = [str(name) for name in systems]
    if len(population) != len(set(population)):
        raise ValueError("the population names a system twice")
    unknown = sorted(set(map(str, witnesses)) - set(population))
    if unknown:
        raise ValueError(
            f"{len(unknown)} witnessed systems are not in the population: {unknown[:10]}. A trace "
            "from a different pool cannot gate this one."
        )
    table: dict[str, float | None] = {}
    for name in population:
        value = witnesses.get(name)
        table[name] = None if value is None else _checked_separation(value, name=name)
    return BaselineCensus(
        tau_angstrom=tau,
        realised_cut_angstrom=tau,
        witnesses=table,
        draws={str(k): int(v) for k, v in (draws or {}).items() if str(k) in table},
    )


@dataclass(frozen=True)
class AdmissionCensus:
    """Both gates over one population at one ``tau``, with the joint table.

    Admission is the CONJUNCTION: a system enters the pool only if it clears gate 1 and is not
    refused by gate 2. The two censuses stay separate on the artifact so a reader can see which
    gate removed what, which is the whole reason the correction that added gate 2 was needed --
    domain motion had gate 1 and read it as though it were both.
    """

    separation: SeparationCensus
    baseline: BaselineCensus

    def __post_init__(self) -> None:
        if set(self.separation.separations) != set(self.baseline.witnesses):
            raise ValueError(
                "the two gates describe different populations; a joint table over them would be "
                "a join with silent drops on both sides"
            )
        if self.separation.tau_angstrom != self.baseline.tau_angstrom:
            raise ValueError(
                f"gate 1 was cut at tau = {self.separation.tau_angstrom} and gate 2 at "
                f"{self.baseline.tau_angstrom}. Both gates are defined at the SAME evaluation "
                "threshold; two taus would make the joint table meaningless."
            )

    @property
    def tau_angstrom(self) -> float:
        return self.separation.tau_angstrom

    @property
    def systems(self) -> tuple[str, ...]:
        return tuple(sorted(self.separation.separations))

    def passes_gate_one(self, system: str) -> bool:
        return system in set(self.separation.admitted)

    def passes_gate_two(self, system: str) -> bool:
        """Not refused by gate 2. An UNWITNESSED system passes: gate 2 never ran on it."""
        return self.baseline.stratum(system) != STRATUM_BASELINE_DEGENERATE

    @property
    def admitted(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.systems
            if self.passes_gate_one(name) and self.passes_gate_two(name)
        )

    @property
    def joint_counts(self) -> dict[str, int]:
        """The 2x2: which gate refused what, and how much the two overlap."""
        table = {
            "fails_gate_one_only": 0,
            "fails_gate_two_only": 0,
            "fails_both": 0,
            "passes_both": 0,
        }
        for name in self.systems:
            one = self.passes_gate_one(name)
            two = self.passes_gate_two(name)
            if one and two:
                table["passes_both"] += 1
            elif one:
                table["fails_gate_two_only"] += 1
            elif two:
                table["fails_gate_one_only"] += 1
            else:
                table["fails_both"] += 1
        return table

    @property
    def joint_counts_where_both_measured(self) -> dict[str, int]:
        """The same 2x2, restricted to systems where BOTH gates were actually evaluated.

        The unrestricted table above counts an unwitnessed system as passing gate 2, which is the
        right ADMISSION rule and the wrong basis for a statement about how much the two gates
        overlap. Quoting overlap from the unrestricted table is how a gate applied to 19% of a pool
        gets described as one applied to all of it.
        """
        table = {
            "fails_gate_one_only": 0,
            "fails_gate_two_only": 0,
            "fails_both": 0,
            "passes_both": 0,
            "systems": 0,
        }
        for name in self.systems:
            if self.baseline.witnesses[name] is None:
                continue
            if self.separation.separations[name] is None:
                continue
            table["systems"] += 1
            one = self.passes_gate_one(name)
            two = self.passes_gate_two(name)
            if one and two:
                table["passes_both"] += 1
            elif one:
                table["fails_gate_two_only"] += 1
            elif two:
                table["fails_gate_one_only"] += 1
            else:
                table["fails_both"] += 1
        return table

    def strata_for(self, system: str) -> dict[str, str]:
        """Both strata for one system, named by gate."""
        return {
            "separation_stratum": self.separation.stratum(system),
            "baseline_stratum": self.baseline.stratum(system),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau_angstrom": self.tau_angstrom,
            "rule": (
                "admitted iff gate 1 (s > tau, deposited conformers distinguishable) AND gate 2 "
                "(no witnessed adapter-off draw within tau of holo). Neither gate implies the "
                "other."
            ),
            "systems_considered": len(self.systems),
            "n_admitted": len(self.admitted),
            "gate_one": self.separation.as_dict(),
            "gate_two": self.baseline.as_dict(),
            "joint_counts": self.joint_counts,
            "joint_counts_where_both_measured": self.joint_counts_where_both_measured,
            "admitted": list(self.admitted),
        }


def admission_census(
    separations: Mapping[str, float | None],
    witnesses: Mapping[str, float],
    *,
    tau_angstrom: float,
    draws: Mapping[str, int] | None = None,
) -> AdmissionCensus:
    """Apply both gates to one population at one ``tau``."""
    gate_one = separation_census(separations, tau_angstrom=tau_angstrom)
    gate_two = baseline_census(
        sorted(gate_one.separations), witnesses, tau_angstrom=tau_angstrom, draws=draws
    )
    return AdmissionCensus(separation=gate_one, baseline=gate_two)


def pool_separations(
    reference: CrypticMetricReference, sequence_ids: Sequence[str] | None = None
) -> dict[str, float]:
    """Recompute ``s`` for every sequence of a metric reference, through the shared RMSD kernel.

    ``s`` is the aligned :data:`~sae_bioemu.cryptic_steering.direct_metric.CRYPTIC_METRIC_ATOMS`
    deviation between ``apo_backbone`` and ``holo_backbone`` over the entry's own
    ``metric_indices``, taken through
    :func:`~sae_bioemu.cryptic_steering.direct_metric.aligned_metric_deviation` -- which is
    :func:`~sae_bioemu.steering.aligned_rmsd.aligned_rmsd` on the flattened atom rows. **There is
    no second RMSD in this module.** A separately-implemented superposition here would be a second
    definition of the quantity the objective divides by and the scorer thresholds, and the two
    would drift.

    Recomputed rather than read off ``entry.separation_angstrom``, even though the two are checked
    equal on every read that presents the artifact: the admission rule is the reason a protein is
    in or out of a pool, and it should rest on a measurement this call performed.
    """
    import torch

    from sae_bioemu.cryptic_steering.direct_metric import aligned_metric_deviation

    names = (
        tuple(reference.sequence_ids)
        if sequence_ids is None
        else tuple(str(name) for name in sequence_ids)
    )
    measured: dict[str, float] = {}
    for name in names:
        entry = reference.entries.get(name)
        if entry is None:
            raise KeyError(
                f"the metric reference does not describe {name!r}, so it has no apo-holo "
                "separation and cannot be admitted or refused by a rule defined on one"
            )
        measured[name] = float(
            aligned_metric_deviation(
                entry.apo_backbone.to(torch.float64),
                entry.holo_backbone.to(torch.float64),
                torch.arange(len(entry.metric_indices)),
            )
        )
    return measured
