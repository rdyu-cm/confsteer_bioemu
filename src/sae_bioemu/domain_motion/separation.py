"""The pool gate: keep the systems whose two conformers are further apart than tau.

SCOPE: DOMAIN MOTIONS. This is a filter on the TRAINING pool and it is one line of arithmetic; the
reasons it is that line and not another are the content of this module.

WHY GATE AT ALL
---------------

The evaluation set occupies one regime and the DynDom training pool occupies another. All 22
``multiconf_domainmotion`` benchmark systems are separated by more than tau, at a median true
N/CA/C/O separation of 6.84 Angstrom. The DynDom pool's median true separation is 1.82 Angstrom,
and 1058 of its 1476 systems have their two conformers CLOSER to each other than tau -- which means
a draw sitting on conformer i "succeeds" against conformer j and the metric cannot resolve the
motion at all. Training on those systems teaches the adapter to satisfy a threshold that was never
in question.

It also measurably improves the only predictive number the pool survey produced. Measured, under
the TRUE metric (``docs/domain_motion/dm1_true_metric_separation.json``, joined to the DM-1
survey's per-system rank correlations):

    ALL                      n=1476 | sep med 1.82 | rho med -0.243 | wrong sign 32.1%
    true separation > 3.0    n= 418 | sep med 4.98 | rho med -0.422 | wrong sign 21.8%

THE PROXY IS NOT THE METRIC, SO THE THRESHOLD IS NOT APPLIED TO IT
------------------------------------------------------------------

``docs/domain_motion/dm1_coordinate_predicts_metric.json`` measured separations with a
virtual-C-beta Kabsch PROXY and gated at 3.0 Angstrom on that, keeping 452. The benchmark's
separations are a true N/CA/C/O RMSD, and applying one threshold across both is not
apples-to-apples: the two quantities are measured over different atoms and there is no reason their
ratio is one. So the gate is applied to :func:`reference_separation`, the same aligned N/CA/C/O
RMSD the metric and the objective use, recomputed from the deposited coordinates.

The effect is a strictly TIGHTER gate and not a reshuffle, which is worth stating because it was
not the only possible outcome: the true metric keeps 418 against the proxy's 452, every one of the
418 is also in the proxy's 452 (34 proxy-only, 0 true-only), and the two separations rank at
Spearman 0.997 over all 1476 systems. The proxy was a good ORDERING and a slightly loose
CALIBRATION -- its median separation is 1.99 against the true 1.82 -- which is exactly the kind of
difference a threshold, and not a rank statistic, is sensitive to.

GATE ON SEPARATION ONLY
------------------------

An audit of this design suggested also dropping the systems BioEmu already succeeds on natively --
1049 of 1476 by best-of-32, of which 191 survive the true-metric separation gate, so the second
filter would take the pool from 418 to 227. **That is refused, and the disagreement is recorded
here rather than settled silently.** Native success is measured BEST-OF-32:
a system booked as a native hit still has most of its 32 individual draws outside tau, and those
draws are exactly the observations a per-draw objective learns from. Discarding the system discards
them. The two properties are also of different kinds: SEPARATION is a property of the SYSTEM and
belongs in a pool filter, while SUCCESS is a property of the DRAW, and the hinge already handles it
per draw by going exactly flat for any draw already inside ``tau_p``. A native-hit system whose
median draw is at 5 Angstrom contributes gradient from its bad draws and silence from its good ones,
which is the correct treatment and is not available to a filter that removed it.

SEQUENCE GROUPING STILL APPLIES ON TOP
---------------------------------------

The gate counts SYSTEMS; the independent statistical unit is the merged sequence component
(``cryptic_steering/split.py``). Measured on the gated pool: the 418 kept systems carry
417 with a merged component (``CYANO1R1`` has none and is reported rather than counted as its own
group) falling into **247 independent groups**, 186 of them singletons, with the largest holding 66
systems -- 15.8% of the pool in one group. So a system count quoted as a sample size would
overstate the design by about 1.7x, and the split built on top of this must assign whole
components, never systems. :func:`independent_group_counts` is what turns a kept set into that
number and refuses a system whose component is unknown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from sae_bioemu.domain_motion.objective import (
    DIRECT_METRIC_TAU_ANGSTROM,
    aligned_backbone_rmsd,
)

__all__ = [
    "SEPARATION_GATE_ANGSTROM",
    "GateOutcome",
    "apply_separation_gate",
    "independent_group_counts",
    "reference_separation",
    "shared_metric_residues",
]

#: The pool gate, in Angstrom, and deliberately the benchmark's own tau rather than a second
#: number. A system separated by less than tau is one the metric cannot resolve; a system separated
#: by exactly tau is one where reaching either conformer scores against both. The comparison is
#: STRICT (``> tau``) for that reason.
SEPARATION_GATE_ANGSTROM = DIRECT_METRIC_TAU_ANGSTROM


def shared_metric_residues(
    first_resolved: Sequence[bool], second_resolved: Sequence[bool]
) -> tuple[int, ...]:
    """Return the residue indices resolved in BOTH deposited conformers, ascending.

    A residue is in the metric set only when both conformers place it on this sequence index AND
    both supply all four backbone atoms for it. The caller decides what "resolved" means and passes
    the two masks; this exists so the intersection is taken in one place and the metric, the
    separation and both levels are read over one atom set.
    """
    if len(first_resolved) != len(second_resolved):
        raise ValueError("the two resolution masks must cover the same residue frame")
    return tuple(
        index
        for index, (left, right) in enumerate(zip(first_resolved, second_resolved, strict=True))
        if bool(left) and bool(right)
    )


def reference_separation(
    first: Tensor, second: Tensor, metric_residues: Sequence[int] | Tensor
) -> float:
    """Return ``s_p``: the superposed N/CA/C/O RMSD between the two deposited conformers.

    The SAME function the objective measures a draw with
    (:func:`~sae_bioemu.domain_motion.objective.aligned_backbone_rmsd`), applied to the two
    crystals. That identity is the point: ``s_p`` normalises a quantity, and a normaliser computed
    by a second implementation is a per-protein scale factor of unknown size.

    Symmetric up to the arithmetic floor -- superposing i onto j and j onto i give the same optimal
    rotation up to inversion and the same per-atom distances -- so which conformer is passed first
    does not matter and the caller is not asked to decide.
    """
    index = (
        metric_residues
        if isinstance(metric_residues, Tensor)
        else torch.as_tensor(list(metric_residues), dtype=torch.long)
    )
    with torch.no_grad():
        return float(aligned_backbone_rmsd(first, second, index).reshape(()))


@dataclass(frozen=True)
class GateOutcome:
    """Which systems the separation gate kept, which it dropped, and the counts behind both."""

    threshold_angstrom: float
    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    #: Every system's measured separation, kept and dropped alike. Reported in full because a gate
    #: that cannot say what it removed is the funnel defect this programme keeps rediscovering.
    separation_angstrom: Mapping[str, float]

    @property
    def considered(self) -> int:
        return len(self.kept) + len(self.dropped)

    def summary(self) -> dict[str, object]:
        kept = sorted(self.separation_angstrom[name] for name in self.kept)
        return {
            "threshold_angstrom": self.threshold_angstrom,
            "considered": self.considered,
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "kept_separation_median": (kept[len(kept) // 2] if kept else float("nan")),
        }


def apply_separation_gate(
    separations: Mapping[str, float], *, threshold: float = SEPARATION_GATE_ANGSTROM
) -> GateOutcome:
    """Keep the systems whose two conformers are STRICTLY further apart than ``threshold``.

    ``separations`` must already be true N/CA/C/O separations from :func:`reference_separation`.
    A non-finite entry is a system whose separation could not be measured, and it is DROPPED with
    the rest rather than silently kept: an unmeasurable normaliser is not a usable ``s_p``.
    """
    if not isinstance(threshold, float) or not threshold > 0.0:
        raise ValueError("threshold must be a positive float in Angstrom")
    if not separations:
        raise ValueError("the separation gate needs at least one measured system")
    kept: list[str] = []
    dropped: list[str] = []
    for name in sorted(separations):
        value = float(separations[name])
        (kept if value == value and value > threshold else dropped).append(name)
    return GateOutcome(
        threshold_angstrom=float(threshold),
        kept=tuple(kept),
        dropped=tuple(dropped),
        separation_angstrom={name: float(value) for name, value in separations.items()},
    )


def independent_group_counts(
    systems: Sequence[str], component_of: Mapping[str, str]
) -> dict[str, object]:
    """Return how many INDEPENDENT sequence groups a set of systems really carries.

    ``component_of`` maps a system to its merged sequence component -- the unit
    :func:`sae_bioemu.cryptic_steering.split.merge_clusters_by_identity` produces and the unit a
    train/validation/test split must assign whole. A system with no component is reported by name
    and never quietly counted as its own group: an unclustered system is an unknown relationship,
    not an independent one, and treating it as independent is how a training protein reaches a
    held-out metric.
    """
    unmapped = sorted(name for name in systems if name not in component_of)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} systems have no merged sequence component, so their independence "
            f"from the rest of the pool is unknown; first: {unmapped[:5]}. Counting them as their "
            "own groups would be the assumption that produces leakage"
        )
    components = [component_of[name] for name in systems]
    sizes: dict[str, int] = {}
    for component in components:
        sizes[component] = sizes.get(component, 0) + 1
    ordered = sorted(sizes.values(), reverse=True)
    return {
        "systems": len(systems),
        "groups": len(sizes),
        "largest_group_systems": ordered[0] if ordered else 0,
        "largest_group_share": (ordered[0] / len(systems)) if systems else float("nan"),
        "singleton_groups": sum(1 for size in ordered if size == 1),
    }
