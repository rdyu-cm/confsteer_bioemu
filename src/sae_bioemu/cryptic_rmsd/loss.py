"""The pocket-RMSD steering objective: the shared loss, held to the direct metric's contract.

SCOPE: CRYPTIC POCKETS. Training-only, for the reason the cryptic, domain-motion and kinase-RMSD
losses are: no evaluation computes a loss, so a change here cannot change a generated structure or
a scored number and must not be able to invalidate an evaluation.

**This subclass adds no term to** ``total``, **and that is the design rather than an omission.**
The direct metric IS the property term::

    prediction = m + s_p           the measured N/CA/C/O deviation, on this protein's own origin
    target     = s_p               that origin
    scale      = s_p               that protein's deposited apo<->holo separation
    property_loss = MSE

so the shared ``SteeringLoss`` computes ``mean_n ((prediction - target)/scale)^2`` =
``mean_n (m_n/s_p)^2``, which is ``2 * mean_n metric_mse(m_n, s_p)``; the factor of two lives in
:attr:`PocketRmsdSteeringLoss.CONVENTION_FACTOR` rather than in the weight, so ``lambda_m``
multiplies ``objective.metric_mse`` here exactly as it does on the cryptic-pocket, domain-motion
and kinase-RMSD paths (D-87: one name, one quantity).

No benchmark ``tau`` reaches any part of this -- not its value, not its gradient, and not the
digest of the per-system target table the checkpoint is bound to. See
:mod:`sae_bioemu.cryptic_rmsd.objective` design point 1.

HOW THIS DIFFERS FROM ``CrypticSteeringLoss``, TERM BY TERM
-------------------------------------------------------------

``CrypticSteeringLoss`` prices ``lambda_y * target(the progress coordinate) + lambda_perp * zeta_pocket^2 +
lambda_rigid * rigidity + lambda_m * metric + ...``. Here:

* ``lambda_y`` prices a PROGRESS-COORDINATE target and there is none, so it is refused unless zero.
  It is SPELLED at zero in the config rather than omitted, because it is in
  ``bioemu/steering_workflow.py``'s ``_SHARED_RUNTIME_KEYS`` and read unconditionally -- a config
  that leaves it out fails the runtime schema rather than describing an objective without it. This
  is the DOMAIN-MOTION idiom and deliberately not the cryptic one: on a cryptic arm ``lambda_y = 0``
  REMOVES the ``the progress coordinate`` proxy from an objective that still has a ``the progress coordinate`` channel, so the cryptic loss
  builds the term and journals it at zero weight. Here the channel is already filled by the graded
  Angstrom deviation, so a nonzero ``lambda_y`` would be one name meaning two opposite things,
  decided by which config a reader had open.
* ``lambda_perp``, ``lambda_perp_fraction`` and ``lambda_rigid`` are refused by the DESCRIPTOR, by
  name and with reasons, before a GPU is allocated -- see
  :data:`sae_bioemu.cryptic_rmsd.property._REFUSED`.
* ``lambda_c`` is refused HERE, because the shared constructor would accept it and drop it
  silently: the two levels measure deviations to DIFFERENT deposited conformers, so their
  difference is not a dose gap.

WHAT THIS CLASS DOES DO
------------------------

1. **It enforces the origin contract, elementwise, on every batch.** ``prediction - target`` must
   BE the measured ``m`` on the wire. A shifted origin that was not shifted, or was shifted by
   another protein's number, is a finite and plausible loss with no other symptom.
2. **It refuses ``lambda_g = 0``.** Under a direct metric the residues OUTSIDE the benchmark's
   metric region receive EXACTLY ZERO gradient from the property term -- measured on ``O74933``,
   whose shipped ranges are ``[[105, 120], [280, 330]]`` and which resolves 67 of those positions
   on a 486-residue BioEmu draw: the maximum absolute gradient outside the region is exactly 0.0
   while inside it is 7.65e-3, so 419 of 486 residues are entirely unconstrained by this term.
   "Satisfy the metric by deforming the rest of the protein" is priced by nothing else in this
   loss, because ``lambda_l`` is zero on the shipped arms and D-79 section 7.3 measured
   ``lambda_g`` at 0.04% of the loss budget on an objective that touches every residue through
   ``the progress coordinate``. It stops being decorative the moment it is the only structural statement there is.
3. **It journals the deviation in Angstrom, under its own column name.**
   ``pocket_conformer_rmsd_angstrom`` is DISTINCT from the cryptic ``metric_deviation_angstrom``,
   the domain-motion ``direct_metric_angstrom`` and the kinase ``kinase_conformer_rmsd_angstrom``,
   because ``property_registry.journal_column_weights`` refuses a column declared against two
   weights and because the four are not the same quantity. This one is the deviation
   ``success@B`` grades, measured over the benchmark's own metric region, on a checkpoint whose
   targets came from that same region's ``s_p``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import torch

from sae_bioemu.cryptic_rmsd.objective import METRIC_OBJECTIVE_MSE, validated_metric_objective
from sae_bioemu.steering.training import (
    SIGMA_LOSS_NORMALIZATION,
    PropertyLossKind,
    ResidualPenaltyKind,
    SteeringLoss,
    SteeringLossComponents,
    _validate_loss_tensor,
)

__all__ = [
    "POCKET_RMSD_JOURNAL_COLUMN",
    "POCKET_RMSD_LOSS_NORMALIZATION",
    "PocketRmsdSteeringLoss",
    "PocketRmsdSteeringLossComponents",
]

#: The declared normalisation this objective runs under: a required, strictly positive,
#: per-observation divisor applied to the absolute channel and to no other. It is what
#: ``resolve_property_loss_normalization(relative_sequence, system_constant)`` returns, and
#: ``ordinary_workflow._require_matched_loss_normalization`` refuses a run where the two disagree.
#:
#: **The divisor is NOT a conformational spread, and the name is inherited rather than
#: descriptive.** It is ``s_p``, this protein's own deposited apo<->holo separation in Angstrom,
#: which is what ``_unit_sigma`` returns on this origin. That is recorded per system in the arm
#: identity, so a report reading ``sequence_sigma`` on a pocket-RMSD run can find out what the
#: number actually was.
POCKET_RMSD_LOSS_NORMALIZATION = SIGMA_LOSS_NORMALIZATION

#: The persisted training column carrying the UNCLAMPED deviation in Angstrom. Its own name, gated
#: on ``lambda_m``: see the module docstring's point 3.
POCKET_RMSD_JOURNAL_COLUMN = "pocket_conformer_rmsd_angstrom"


def _validated_lambda_m(value: object) -> float:
    """Return ``lambda_m`` as a finite nonnegative float, or fail closed.

    Extracted so the weight can be validated BEFORE ``super().__init__()``, which is where the
    shared driver check runs. A BYTE-FOR-BYTE TWIN of the cryptic, domain-motion and kinase-RMSD
    helpers rather than an import, for the reason :mod:`sae_bioemu.cryptic_rmsd.frames` is a copy:
    those files are in other properties' training groups, so an import would bind every
    ``pocket_conformer_rmsd`` checkpoint to code a run of this property does not execute.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("lambda_m must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("lambda_m must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class PocketRmsdSteeringLossComponents(SteeringLossComponents):
    """The shared components plus the measured deviation, in Angstrom."""

    #: Mean over observations of the superposed N/CA/C/O RMS deviation to the deposited conformer
    #: the level names, over that protein's benchmark metric region, in ANGSTROM and UNWEIGHTED.
    #: Not a term of :attr:`total` -- the priced form of it is -- and journalled for that reason:
    #: the priced value is a squared ratio and is not in Angstrom at all, so the column an arm's
    #: progress is read from has to be this one.
    pocket_conformer_rmsd_angstrom: torch.Tensor | None = None

    def journal_columns(self) -> dict[str, float | None]:
        return {
            POCKET_RMSD_JOURNAL_COLUMN: (
                None
                if self.pocket_conformer_rmsd_angstrom is None
                else float(self.pocket_conformer_rmsd_angstrom.detach().reshape(()))
            )
        }


class PocketRmsdSteeringLoss(SteeringLoss):
    """The shared steering objective, contracted to the direct N/CA/C/O cryptic-pocket metric."""

    #: One per-observation input the shared runner reads off its generated sample: the measured
    #: deviation in Angstrom, for the journal and for the origin cross-check.
    #:
    #: ``metric_deviation_angstrom`` is the TRANSPORT name -- the field on the shared
    #: ``GeneratedTrainingSample`` -- which the cryptic, domain-motion and kinase-RMSD direct-metric
    #: terms fill too. The wire is shared because the four are the same SHAPE of input; the
    #: QUANTITIES are not shared and keep separate journal columns.
    EXTRA_SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ("metric_deviation_angstrom",)

    #: None. Every extended weight belongs to another objective's term, and a nonzero value for any
    #: of them is refused by the DESCRIPTOR with a reason before a GPU is allocated, and by the
    #: shared constructor if it gets that far. ``lambda_m`` is NOT here because it is not in the
    #: shared ``EXTENDED_LOSS_TERMS`` vocabulary at all -- that tuple lives in
    #: ``steering/training.py``, a member of the shared ``training_only`` group -- so it is
    #: validated by this class directly.
    PRICED_EXTENDED_TERMS: ClassVar[frozenset[str]] = frozenset()

    #: ``lambda_m`` is THE driver, and ``lambda_y`` is refused outright below. ``lambda_y`` is
    #: declared here anyway because ``_require_a_driving_weight`` reads this tuple and the shared
    #: weight must remain expressible as a driver on the shared surface; what makes it unusable on
    #: this property is the explicit refusal, which names the reason. A loss with NEITHER is still
    #: refused by name -- ``lambda_r``, ``lambda_g`` and ``lambda_l`` are penalties, and an
    #: objective made of them alone has its global minimum at the adapter doing nothing.
    DRIVING_WEIGHTS: ClassVar[tuple[str, ...]] = ("lambda_y", "lambda_m")

    #: The CONVENTION FACTOR between the shared property channel and this property's own ``L``.
    #: The shared MSE channel computes ``normalized_error.square().mean()`` = ``mean(z^2)`` =
    #: ``2 * mean(L)``, so the half lives in the arithmetic rather than in the weight. That is what
    #: makes ``lambda_m`` mean ``L``'s multiplier here, on the cryptic-pocket term, on domain motion
    #: and on kinase RMSD alike, which is the whole content of D-87: one name, one quantity.
    CONVENTION_FACTOR: ClassVar[dict[str, float]] = {METRIC_OBJECTIVE_MSE: 0.5}

    #: Which shared property loss this form REQUIRES. Not a default and not a preference: the entry
    #: is the arithmetic the objective is defined as, so a configuration naming another loss is
    #: refused rather than silently trained.
    REQUIRED_PROPERTY_LOSS: ClassVar[dict[str, PropertyLossKind]] = {
        METRIC_OBJECTIVE_MSE: PropertyLossKind.MSE
    }

    #: How far ``prediction - target`` may sit from the measured ``m``, in Angstrom. Loose enough
    #: for float32 addition and subtraction of two numbers of order 10 Angstrom; far tighter than
    #: any real disagreement, because a missing origin shift moves it by a whole ``s_p`` and
    #: another protein's origin by the difference of two separations.
    ORIGIN_TOLERANCE_ANGSTROM: ClassVar[float] = 1e-3

    @property
    def property_term_weight(self) -> float:
        """``lambda_m`` converted into the shared channel's own convention."""
        return self.CONVENTION_FACTOR[self.metric_objective] * self.lambda_m

    def __init__(
        self,
        *,
        lambda_y: float,
        lambda_r: float,
        lambda_g: float,
        lambda_c: float = 0.0,
        lambda_l: float = 0.0,
        lambda_perp: float = 0.0,
        lambda_perp_fraction: float = 0.0,
        lambda_rigid: float = 0.0,
        lambda_dihedral: float = 0.0,
        lambda_m: float = 0.0,
        metric_objective: str | None = None,
        property_loss: PropertyLossKind | str | None = None,
        property_loss_normalization: str = POCKET_RMSD_LOSS_NORMALIZATION,
        residual_penalty: ResidualPenaltyKind | str = ResidualPenaltyKind.FRACTION,
        rho_max: float | None = None,
    ) -> None:
        # The FORM first, because it decides what the property loss is allowed to be. There is
        # exactly one form on this path and it is still validated and recorded: see
        # ``objective.validated_metric_objective``, which refuses the hinge spelling BY NAME.
        self.metric_objective = validated_metric_objective(metric_objective)
        required = self.REQUIRED_PROPERTY_LOSS[self.metric_objective]
        if property_loss is None:
            property_loss = required
        # Validated BEFORE the base constructor, because the base constructor runs the driver check
        # and ``lambda_m`` is this class's declared driver.
        self.lambda_m = _validated_lambda_m(lambda_m)
        super().__init__(
            lambda_y=lambda_y,
            lambda_r=lambda_r,
            lambda_g=lambda_g,
            lambda_c=lambda_c,
            lambda_l=lambda_l,
            lambda_perp=lambda_perp,
            lambda_perp_fraction=lambda_perp_fraction,
            lambda_rigid=lambda_rigid,
            lambda_dihedral=lambda_dihedral,
            additional_driving_weights={"lambda_m": self.lambda_m},
            property_loss=property_loss,
            property_loss_normalization=property_loss_normalization,
            residual_penalty=residual_penalty,
            rho_max=rho_max,
        )
        if self.lambda_y != 0.0:
            raise ValueError(
                "lambda_y prices a PROGRESS-COORDINATE target and this objective has none: its "
                "property term IS the graded Angstrom deviation to a deposited cryptic conformer, "
                "priced by lambda_m. On pocket_progress_coordinate lambda_y is the weight on the progress coordinate; "
                "here there is no the progress coordinate, no property_reference table and no sigma_eligibility band -- "
                "the shared target channel is already filled by the deviation, so a nonzero value "
                "would multiply a channel this objective has spent. Set lambda_y to 0 and lambda_m "
                "to the metric weight"
            )
        if self.property_loss is not required:
            raise ValueError(
                f"metric_objective={self.metric_objective!r} IS the shared property term computed "
                f"as {required.value!r}, so property_loss={self.property_loss.value!r} would train "
                "a second objective under an identity naming the first. The zero of this form is "
                "at m = 0 and the target is a pure origin, so the two-sided square IS 0.5*(m/s_p)^2 "
                "up to the convention factor"
            )
        if self.property_loss_normalization != POCKET_RMSD_LOSS_NORMALIZATION:
            raise ValueError(
                "the direct-metric objective divides the Angstrom offset by THIS protein's own "
                f"apo<->holo separation s_p, which is the {POCKET_RMSD_LOSS_NORMALIZATION!r} "
                "contract: a required, strictly positive, per-observation divisor on the absolute "
                f"channel. Received {self.property_loss_normalization!r}, under which the offset "
                "would be priced in raw Angstrom and the objective would weight the largest "
                "transitions most -- while success@B weights every system once"
            )
        if self.lambda_g <= 0.0:
            raise ValueError(
                "lambda_g must be positive for a direct-metric objective. Residues outside the "
                "benchmark's metric region receive EXACTLY ZERO gradient from the property term -- "
                "measured on O74933, where 419 of a 486-residue draw's residues see a maximum "
                "absolute gradient of 0.0 against 7.65e-3 inside the region -- so at lambda_g = 0 "
                "nothing in the objective prevents the adapter from satisfying the metric by "
                "deforming the rest of the protein, and the resulting structure would score and be "
                "unusable. D-79 section 7.3 measured lambda_g at 0.04% of the loss budget on the "
                "the progress coordinate arms, which is a true statement about an objective that reaches every residue "
                "and false about this one"
            )
        if self.lambda_c > 0.0:
            raise ValueError(
                "the paired dose term prices a GAP between two levels of one draw, and the two "
                "pocket-RMSD levels measure deviations to DIFFERENT deposited conformers. Their "
                "difference is not a dose gap -- it is the difference of two unsigned distances to "
                "two crystals -- so lambda_c would weight a quantity with no dose meaning. Draw "
                "cancellation additionally needs a property LINEAR in the additive draw "
                "decomposition, and an RMS deviation is not"
            )

    def forward(  # type: ignore[override]
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        residual_rms: torch.Tensor,
        residual_fraction: torch.Tensor,
        geometry_loss: torch.Tensor | None = None,
        *,
        property_scale: torch.Tensor | None = None,
        pair_group: torch.Tensor | None = None,
        paired_loss: torch.Tensor | None = None,
        raw_residual_fraction: torch.Tensor | None = None,
        local_structure: torch.Tensor | None = None,
        metric_deviation_angstrom: torch.Tensor | None = None,
    ) -> PocketRmsdSteeringLossComponents:
        """Return the shared components plus the Angstrom deviation column.

        ``prediction`` is ``m + s_p``, ``target`` is ``s_p`` and ``property_scale`` is ``s_p``, so
        the shared channel computes ``mean_n (m_n/s_p)^2``. Both halves of that are re-derived here
        against the unclamped ``m`` on the wire rather than trusted from the call site, because a
        missing shift and a swapped origin both produce a plausible loss curve and no other symptom.
        """
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must share a shape, one value per observation")
        if bool(torch.any(prediction.detach() < target.detach())):
            raise ValueError(
                "prediction must be the deviation placed on this protein's own target, so "
                "prediction >= target holds for every observation. A smaller value means the "
                "origin shift by s_p was skipped or the wrong protein's origin was used, and "
                "neither has a downstream symptom"
            )
        if metric_deviation_angstrom is None:
            raise ValueError(
                "this objective places its zero at m = 0 and carries the protein's own origin s_p "
                "in BOTH prediction and target, so the measured m is the only thing that can say "
                "the shift was applied and applied per system. It must be supplied on "
                "metric_deviation_angstrom"
            )
        _validate_loss_tensor(metric_deviation_angstrom, "metric_deviation_angstrom")
        if metric_deviation_angstrom.shape != prediction.shape:
            raise ValueError(
                "metric_deviation_angstrom must share the prediction shape, one deviation per "
                "observation"
            )
        if metric_deviation_angstrom.device != prediction.device:
            raise ValueError("loss tensors must share a device")
        if bool(torch.any(metric_deviation_angstrom.detach() < 0)):
            raise ValueError(
                "metric_deviation_angstrom is an RMS deviation and must be nonnegative"
            )
        offset = (prediction.detach() - target.detach()) - metric_deviation_angstrom.detach()
        if bool(torch.any(offset.abs() > self.ORIGIN_TOLERANCE_ANGSTROM)):
            raise ValueError(
                "prediction - target must BE the measured deviation: the prediction is m placed on "
                "this protein's own origin and the target is that origin, so the two cancel "
                f"exactly. The largest disagreement is {float(offset.abs().max()):.6f} Angstrom, "
                f"above {self.ORIGIN_TOLERANCE_ANGSTROM}. A missing shift prices (m - s_p)^2 and "
                "another protein's origin prices a difference of separations; both are finite, "
                "plausible and wrong"
            )
        shared = super().forward(
            prediction,
            target,
            residual_rms,
            residual_fraction,
            geometry_loss,
            property_scale=property_scale,
            pair_group=pair_group,
            paired_loss=paired_loss,
            raw_residual_fraction=raw_residual_fraction,
            local_structure=local_structure,
        )
        # ``total`` is the shared total UNCHANGED: this objective appends no term.
        return PocketRmsdSteeringLossComponents(
            shared.total,
            shared.target,
            shared.residual,
            shared.geometry,
            shared.contrastive,
            shared.local_structure,
            metric_deviation_angstrom.mean(),
        )

    def objective_config_extra(self) -> dict[str, object]:
        """Record the FORM and the weight that prices it.

        The form rides the identity unconditionally, including where it is the only one: if a
        second form is ever added, a checkpoint that named neither could be resumed under either.
        The per-protein ``s_p`` values are deliberately NOT here -- they are measurements of a
        deposited pair, they reach the identity through the arm's own block and through the
        system-target table's digest, and hundreds of them in an objective identity would be a
        measurement table wearing an identity's clothes.
        """
        config: dict[str, object] = dict(super().objective_config_extra())
        config["metric_objective"] = self.metric_objective
        # The weight that prices the metric. Not redundant bookkeeping:
        # ``pocket_conformer_rmsd_angstrom`` is a persisted journal column and every persisted
        # column is gated on an objective-identity key, so that a journal written by one objective
        # cannot be resumed under another that prices the term differently.
        config["lambda_m"] = float(self.lambda_m)
        return config
