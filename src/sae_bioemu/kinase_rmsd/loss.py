"""The kinase-RMSD steering objective: the shared loss, held to the direct metric's contract.

SCOPE: KINASES. Training-only, for the reason the cryptic and domain-motion losses are: no
evaluation computes a loss, so a change here cannot change a generated structure or a scored number
and must not be able to invalidate an evaluation.

**This subclass adds no term to** ``total``, **and that is the design rather than an omission.**
The direct metric IS the property term, under BOTH of
:data:`~sae_bioemu.kinase_rmsd.objective.METRIC_OBJECTIVE_FORMS`.

Under the DEFAULT, :data:`~sae_bioemu.kinase_rmsd.objective.METRIC_OBJECTIVE_MSE`::

    prediction = m + s_p           the measured N/CA/C/O RMSD, on this kinase's own origin
    target     = s_p               that origin
    scale      = s_p               that kinase's deposited-chain separation
    property_loss = MSE

so the shared ``SteeringLoss`` computes ``mean_n ((prediction - target)/scale)^2`` = ``mean_n
(m_n/s_p)^2``, which is ``2 * mean_n metric_mse(m_n, s_p)``; the factor of two lives in
:attr:`KinaseRmsdSteeringLoss.CONVENTION_FACTOR` rather than in the weight, so ``lambda_m``
multiplies ``objective.metric_mse`` here exactly as it does on the cryptic and domain-motion paths.
``tau`` reaches no part of this form -- not its value, not its gradient, and not the digest of the
per-system target table the checkpoint is bound to.

Under the ABLATION, :data:`~sae_bioemu.kinase_rmsd.objective.METRIC_OBJECTIVE_HINGE`::

    prediction = max(m, tau_p)     the measured RMSD, clamped at this kinase's threshold
    target     = tau_p             that threshold
    scale      = s_p               that kinase's separation
    property_loss = HUBER          delta = 1.0, which is :data:`.objective.HINGE_TRANSITION`

and the one-sidedness comes from the CLAMP, not from the loss.

WHY ``lambda_y`` IS SPELLED AND REFUSED RATHER THAN OMITTED
------------------------------------------------------------

``lambda_y`` is in ``bioemu/steering_workflow.py``'s ``_SHARED_RUNTIME_KEYS`` and is read
UNCONDITIONALLY, so a config that leaves it out does not describe an objective without it -- it
fails the runtime schema. The config therefore says ``lambda_y: 0.0`` out loud and this
constructor RAISES BY NAME on any nonzero value.

That is the DOMAIN-MOTION idiom and deliberately not the cryptic one. On a cryptic arm ``lambda_y``
prices the ``the progress coordinate`` progress-coordinate target and setting it to zero REMOVES that proxy, so the
cryptic loss builds the term and journals it at zero weight. Here there is no ``the progress coordinate`` at all: this
property has no progress coordinate, no ``property_reference`` table and no ``sigma_eligibility``
band, so a nonzero ``lambda_y`` would be a weight multiplying a channel this objective has already
filled with its own graded Angstrom metric -- one name meaning two opposite things, decided by
which config a reader had open. It is refused.

WHAT THIS CLASS DOES DO
------------------------

1. **It enforces the contract above, elementwise, on every batch.** Under the MSE the check is an
   EQUALITY against the measured ``m`` on the wire: ``prediction - target`` must BE ``m``. A
   shifted origin that was not shifted, or was shifted by another kinase's number, is a finite and
   plausible loss with no other symptom.
2. **It refuses ``lambda_g = 0``.** Under a direct metric the geometry penalty is load-bearing
   rather than a regulariser, and on THIS property more so than on domain motion: the metric set is
   a ~15-residue activation loop inside a ~260-residue domain, so about 94% of the protein receives
   EXACTLY ZERO gradient from the property term. "Satisfy the metric by deforming the rest of the
   kinase" is unconstrained by the objective unless ``lambda_g`` constrains it.
3. **It journals the metric in Angstrom, under its own column name.**
   ``kinase_conformer_rmsd_angstrom`` is DISTINCT from the cryptic ``metric_deviation_angstrom``
   and from the domain-motion ``direct_metric_angstrom``, because
   ``property_registry.journal_column_weights`` refuses a column declared against two weights and
   because the three are not the same quantity: one is measured over a pocket's metric region, one
   over a domain-motion metric set, and this one over an activation loop against a Kincore-labelled
   chain. One name carries one gate and one meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import torch

from sae_bioemu.kinase_rmsd.objective import (
    HINGE_TRANSITION,
    METRIC_OBJECTIVE_HINGE,
    METRIC_OBJECTIVE_MSE,
    validated_metric_objective,
)
from sae_bioemu.steering.training import (
    SIGMA_LOSS_NORMALIZATION,
    PropertyLossKind,
    ResidualPenaltyKind,
    SteeringLoss,
    SteeringLossComponents,
    _validate_loss_tensor,
)

__all__ = [
    "KINASE_RMSD_JOURNAL_COLUMN",
    "KINASE_RMSD_LOSS_NORMALIZATION",
    "KinaseRmsdSteeringLoss",
    "KinaseRmsdSteeringLossComponents",
]

#: The declared normalisation this objective runs under: a required, strictly positive,
#: per-observation divisor applied to the absolute channel and to no other.
#:
#: **The divisor is NOT a conformational spread, and the name is inherited rather than
#: descriptive.** It is ``s_p``, this kinase's own deposited-chain separation in Angstrom. That is
#: recorded per system in the artifact identity, so a report reading ``sequence_sigma`` on a
#: kinase-RMSD run can find out what the number actually was.
KINASE_RMSD_LOSS_NORMALIZATION = SIGMA_LOSS_NORMALIZATION

#: The persisted training column carrying the UNCLAMPED metric in Angstrom. Its own name, gated on
#: ``lambda_m``: see the module docstring's point 3.
KINASE_RMSD_JOURNAL_COLUMN = "kinase_conformer_rmsd_angstrom"


def _validated_lambda_m(value: object) -> float:
    """Return ``lambda_m`` as a finite nonnegative float, or fail closed.

    Extracted so the weight can be validated BEFORE ``super().__init__()``, which is where the
    shared driver check runs. A BYTE-FOR-BYTE TWIN of the cryptic and domain-motion helpers rather
    than an import, for the reason :mod:`sae_bioemu.kinase_rmsd.frames` is a copy: those files are
    in ``cryptic_training_only`` and ``domain_motion_training_only``, so an import would bind every
    kinase-RMSD checkpoint to another programme's code that a kinase-RMSD run does not execute.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("lambda_m must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("lambda_m must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class KinaseRmsdSteeringLossComponents(SteeringLossComponents):
    """The shared components plus the measured metric, in Angstrom."""

    #: Mean over observations of the superposed N/CA/C/O RMSD to the deposited chain the level
    #: names, over that kinase's metric set, in ANGSTROM and UNWEIGHTED. Not a term of
    #: :attr:`total` -- the priced form of it is -- and journalled for that reason.
    kinase_conformer_rmsd_angstrom: torch.Tensor | None = None

    def journal_columns(self) -> dict[str, float | None]:
        return {
            KINASE_RMSD_JOURNAL_COLUMN: (
                None
                if self.kinase_conformer_rmsd_angstrom is None
                else float(self.kinase_conformer_rmsd_angstrom.detach().reshape(()))
            )
        }


class KinaseRmsdSteeringLoss(SteeringLoss):
    """The shared steering objective, contracted to the direct N/CA/C/O kinase metric."""

    #: One per-observation input the shared runner reads off its generated sample: the measured
    #: RMSD in Angstrom, for the journal.
    #:
    #: ``metric_deviation_angstrom`` is the TRANSPORT name -- the field on the shared
    #: ``GeneratedTrainingSample`` -- which the cryptic and domain-motion direct-metric terms fill
    #: too. The wire is shared because the three are the same SHAPE of input; the QUANTITIES are
    #: not shared and keep separate journal columns.
    EXTRA_SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ("metric_deviation_angstrom",)

    #: None. Every extended weight belongs to another programme's term, and a nonzero value for any
    #: of them is refused by the shared constructor with a reason rather than accepted and dropped.
    #: ``lambda_m`` is NOT here because it is not in the shared ``EXTENDED_LOSS_TERMS`` vocabulary
    #: at all, so it is validated by this class directly.
    PRICED_EXTENDED_TERMS: ClassVar[frozenset[str]] = frozenset()

    #: ``lambda_m`` is THE driver, and ``lambda_y`` is refused outright below. ``lambda_y`` is
    #: declared here anyway because ``_require_a_driving_weight`` reads this tuple and the shared
    #: weight must remain expressible as a driver on the shared surface; what makes it unusable on
    #: this property is the explicit refusal, which names the reason.
    DRIVING_WEIGHTS: ClassVar[tuple[str, ...]] = ("lambda_y", "lambda_m")

    #: The CONVENTION FACTOR between the shared property channel and this property's own ``L``, by
    #: form. The shared channel does not agree with itself across the two forms:
    #:
    #:   MSE    ``normalized_error.square().mean()``      = ``mean(z^2)``  = ``2 * mean(L)``
    #:   HINGE  ``functional.huber_loss(..., delta=1.0)`` = ``mean(L)``    exactly
    #:
    #: so one weight multiplying the shared channel would mean two different things on the two
    #: forms, by a factor of two. Putting the factor in the arithmetic makes ``lambda_m`` mean
    #: ``L``'s multiplier under both forms and on the cryptic and domain-motion paths alike, which
    #: is the whole content of D-87: one name, one quantity.
    CONVENTION_FACTOR: ClassVar[dict[str, float]] = {
        METRIC_OBJECTIVE_MSE: 0.5,
        METRIC_OBJECTIVE_HINGE: 1.0,
    }

    @property
    def property_term_weight(self) -> float:
        """``lambda_m`` converted into the shared channel's own convention."""
        return self.CONVENTION_FACTOR[self.metric_objective] * self.lambda_m

    #: Which shared property loss each form REQUIRES. Not a default and not a preference: the two
    #: entries are the arithmetic the two objectives are defined as, so a configuration naming one
    #: form and the other loss is refused rather than silently trained.
    REQUIRED_PROPERTY_LOSS: ClassVar[dict[str, PropertyLossKind]] = {
        METRIC_OBJECTIVE_MSE: PropertyLossKind.MSE,
        METRIC_OBJECTIVE_HINGE: PropertyLossKind.HUBER,
    }

    #: How far ``prediction - target`` may sit from the measured ``m`` on the MSE path, in Angstrom.
    #: Loose enough for float32 addition and subtraction of two numbers of order 10 Angstrom; far
    #: tighter than any real disagreement, because a missing origin shift moves it by a whole
    #: ``s_p`` and another kinase's origin by the difference of two separations.
    ORIGIN_TOLERANCE_ANGSTROM: ClassVar[float] = 1e-3

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
        property_loss_normalization: str = KINASE_RMSD_LOSS_NORMALIZATION,
        residual_penalty: ResidualPenaltyKind | str = ResidualPenaltyKind.FRACTION,
        rho_max: float | None = None,
    ) -> None:
        # The FORM first, because it decides what the property loss is allowed to be.
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
                "property term IS the graded Angstrom RMSD to a deposited DFG conformer, priced by "
                "lambda_m. On the DFG-coordinate property lambda_y is the weight on the progress coordinate; here there "
                "is no the progress coordinate, no property_reference table and no sigma_eligibility band, so a nonzero "
                "value would multiply a channel this objective has already filled. Set lambda_y to "
                "0 and lambda_m to the metric weight"
            )
        if self.property_loss is not required:
            raise ValueError(
                f"metric_objective={self.metric_objective!r} IS the shared property term computed "
                f"as {required.value!r}, so property_loss={self.property_loss.value!r} would train "
                "a third objective under an identity naming one of the two. "
                f"{METRIC_OBJECTIVE_MSE!r} needs {PropertyLossKind.MSE.value!r}: its zero is at "
                "m = 0 and the target is a pure origin, so the two-sided square IS 0.5*(m/s_p)^2 "
                f"up to the convention factor. {METRIC_OBJECTIVE_HINGE!r} needs "
                f"{PropertyLossKind.HUBER.value!r}: under MSE the thresholded term would become "
                "mean((m - tau_p)^2/s_p^2), unbounded in the far tail, and one kinase whose "
                "activation loop cannot be reached would own the batch gradient"
            )
        if self.property_loss_normalization != KINASE_RMSD_LOSS_NORMALIZATION:
            raise ValueError(
                "the direct-metric objective divides the Angstrom offset by THIS kinase's own "
                f"chain separation s_p, which is the {KINASE_RMSD_LOSS_NORMALIZATION!r} contract: "
                "a required, strictly positive, per-observation divisor on the absolute channel. "
                f"Received {self.property_loss_normalization!r}, under which the offset would be "
                "priced in raw Angstrom and the objective would weight the largest activation-loop "
                "swings most"
            )
        if self.lambda_g <= 0.0:
            raise ValueError(
                "lambda_g must be positive for a direct-metric objective. The metric set here is "
                "the activation loop -- a median of about 15 residues inside a ~260-residue kinase "
                "domain -- so roughly 94% of the protein receives exactly zero gradient from the "
                "property term, because those residues enter neither the superposition nor the "
                "measurement. At lambda_g = 0 nothing in the objective prevents the adapter from "
                "satisfying the metric by deforming the rest of the kinase, and the resulting "
                "structure would score and be unusable"
            )
        if self.lambda_c > 0.0:
            raise ValueError(
                "the paired dose term prices a GAP between two levels of one draw, and the two "
                "kinase-RMSD levels measure distances to DIFFERENT deposited chains. Their "
                "difference is not a dose gap -- it is the difference of two unsigned distances to "
                "two crystals -- so lambda_c would weight a quantity with no dose meaning. The "
                "DFG-coordinate property's own lambda_c was removed for the adjacent reason (K-20): "
                "draw cancellation needs a property LINEAR in the additive draw decomposition, and "
                "an RMSD is not"
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
    ) -> KinaseRmsdSteeringLossComponents:
        """Return the shared components plus the Angstrom metric column.

        Under the hinge ``prediction`` is ``max(m, tau_p)`` and ``target`` is ``tau_p``; under the
        default MSE they are ``m + s_p`` and ``s_p``. ``property_scale`` is ``s_p`` under both.
        """
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must share a shape, one value per observation")
        if bool(torch.any(prediction.detach() < target.detach())):
            raise ValueError(
                "prediction must be the metric placed on this kinase's own target, so "
                f"prediction >= target holds for every observation. Under "
                f"{METRIC_OBJECTIVE_HINGE!r} a smaller value means the CLAMP at tau_p was skipped "
                "and the shared Huber would then be two-sided: the objective would pay gradient to "
                "polish draws already inside tau, which is the defect the one-sided hinge exists "
                f"to avoid. Under {METRIC_OBJECTIVE_MSE!r} it means the origin shift by s_p was "
                "skipped or the wrong kinase's origin was used. Neither has a downstream symptom"
            )
        if metric_deviation_angstrom is not None:
            _validate_loss_tensor(metric_deviation_angstrom, "metric_deviation_angstrom")
            if metric_deviation_angstrom.shape != prediction.shape:
                raise ValueError(
                    "metric_deviation_angstrom must share the prediction shape, one RMSD per "
                    "observation"
                )
            if metric_deviation_angstrom.device != prediction.device:
                raise ValueError("loss tensors must share a device")
            if bool(torch.any(metric_deviation_angstrom.detach() < 0)):
                raise ValueError("metric_deviation_angstrom is an RMSD and must be nonnegative")
        if self.metric_objective == METRIC_OBJECTIVE_MSE:
            if metric_deviation_angstrom is None:
                raise ValueError(
                    f"metric_objective={METRIC_OBJECTIVE_MSE!r} places the objective's zero at "
                    "m = 0 and carries the kinase's own origin s_p in BOTH prediction and target, "
                    "so the measured m is the only thing that can say the shift was applied and "
                    "applied per system. It must be supplied on metric_deviation_angstrom"
                )
            offset = (prediction.detach() - target.detach()) - metric_deviation_angstrom.detach()
            if bool(torch.any(offset.abs() > self.ORIGIN_TOLERANCE_ANGSTROM)):
                raise ValueError(
                    "prediction - target must BE the measured RMSD under "
                    f"metric_objective={METRIC_OBJECTIVE_MSE!r}: the prediction is m placed on this "
                    "kinase's own origin and the target is that origin, so the two cancel exactly. "
                    f"The largest disagreement is {float(offset.abs().max()):.6f} Angstrom, above "
                    f"{self.ORIGIN_TOLERANCE_ANGSTROM}. A missing shift prices (m - s_p)^2 and "
                    "another kinase's origin prices a difference of separations; both are finite, "
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
        metric = None if metric_deviation_angstrom is None else metric_deviation_angstrom.mean()
        # ``total`` is the shared total UNCHANGED: this objective appends no term.
        return KinaseRmsdSteeringLossComponents(
            shared.total,
            shared.target,
            shared.residual,
            shared.geometry,
            shared.contrastive,
            shared.local_structure,
            metric,
        )

    def objective_config_extra(self) -> dict[str, object]:
        """Record the FORM, the knee that only one form has, and the weight that prices it.

        The form rides the identity unconditionally, including where it is the default: two runs at
        one ``lambda_m`` under the two forms are two objectives, and an identity naming neither
        would let a checkpoint trained under one be resumed under the other.
        """
        config: dict[str, object] = dict(super().objective_config_extra())
        config["metric_objective"] = self.metric_objective
        if self.metric_objective == METRIC_OBJECTIVE_HINGE:
            config["hinge_transition"] = float(HINGE_TRANSITION)
        # The weight that prices the metric. It is not redundant bookkeeping:
        # ``kinase_conformer_rmsd_angstrom`` is a persisted journal column and every persisted
        # column is gated on an objective-identity key, so that a journal written by one objective
        # cannot be resumed under another that prices the term differently.
        config["lambda_m"] = float(self.lambda_m)
        return config
