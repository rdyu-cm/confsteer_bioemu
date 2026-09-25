"""The domain-motion steering objective: the shared loss, held to the direct metric's contract.

SCOPE: DOMAIN MOTIONS. Training-only, for the reason the cryptic loss is: no evaluation computes a
loss, so a change here cannot change a generated structure or a scored number and must not be able
to invalidate an evaluation.

**This subclass adds no term to** ``total``, **and that is the design rather than an omission.**
The direct metric IS the property term, under BOTH of
:data:`~sae_bioemu.domain_motion.objective.METRIC_OBJECTIVE_FORMS`.

Under the DEFAULT, :data:`~sae_bioemu.domain_motion.objective.METRIC_OBJECTIVE_MSE`::

    prediction = m + s_p           the measured N/CA/C/O RMSD, on this system's own origin
    target     = s_p               that origin
    scale      = s_p               that system's reference separation
    property_loss = MSE

so the shared ``SteeringLoss`` computes ``mean_n ((prediction - target)/scale)^2`` = ``mean_n
(m_n/s_p)^2``, which is ``2 * mean_n metric_mse(m_n, s_p)``. **The factor of two is constant and is
absorbed into** ``lambda_y``; the objective module states ``L`` in the ``0.5*z^2`` convention
because that is the ``tau = 0, transition -> infinity`` limit of the hinge and pinning the two to
each other is what makes "the MSE is the hinge's special case" a measured claim rather than a
sentence. ``tau`` reaches no part of this form -- not its value, not its gradient, and not the
digest of the per-system target table the checkpoint is bound to -- which is the point of design
point 1: ``tau`` is an EVALUATION parameter.

The ORIGIN is not cosmetic and it is not free. ``SystemTargetConstant`` requires a strictly positive
per-system target, and it lives in ``steering/schedule.py``, which is in ``generation_shared``: a
one-line widening there to admit ``target = 0`` would move the digest of every steering checkpoint
there has ever been, for a change none of them could see. So the MSE form places its zero at ``s_p``
and cancels it exactly. What stops that being a silent re-parameterisation is the elementwise check
below: ``prediction - target`` is re-derived against the measured ``m`` that rides the journal wire,
on every batch.

Under the ABLATION, :data:`~sae_bioemu.domain_motion.objective.METRIC_OBJECTIVE_HINGE`::

    prediction = max(m, tau_p)     the measured RMSD, clamped at this system's threshold
    target     = tau_p             that threshold
    scale      = s_p               that system's reference separation
    property_loss = HUBER          delta = 1.0, which is :data:`.objective.HINGE_TRANSITION`

and ``mean_n huber((prediction - target)/scale)`` is exactly ``mean_n huberised_hinge((m_n -
tau_p)/s_p)``, term for term. The one-sidedness comes from the CLAMP, not from the loss:
``prediction >= target`` elementwise, so the Huber never sees a negative argument and a draw
already inside ``tau_p`` contributes exactly zero value and exactly zero gradient.

Under both forms ``lambda_y`` plays the metric weight and must be positive.

Routing it through the shared term rather than appending a fourth weight is deliberate. The
extended weight vocabulary (``lambda_perp``, ``lambda_rigid``, ``lambda_dihedral``, ...) lives on
the shared ``SteeringLoss`` surface, so a ``lambda_m`` of our own would be an edit to
``steering/training.py`` -- a file in the SHARED ``training_only`` group, whose blast radius is
every radius-of-gyration, cryptic and kinase checkpoint, for a term none of them price.

WHAT THIS CLASS DOES DO
------------------------

1. **It enforces the contract above, elementwise, on every batch.** Under the hinge,
   ``prediction < target`` for one observation means a caller forgot the clamp, and the consequence
   is silent: the shared Huber is TWO-SIDED, so the objective would quietly start paying gradient to
   polish draws that already succeed, which is the exact failure mode design point 1 rejects. Under
   the MSE the check is STRONGER, because the measured ``m`` is on the wire and the contract is an
   equality: ``prediction - target`` must BE ``m``, to a float tolerance. A shifted origin that was
   not shifted, or was shifted by another system's number, is a finite and plausible loss and has no
   other symptom. It is refused rather than detected in a plot three arms later.
2. **It refuses ``lambda_g = 0``.** Under a progress coordinate the geometry penalty is a fidelity
   regulariser and zero is a defensible ablation. Under a DIRECT METRIC it is load-bearing:
   residues outside ``metric_residues`` -- the ones NOT resolved in both crystals -- receive
   EXACTLY ZERO gradient from the property term, because they appear in neither the superposition
   nor the measurement. "Satisfy the metric by deforming the rest of the protein" is therefore
   entirely unconstrained by the objective unless ``lambda_g`` (and, where a control is generated,
   ``lambda_l``) constrains it. Those two terms are structural here, not cosmetic.
3. **It journals the metric in Angstrom.** The hinge is flat below ``tau_p``, so a hinge of zero
   says nothing about whether the draw is at 2.9 Angstrom or at 0.4, and an arm whose only recorded
   term is the hinge is undiagnosable exactly where it is succeeding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

import torch

from sae_bioemu.domain_motion.objective import (
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
    "DIRECT_METRIC_LOSS_NORMALIZATION",
    "DomainMotionSteeringLoss",
    "DomainMotionSteeringLossComponents",
]

#: The declared normalisation this objective runs under. It is
#: :data:`~sae_bioemu.steering.training.SIGMA_LOSS_NORMALIZATION` because that is the one declared
#: contract whose SHAPE is what the direct metric needs -- a required, strictly positive,
#: per-observation divisor applied to the absolute channel and to no other -- and because the
#: alternative is a fifth declaration, which would be an edit to the shared ``training.py``.
#:
#: **The divisor is NOT a conformational spread, and the name is inherited rather than
#: descriptive.** It is ``s_p``, this system's own deposited-conformer separation in Angstrom
#: (objective design point 3). That is recorded per system in the artifact identity, so a report
#: reading ``sequence_sigma`` on a domain-motion run can find out what the number actually was.
DIRECT_METRIC_LOSS_NORMALIZATION = SIGMA_LOSS_NORMALIZATION


def _validated_lambda_m(value: object) -> float:
    """Return ``lambda_m`` as a finite nonnegative float, or fail closed.

    Extracted so the weight can be validated BEFORE ``super().__init__()``, which is where the
    shared driver check runs. A BYTE-FOR-BYTE TWIN of
    ``cryptic_steering.loss._validated_lambda_m`` rather than an import, for the reason
    ``huberised_hinge`` is duplicated between the two packages: that file is in
    ``cryptic_training_only`` and this one is in ``domain_motion_training_only``, so an import
    would bind every domain-motion checkpoint to cryptic code a domain-motion run does not
    execute -- the over-binding half of the defect the property registry exists to end.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("lambda_m must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("lambda_m must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class DomainMotionSteeringLossComponents(SteeringLossComponents):
    """The shared components plus the measured metric, in the unit the benchmark grades."""

    #: Mean over observations of the superposed N/CA/C/O RMSD to the level's own reference
    #: conformer, in ANGSTROM and UNWEIGHTED. Not a term of :attr:`total` -- the hinge of it is --
    #: and journalled for that reason: the hinge is identically zero for every successful draw, so
    #: it cannot show an arm improving from 2.9 Angstrom to 1.2.
    direct_metric_angstrom: torch.Tensor | None = None

    def journal_columns(self) -> dict[str, float | None]:
        return {
            "direct_metric_angstrom": (
                None
                if self.direct_metric_angstrom is None
                else float(self.direct_metric_angstrom.detach().reshape(()))
            )
        }


class DomainMotionSteeringLoss(SteeringLoss):
    """The shared steering objective, contracted to the direct N/CA/C/O metric.

    See the module docstring for why no term is appended and what is enforced instead.
    """

    #: One per-observation input the shared runner reads off its generated sample: the measured
    #: RMSD in Angstrom, for the journal. Declared here rather than named in the runner, which is
    #: what keeps a domain-motion tensor out of the shared training loop's signature.
    #:
    #: ``metric_deviation_angstrom`` is the TRANSPORT name -- the field on the shared
    #: ``GeneratedTrainingSample`` -- and the cryptic direct-metric term fills the same wire with
    #: its own deviation. The wire is shared because the two are the same SHAPE of input and a
    #: second field of identical shape is how one call site comes to fill the other's; the two
    #: quantities are NOT shared and keep separate journal columns, because they are measured over
    #: different atom sets against different references. This is the ``pocket_off_path_squared``
    #: precedent, which the kinase programme reuses for exactly this reason.
    EXTRA_SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ("metric_deviation_angstrom",)

    #: None. Every extended weight belongs to another programme's term, and a nonzero value for any
    #: of them is refused by the shared constructor with a reason rather than accepted and dropped.
    #: ``lambda_m`` is NOT here because it is not in the shared ``EXTENDED_LOSS_TERMS`` vocabulary
    #: at all -- neither is the cryptic one -- so it is validated by this class directly.
    PRICED_EXTENDED_TERMS: ClassVar[frozenset[str]] = frozenset()

    #: ``lambda_m`` is THE driver, and ``lambda_y`` is refused outright above. This objective has
    #: no progress-coordinate target for ``lambda_y`` to price, so declaring it a driver as well
    #: would let a config satisfy the check with a weight applied to nothing.
    DRIVING_WEIGHTS: ClassVar[tuple[str, ...]] = ("lambda_y", "lambda_m")

    #: The CONVENTION FACTOR between the shared property channel and this property's own ``L``,
    #: by form, and the reason it lives here rather than in the weight (D-87).
    #:
    #: ``objective.metric_mse`` is ``0.5*(m/s_p)^2`` and ``objective.huberised_hinge`` is
    #: ``0.5*z^2`` on its quadratic branch, which is the convention the cryptic ``lambda_m``
    #: multiplies. The shared channel does not agree with itself across the two forms:
    #:
    #:   MSE    ``normalized_error.square().mean()``            = ``mean(z^2)``   = ``2 * mean(L)``
    #:   HINGE  ``functional.huber_loss(..., delta=1.0)``       = ``mean(L)``     exactly
    #:
    #: So a single weight multiplying the shared channel MEANT TWO DIFFERENT THINGS on the two
    #: forms, by a factor of two, before this table existed -- an arm at one number would have
    #: descended twice the loss under one form that it did under the other. Putting the factor in
    #: the arithmetic makes ``lambda_m`` mean ``L``'s multiplier under both forms and on the
    #: cryptic path alike, which is the whole content of D-87: one name, one quantity.
    #:
    #: The shipped MSE arm therefore moves from ``lambda_y = 0.5`` to ``lambda_m = 1.0`` and
    #: descends the BIT-IDENTICAL loss: ``0.5 * 1.0 * mean(z^2)`` is the same expression, the same
    #: operand order and the same rounding as ``0.5 * mean(z^2)``.
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
    #: ``s_p`` and another system's origin by the difference of two separations.
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
        # THE WEIGHT ON THE GRADED METRIC (D-87), spelled exactly as the cryptic pocket spells it
        # and meaning exactly what it means there: the multiplier on this property's own
        # ``metric_mse``/``huberised_hinge``, per observation, meaned over the batch. NOT a member
        # of the shared ``EXTENDED_LOSS_TERMS`` vocabulary -- neither is the cryptic one, which is
        # why this costs no shared-vocabulary edit -- so this class validates it itself, in both
        # directions, and declares it a DRIVER so ``lambda_y = 0`` is legal beside it.
        lambda_m: float = 0.0,
        metric_objective: str | None = None,
        property_loss: PropertyLossKind | str | None = None,
        property_loss_normalization: str = DIRECT_METRIC_LOSS_NORMALIZATION,
        residual_penalty: ResidualPenaltyKind | str = ResidualPenaltyKind.FRACTION,
        rho_max: float | None = None,
    ) -> None:
        # The FORM first, because it decides what the property loss is allowed to be. ``None``
        # resolves to plain MSE, which is what an arm that names no form means.
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
                "property term IS the graded Angstrom metric, priced by lambda_m. A nonzero "
                "lambda_y here would be the second meaning for one name that D-87 removed -- on a "
                "cryptic arm lambda_y = 0 REMOVES the the progress coordinate proxy, and here it would have deleted the "
                "objective. Set lambda_y to 0 and lambda_m to the metric weight"
            )
        if self.property_loss is not required:
            raise ValueError(
                f"metric_objective={self.metric_objective!r} IS the shared property term computed "
                f"as {required.value!r}, so property_loss={self.property_loss.value!r} would train "
                "a third objective under an identity naming one of the two. "
                f"{METRIC_OBJECTIVE_MSE!r} needs {PropertyLossKind.MSE.value!r}: its zero is at "
                "m = 0 and the target is a pure origin, so the two-sided square IS 0.5*(m/s_p)^2 "
                f"up to the factor lambda_y absorbs. {METRIC_OBJECTIVE_HINGE!r} needs "
                f"{PropertyLossKind.HUBER.value!r}: under MSE the thresholded term would become "
                "mean((m - tau_p)^2/s_p^2), unbounded in the far tail, and one hopeless system "
                "would own the batch gradient -- the thing the hinge exists to prevent"
            )
        if self.property_loss_normalization != DIRECT_METRIC_LOSS_NORMALIZATION:
            raise ValueError(
                "the direct-metric objective divides the Angstrom offset by THIS system's own "
                f"reference separation s_p, which is the {DIRECT_METRIC_LOSS_NORMALIZATION!r} "
                "contract: a required, strictly positive, per-observation divisor on the absolute "
                f"channel. Received {self.property_loss_normalization!r}, under which the offset "
                "would be priced in raw Angstrom and the objective would weight the largest "
                "transitions most, while success@B weights every system once"
            )
        if self.lambda_g <= 0.0:
            raise ValueError(
                "lambda_g must be positive for a direct-metric objective. Residues outside the "
                "metric set -- those not resolved in BOTH deposited conformers -- receive exactly "
                "zero gradient from the property term, because they enter neither the "
                "superposition nor the measurement. At lambda_g = 0 nothing in the objective "
                "prevents the adapter from satisfying the metric by deforming the rest of the "
                "protein, and the resulting structure would score and be unusable"
            )
        if self.lambda_c > 0.0:
            raise ValueError(
                "the paired dose term prices a GAP between two levels of one draw, and the two "
                "domain-motion levels measure distances to DIFFERENT reference conformers. Their "
                "difference is not a dose gap -- it is the difference of two unsigned distances "
                "to two structures -- so lambda_c would weight a quantity with no dose meaning"
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
    ) -> DomainMotionSteeringLossComponents:
        """Return the shared components plus the Angstrom metric column.

        Under the hinge ``prediction`` is ``max(m, tau_p)`` and ``target`` is ``tau_p``; under the
        default MSE they are ``m + s_p`` and ``s_p``. ``property_scale`` is ``s_p`` under both. The
        elementwise checks below are the ones that matter: without the clamp the shared Huber is
        two-sided and the objective starts paying gradient to improve draws that already succeed,
        and without the origin shift the MSE prices ``(m - s_p)^2`` -- a finite, plausible loss that
        asks every draw to sit one whole reference separation away from its own target conformer --
        at every other check green.

        ``metric_deviation_angstrom`` is the UNCLAMPED ``m``, for the journal only. It enters no
        term: it is already computed on the path to ``prediction``, and it is journalled because
        the hinge is identically zero for every successful draw and so cannot show an arm
        improving from 2.9 Angstrom to 1.2. Its persisted column is ``direct_metric_angstrom``,
        gated on :meth:`objective_config_extra`'s ``direct_metric_weight``.
        """
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must share a shape, one value per observation")
        if bool(torch.any(prediction.detach() < target.detach())):
            raise ValueError(
                "prediction must be the metric placed on this system's own target, so "
                "prediction >= target holds for every observation. Under "
                f"{METRIC_OBJECTIVE_HINGE!r} a smaller value means the CLAMP at tau_p was skipped "
                "and the shared Huber would then be two-sided: the objective would pay gradient to "
                "polish draws already inside tau, which is the defect the one-sided hinge exists "
                f"to avoid. Under {METRIC_OBJECTIVE_MSE!r} it means the origin shift by s_p was "
                "skipped or the wrong system's origin was used. Neither has a downstream symptom"
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
                    "m = 0 and carries the system's own origin s_p in BOTH prediction and target, "
                    "so the measured m is the only thing that can say the shift was applied and "
                    "applied per system. It must be supplied on metric_deviation_angstrom"
                )
            offset = (prediction.detach() - target.detach()) - metric_deviation_angstrom.detach()
            if bool(torch.any(offset.abs() > self.ORIGIN_TOLERANCE_ANGSTROM)):
                raise ValueError(
                    "prediction - target must BE the measured RMSD under "
                    f"metric_objective={METRIC_OBJECTIVE_MSE!r}: the prediction is m placed on this "
                    "system's own origin and the target is that origin, so the two cancel exactly. "
                    f"The largest disagreement is {float(offset.abs().max()):.6f} Angstrom, above "
                    f"{self.ORIGIN_TOLERANCE_ANGSTROM}. A missing shift prices (m - s_p)^2 and "
                    "another system's origin prices a difference of separations; both are finite, "
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
        # ``total`` is the shared total UNCHANGED: this objective appends no term. See the module
        # docstring -- the priced form IS the shared target term, by construction of its inputs.
        return DomainMotionSteeringLossComponents(
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
        one ``lambda_y`` under the two forms are two objectives, and an identity naming neither
        would let a checkpoint trained under one be resumed under the other.

        :data:`~sae_bioemu.domain_motion.objective.HINGE_TRANSITION` decides how much of a batch's
        gradient one hopeless system can own, so it is objective IDENTITY -- and it is recorded only
        on the path that HAS a knee. Under the MSE the quadratic branch is the whole function, so a
        recorded transition would be a parameter applied to nothing, which is the shape the whole
        conditional-key discipline exists to prevent. It is pinned to the shared Huber's ``delta``,
        hard-coded to 1.0 in ``SteeringLoss.forward``; recording it makes a future divergence
        between the two visible in a checkpoint rather than silent.
        """
        config: dict[str, object] = dict(super().objective_config_extra())
        config["metric_objective"] = self.metric_objective
        if self.metric_objective == METRIC_OBJECTIVE_HINGE:
            config["hinge_transition"] = float(HINGE_TRANSITION)
        # The weight that prices the metric, RESTATED under a name only this objective writes.
        #
        # It is not redundant bookkeeping: ``direct_metric_angstrom`` is a persisted journal
        # column and every persisted column is gated on an objective-identity key, so that a
        # journal written by one objective cannot be resumed under another that prices the term
        # differently. ``lambda_y`` cannot serve -- it is a SHARED weight and every objective
        # there has ever been carries a positive one, so a column gated on it would be demanded
        # of every property's rows including those that cannot compute it. Here ``lambda_y`` IS
        # the metric weight (the hinge is the shared target term by construction), so the honest
        # gate is that same number under a name that says what it prices.
        config["lambda_m"] = float(self.lambda_m)
        # ONE NUMBER, ONE NAME. ``direct_metric_weight`` was this weight's third spelling, and it
        # existed only because the second -- the shared ``lambda_y`` -- could not serve as a journal
        # column's gate: every objective there has ever been carries a positive ``lambda_y``, so a
        # column gated on it would be demanded of every property's rows. With the weight spelled
        # ``lambda_m``, which only this objective and the cryptic one write, the honest gate is the
        # weight itself and the restatement is gone. The journal column's declaration in the
        # property catalog moves with it.
        return config
