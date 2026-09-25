"""The radius-of-gyration steerable property: its dose set, its frame, and its objective.

Radius of gyration was the only steered property for the whole of the eleven-arm negative and
the ablation series, and every one of the facts below used to be a row in a table inside
``bioemu/steering_workflow.py``. They are here for the reason the cryptic and kinase
descriptors are in their own packages: a table keyed by property name, held in a file every
property's fingerprint hashes, means a change to one property's row re-keys the other two.

This module is hashed in ``rg_generation_shared``, a group only the radius-of-gyration
composition names. Editing it re-keys radius-of-gyration checkpoints -- which is right, because
it decides the dose set, the frame the target is built in and the callback versions -- and
moves no cryptic or kinase digest.

WHY THERE IS NO ``rg_steering`` PACKAGE
---------------------------------------

The cryptic and kinase programmes have packages because each brought a coordinate, an artifact
format, a pair geometry and an objective of its own. Radius of gyration brought none of those:
its surrogate is ``steering/surrogate.py``, its scorer is ``bioemu/steering_evaluation.py`` and
its objective is the SHARED ``SteeringLoss`` with no added term. A package holding one
descriptor would be a directory, not a separation. The separation that matters is the GROUP,
and this file has its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sae_bioemu.steering.adapter import TargetKind
from sae_bioemu.steering.property_registry import PropertyMeasurement, SteerableProperty
from sae_bioemu.steering.schedule import CONTINUOUS_DOSE_LEVELS, TARGET_ORIGIN_SEQUENCE_MEAN

__all__ = ["RADIUS_OF_GYRATION", "RADIUS_OF_GYRATION_PROPERTY_NAME"]

RADIUS_OF_GYRATION_PROPERTY_NAME = "radius_of_gyration"

#: Every weight in the shared loss vocabulary this objective does not price, with the reason.
#:
#: All five are refused rather than ignored, and that is the fail-CLOSED direction: since D-43
#: the property decides which loss CLASS is constructed, and the shared ``SteeringLoss`` accepts
#: none of these, so a radius-of-gyration config that set one would have it read from the
#: config, dropped by the constructor and absent from the objective identity -- a weight
#: configured, recorded nowhere and applied to nothing. That is the exact shape that voided six
#: ``lambda_l`` arms.
#:
#: A ZERO or null value is still accepted, and deliberately: the shipped radius-of-gyration
#: template writes the off-path weights out explicitly at zero to document that they are not for
#: this property, and before D-43 that constructed a byte-identical ``SteeringLoss``. Refusing a
#: zero would break a config that has always been correct. Zero is not the failure; a weight
#: that is priced and then dropped is.
_REFUSED: Mapping[str, str] = {
    "lambda_perp": (
        "the shared objective has no off-path split: there is no experimental transition vector "
        "for a radius of gyration to be perpendicular to"
    ),
    "lambda_perp_fraction": (
        "the per-protein form of the same off-path weight, and there is no off-path term here "
        "for it to parameterise"
    ),
    "lambda_rigid": (
        "the context-rigidity hinge is measured on a pocket's context shell, which a radius of "
        "gyration does not define"
    ),
    "context_setpoint_percentile": (
        "the rigidity hinge's setpoint parameter, and this objective has no hinge to set a point "
        "on"
    ),
    "lambda_dihedral": (
        "the Kincore cosine dihedral distance is measured to a DFG cluster centroid, and no "
        "other objective has a dihedral term"
    ),
}


def _loss_class() -> type:
    """Return the SHARED objective, unchanged since before either project existed.

    A radius-of-gyration run builds ``SteeringLoss`` with exactly the arguments it built before
    the cryptic project existed and never imports another objective at all (D-43).
    """
    from sae_bioemu.steering.training import SteeringLoss

    return SteeringLoss


def _evaluator() -> Any:
    from sae_bioemu.bioemu.steering_evaluation import evaluate_radius_of_gyration

    return evaluate_radius_of_gyration


def _objective_arguments(*, runtime: Mapping[str, Any], arm: Any) -> Mapping[str, Any]:
    """No arguments beyond the shared ones, which is what keeps this objective shared."""
    del runtime, arm
    return {}


def _resolve_arm(*, payload: Mapping[str, Any], base: Any, split: Any, runtime: Any) -> None:
    """Radius of gyration resolves no arm.

    Returning ``None`` is what keeps the whole cryptic and kinase packages out of a
    radius-of-gyration run's import graph and their identities out of its checkpoint.
    """
    del payload, base, split, runtime
    return None


def _no_arm_arguments(arm: Any) -> Mapping[str, Any]:
    del arm
    return {}


#: The two target kinds a radius-of-gyration adapter may be conditioned on. Both are per-sequence
#: SCALARS -- a standardized absolute Rg on one path, the bounded requested level ``o`` on the
#: other -- and the per-residue ``directional_residue`` kind is deliberately not among them.
_SEQUENCE_TARGET_KINDS = frozenset(
    {TargetKind.ABSOLUTE_SEQUENCE.value, TargetKind.RELATIVE_SEQUENCE.value}
)


def _measure_sample(*, batch: Any) -> PropertyMeasurement:
    """Return the differentiable radius of gyration of each steered member, and its target.

    Lifted verbatim out of the first branch of
    ``bioemu/steering_properties.py:generate_exact_property_sample``, which is a member of the
    SHARED ``generation_shared`` group: while it lived there, a fourth property could not be given
    a measurement at all without re-keying this one. The arithmetic is unchanged --
    :func:`~sae_bioemu.steering.surrogate.differentiable_radius_of_gyration` over the padded
    C-alpha Angstrom coordinates under the batch's own validity mask -- and a test pins an Rg
    sample produced through this hook to the literal value the branch produced.

    ``target`` is echoed unchanged: on the absolute path it IS the standardized target and on the
    relative path the workflow builds the real target from this sequence's reference statistics,
    which is not derivable from a structure.

    Imported inside the function, as every other descriptor's callables are: this module is
    resolved by ``property_registry.resolve`` on every radius-of-gyration run, and the surrogate
    brings torch geometry a catalog read must not pay for.
    """
    from sae_bioemu.steering.surrogate import differentiable_radius_of_gyration

    if batch.adapter.target_kind not in _SEQUENCE_TARGET_KINDS or batch.target.shape != (
        len(batch.records),
    ):
        raise ValueError(
            "radius_of_gyration requires one absolute or relative sequence target per member"
        )
    return PropertyMeasurement(
        prediction=differentiable_radius_of_gyration(batch.coordinates_angstrom, batch.mask),
        target=batch.target,
    )


RADIUS_OF_GYRATION = SteerableProperty(
    name=RADIUS_OF_GYRATION_PROPERTY_NAME,
    # CONTINUOUS (D-44): an intermediate level names a real structural ask, so the set is
    # unchanged and an Rg schedule, checkpoint, journal row and evaluation are bit-identical
    # across every revision of the dose protocol the two-state properties have needed.
    dose_levels=CONTINUOUS_DOSE_LEVELS,
    # D-46: Rg predicts an ABSOLUTE length, in the same frame as its reference table's mean, so
    # level 0 targets that mean and the dose is a multiple of the sequence's own sigma.
    target_origin=TARGET_ORIGIN_SEQUENCE_MEAN,
    two_state=False,
    property_callback_identity="radius_of_gyration_v1",
    # ``v3`` marked the D14 split; ``v4`` pairs with the v2 training-code schema, so a v3
    # identity can never be silently accepted by v2-fingerprinted code or the reverse. It
    # deliberately does NOT move for a code change: the version names the callback FORMAT, and
    # the code change is carried by the training fingerprint this marker is concatenated with.
    generation_callback_version="bioemu_exact_rg:v4",
    loss_class=_loss_class,
    runtime_objective_arguments=_objective_arguments,
    resolve_arm=_resolve_arm,
    adapter_arguments=_no_arm_arguments,
    sample_definition_arguments=_no_arm_arguments,
    checkpoint_arm_arguments=_no_arm_arguments,
    refused_runtime_keys=_REFUSED,
    evaluator=_evaluator,
    evaluator_arm_keyword=None,
    measure_sample=_measure_sample,
    # No selection-metric declaration: radius of gyration's claim IS its target error, so the
    # shared default (``target_mae_sigma`` on the relative path, ``target_mae`` on the absolute
    # one) is the right metric and stays absent from ``adapter_config`` exactly as it always was.
)
