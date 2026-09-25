"""Property-specific, zero-output, RMS-bounded representation adapters."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path

import torch
from torch import nn

from sae_bioemu.core.checkpoint_io import load_sae_checkpoint
from sae_bioemu.core.model import TimeConditionedTopKSAE

from .adapter_config import (
    DEFAULT_RESIDUAL_CAP,
    # IMPORTED, not redeclared. This module used to keep its own copy of the tuple beside
    # adapter_config's. Adding the ``none`` mode to one and not the other made the constructor
    # accept a value its own config layer then rejected, raising from `_approved_architecture`
    # two frames below where the mode had just been validated as fine. One definition, so the
    # two cannot drift again; importing it also keeps it a module attribute here, so existing
    # ``from .adapter import LEVEL_CONDITIONING_MODES`` callers are unaffected.
    LEVEL_CONDITIONING_MODES,
    RESIDUAL_CAP_MODES,
    SITE_CONDITIONING_MODES,
    AdapterArchitecture,
)
from .predictor import _validate_masked_values, _validate_scalar, masked_mean, masked_rms


class TargetKind(str, Enum):
    ABSOLUTE_SEQUENCE = "absolute_sequence"
    DIRECTIONAL_RESIDUE = "directional_residue"
    RELATIVE_SEQUENCE = "relative_sequence"


class AdapterVariant(str, Enum):
    """Supported adapter inputs or output constraints."""

    PRIMARY = "primary"
    LOCAL_TARGET = "local_target"
    SELECTED_SAE_FEATURE = "selected_sae_feature"
    SAE_DECODER_DIRECTION = "sae_decoder_direction"


@dataclass(frozen=True, init=False)
class _SAEDecoderDirectionBinding:
    """An immutable selected SAE decoder column bound to its checkpoint."""

    feature_index: int
    sae_checkpoint_fingerprint: str
    sae_state_fingerprint: str
    direction_fingerprint: str
    _direction: torch.Tensor

    @property
    def direction(self) -> torch.Tensor:
        return self._direction.clone()


def bind_sae_decoder_direction(
    sae_checkpoint: str | Path,
    feature_index: int,
) -> _SAEDecoderDirectionBinding:
    """Load a trusted SAE checkpoint and bind its exact selected decoder column."""
    if not isinstance(sae_checkpoint, str | Path):
        raise TypeError("sae_checkpoint must be a filesystem path")
    path = Path(sae_checkpoint)
    try:
        checkpoint_bytes = path.read_bytes()
        payload = load_sae_checkpoint(io.BytesIO(checkpoint_bytes), map_location="cpu")
        decoder = payload["state_dict"]["decoder.weight"]
        expected_shape = (payload["config"].input_dim, payload["config"].n_latents)
        if not isinstance(decoder, torch.Tensor) or decoder.shape != expected_shape:
            raise ValueError("SAE decoder.weight shape does not match its checkpoint config")
        model = TimeConditionedTopKSAE(payload["config"])
        model.load_state_dict(payload["state_dict"], strict=True)
    except ValueError:
        raise
    except (OSError, KeyError, TypeError, RuntimeError) as error:
        raise ValueError(f"invalid SAE checkpoint at {path}") from error
    decoder_weight = model.decoder.weight.detach().to("cpu")
    if not decoder_weight.is_floating_point() or not torch.isfinite(decoder_weight).all():
        raise ValueError("SAE decoder.weight must be finite and floating")
    if (
        not isinstance(feature_index, int)
        or isinstance(feature_index, bool)
        or not 0 <= feature_index < decoder_weight.shape[1]
    ):
        raise ValueError("feature_index is outside the SAE decoder width")
    column = decoder_weight[:, feature_index].detach().to("cpu").clone()
    norm = torch.linalg.vector_norm(column)
    if norm == 0:
        raise ValueError("selected SAE decoder direction must be nonzero")
    direction = (column / norm).contiguous()
    binding = object.__new__(_SAEDecoderDirectionBinding)
    object.__setattr__(binding, "feature_index", feature_index)
    object.__setattr__(
        binding, "sae_checkpoint_fingerprint", hashlib.sha256(checkpoint_bytes).hexdigest()
    )
    object.__setattr__(binding, "sae_state_fingerprint", _sae_state_fingerprint(model))
    object.__setattr__(binding, "direction_fingerprint", _direction_fingerprint(direction))
    object.__setattr__(binding, "_direction", direction)
    return binding


def _sae_state_fingerprint(model: TimeConditionedTopKSAE) -> str:
    digest = hashlib.sha256()
    config = vars(model.config)
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().to("cpu").contiguous()
        descriptor = (name, str(tensor.dtype), tuple(tensor.shape))
        digest.update(json.dumps(descriptor, separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _direction_fingerprint(direction: torch.Tensor) -> str:
    return hashlib.sha256(direction.contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class AdapterOutput:
    """The representation after steering and its masked residual diagnostics."""

    modified: torch.Tensor
    residual: torch.Tensor
    residual_rms: torch.Tensor
    residual_fraction: torch.Tensor
    #: The fraction the adapter *asked* for, before the tanh cap, ``RMS(raw)/RMS(h)``.
    #: ``residual_fraction`` is this quantity squashed through ``rho_max tanh(.../rho_max)``,
    #: so it is bounded and its gradient dies exponentially once the cap is reached -- at the
    #: 0.09999966 both paired arms trained to, the derivative is attenuated ~47,000x. Any
    #: penalty meant to *hold* the adapter off the cap has to act on the unsquashed value,
    #: which is why this is carried separately rather than recovered by inverting the tanh.
    #:
    #: Under ``residual_cap='none'`` the squashing is gone, so this and ``residual_fraction``
    #: are the same quantity computed twice -- the first in float64, the second in the
    #: activation dtype. It is still carried, and still separately, because a penalty written
    #: against it must keep working in both modes rather than silently reading a different
    #: number depending on the cap.
    raw_residual_fraction: torch.Tensor | None = None


#: How the requested level reaches the hidden state. ``concat`` is the original geometry:
#: the level is appended to the input vector. ``embedding`` adds a dedicated ``Linear(1, H)``
#: to the first hidden pre-activation, so the level owns a parameter whose gradient is not
#: shared with the activation columns. ``film`` has the level emit a per-unit scale and shift
#: applied after every hidden layer; because the scale *multiplies*, the network cannot route
#: around it by driving one input column's weights to zero, which is what ``concat`` permits.
#:
#: ``branch`` keeps the level out of the first layer entirely. All three earlier modes leave the
#: level competing with the site activations on *magnitude*, not on fan-in: the activations have
#: RMS about 167 while the level is a scalar in [-1, 1], so widening the level's share of the
#: first layer barely moves the emitted field (measured: level-dependent emission stayed at
#: 0.01x the level-independent part with the level owning 64 of 128 columns). What fixes it is
#: normalising the activation branch -- a ``LayerNorm`` after the first hidden GELU -- and then
#: concatenating a separate ``Linear(1, H)`` level branch, so the two arrive at the second layer
#: on comparable scales (measured dose-to-nuisance ratio 0.47 at ``level_gain`` 1, 2.18 at 3).
#: The level branch is deliberately *linear*: putting a GELU on it, or driving it hard, rectifies
#: it. ``cos(odd)`` between ``r(+1) - r(0)`` and ``-(r(-1) - r(0))`` fell from +0.78 at gain 1 to
#: -0.99 at gain 10, i.e. past gain ~3 the response is even and +1 and -1 push the same way. The
#: usable ``level_gain`` window is therefore about 1 to 3.
#:
#: ``none`` is the UNCONDITIONED mode (D-101): the level reaches nothing and the adapter's only
#: input is the hidden activations, so one arm is trained PER DIRECTION. It exists because every
#: mode above measured EVEN in the level once TRAINED -- ``||odd||/||even||`` 0.0133 on cryptic at
#: epoch 37 and 0.0099 on domain motion at epoch 25, with ``cos(r(+1), -r(-1))`` at -0.9998 and
#: -0.9999, i.e. ~1% of the response is directional. Note those are the trained figures; the
#: ``cos(odd)`` numbers quoted above for the warm start are step-0 measurements, and training went
#: FURTHER into that basin rather than out of it. Forcing oddness by multiplying the residual by
#: the level is what ``TargetKind.DIRECTIONAL_RESIDUE`` does and what D2 forbids on
#: ``RELATIVE_SEQUENCE``, so removing the input is the remaining route: with no level there is
#: nothing to symmetrise and no even component can form.
#:
#: The tuple itself lives in :mod:`sae_bioemu.steering.adapter_config` and is imported above.


def _standardized_site_field(site_field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return ``s`` centred and scaled to unit deviation over each sequence's valid residues.

    **Why the channel is normalised at all.** Arms A_binary, A_smooth and B differ only in what
    fills ``s``: a ``{0, 1}`` indicator, a Gaussian field in ``(0, 1]``, and PocketMiner's
    probability in ``[0, 1]``. Those three have different means and different spreads -- a mask
    over a 10% pocket has deviation ``sqrt(0.09) = 0.30``, a smoothed field over the same pocket
    considerably less, and a diffuse probability less again. Fed raw, the site branch would see
    three different *gains* as well as three different shapes, and ``A_smooth`` would not be the
    sharpness-matched control for ``B`` that D-21 REVISED makes it: an A-vs-B difference would
    confound mask quality with channel scale. After this, all three arrive with mean 0 and
    deviation 1 and differ only in shape, which is the comparison the arms are for.

    **Why this arithmetic and not another.** It is the arithmetic ``adapter.py`` already uses to
    put two pathways on comparable scales. ``level_conditioning='branch'`` exists because the
    activations (RMS about 167) out-scaled the level (a scalar in ``[-1, 1]``) by two orders of
    magnitude and the level was ignored; the fix was ``activation_norm``, a ``LayerNorm`` that
    centres and scales the activation pathway before the layer that mixes the two. This is that
    LayerNorm's own computation -- subtract the mean, divide by the biased (``/n``) standard
    deviation -- applied on the only axis a scalar channel varies along, the residue axis, and
    restricted to valid residues by the same mask ``masked_mean`` and ``masked_rms`` already use.

    Putting an ``nn.LayerNorm(H)`` on the site branch's *output* instead would be the wrong
    reading of the same precedent, and it is worth stating why rather than leaving it to be
    rediscovered. That output is ``w s + b`` with ``w, b`` in ``R^H``, a rank-one function of a
    scalar, so LayerNorm over ``H`` gives ``(w s + b - mean(w s + b)) / std(w s + b)``, which
    tends to ``+-(w - mean(w)) / std(w)`` as ``|s|`` grows: the dose collapses onto its sign and
    a probability of 0.55 becomes indistinguishable from one of 0.99. ``activation_norm``
    normalises an ``H``-dimensional pathway with ``H`` independent coordinates; the site channel
    has one, and normalising it along ``H`` destroys it.

    Fails closed on a channel that is constant over a sequence's valid residues. Returning zeros
    there would silently convert a conditioned arm into arm C for that protein -- a null result
    reported as a conditioned one -- and for arms A it cannot happen at all, since the pocket is
    a nonempty proper subset of the residues.
    """
    values = site_field.unsqueeze(-1)
    valid = mask.unsqueeze(-1)
    mean = masked_mean(values, mask)
    centred = torch.where(valid, values - mean.unsqueeze(1), torch.zeros_like(values))
    deviation = masked_rms(centred, mask)
    if not bool(torch.all(deviation > 0)):
        raise ValueError(
            "site_conditioning field is constant over a sequence's valid residues, so it "
            "carries no site information and cannot be standardized; a zero channel here "
            "would silently run a conditioned arm as the unconditioned arm C"
        )
    return centred / deviation[:, None, None]


def _tanhc_from_squared(value: torch.Tensor) -> torch.Tensor:
    """Return ``tanh(sqrt(value))/sqrt(value)`` with its safe zero limit."""
    threshold = value.new_tensor(1e-4)
    series_value = torch.minimum(value, threshold)
    series = 1.0 - series_value / 3.0 + 2.0 * series_value.square() / 15.0
    safe_value = torch.where(value > threshold, value, torch.ones_like(value))
    root = torch.sqrt(safe_value)
    direct = torch.tanh(root) / root
    return torch.where(value <= threshold, series, direct)


def _warm_start_fingerprint(fingerprint: object, scale: float) -> str:
    """Identify a warm start by its direction **and** its gain.

    The scale multiplies the direction into the weights, so it changes the trained function --
    two arms at gain 16 and 40 are different experiments. Recording only the direction file
    made their adapter attribute blocks byte-identical, so the ``s16`` checkpoint could be
    evaluated under the ``s40`` config with the gate passing and the report naming the wrong
    gain. This is the same defect the objective identity had for ``lambda_l``.

    Formatted with ``repr`` so 16 and 16.0 are one value rather than two.
    """
    return f"{fingerprint}@scale={float(scale)!r}"


class BoundedResidualAdapter(nn.Module):
    """A property/model-specific per-residue MLP with a smooth RMS cap.

    The primary input at residue ``i`` is ``[h_i, mean_masked(h), y*]``. Only
    the final projection is initialized to zero, so the initial intervention is
    exactly the identity while hidden layers retain ordinary initialization.

    ``TargetKind.RELATIVE_SEQUENCE`` replaces the standardized absolute value
    ``y*`` with the requested level ``o in [-1, 1]``, one scalar per sequence
    broadcast to every residue (design decision D1). Unlike
    ``TargetKind.DIRECTIONAL_RESIDUE`` the output is deliberately **not** gated
    by the request (D2): gating would make dose proportionality architectural,
    and ``mean_monotonicity`` must be learned rather than constructed. Level 0
    is therefore not a structural no-op; the null control is the adapter-off arm.

    ``site_conditioning`` adds the cryptic-pocket arms' per-residue channel ``s_i``, telling
    the adapter which residues the pocket is. It is a genuine architecture change -- the second
    hidden layer's fan-in goes ``2H -> 3H`` -- rather than an extra input column, which is why
    it is gated and has no default: an adapter built without it cannot load a checkpoint
    trained with it, and vice versa, and the shape mismatch is what makes that audible.
    Arm C (``RELATIVE_SEQUENCE`` + ``branch``, no site channel) is the experiment and is
    completely unaffected; arms A and B are the ceiling and the deployable interpolation.

    ``residual_cap`` chooses whether there is a cap at all.

    ``tanh``, the DEFAULT and every adapter ever trained, realises
    ``residual_fraction = rho_max * tanh(r)`` for the pre-cap ratio
    ``r = RMS_m(raw) / (rho_max * RMS_m(h))``. ``none`` emits ``residual = raw`` and reports
    ``residual_fraction = RMS_m(raw) / RMS_m(h)``, unbounded.

    **Why an off switch exists, and why it is a mode rather than a larger number.** The squash
    is asymptotic, so its slope dies with its headroom: ``d/dr [rho_max tanh(r)] = rho_max
    sech^2(r)``, which at the realised fraction ``0.09999966`` against ``rho_max = 0.10`` is
    attenuated about 47,000x relative to the small-``r`` slope. Two arms that inherited
    ``rho_max: 0.10`` from a different family sat exactly there and flat-lined: the objective
    still had a gradient with respect to the *dose*, and essentially none of it reached the
    parameters. Raising ``rho_max`` moves the shelf; it does not remove it, and a run that
    saturates a larger cap is in the identical state one ``rho_max`` higher. Removing the cap
    is a different experiment, so it is an identity-bearing mode and never a number.

    **``rho_max`` under ``none``: kept, required, and explicitly no longer a bound.** It is not
    refused, because it is not inert -- ``ResidualPenaltyKind.FRACTION_OF_RHO_MAX`` and
    ``RAW_FRACTION_OF_RHO_MAX`` divide by it to put ``lambda_r`` on a scale-free footing, and
    :class:`~sae_bioemu.steering.fixed_residual_probe.FixedResidualProbeAdapter` parametrises
    its requested dose in units of it. What it stops being is the thing that bounds the
    intervention, and that change is recorded rather than left to be inferred: the mode is in
    the checkpoint's adapter identity, and the BioEmu run report carries a ``residual_cap``
    block naming the mode, the ``rho_max`` value, and whether it bounded anything. The one
    thing that *does* bound an uncapped run is
    ``bioemu_pytorch.validation_residual_fraction_maximum``, which
    :mod:`sae_bioemu.bioemu.steering_workflow` therefore REQUIRES under this mode instead of
    treating as optional.

    There is deliberately no clamp, no soft floor and no magnitude warning on the ``none``
    path: any of those would be the cap again under another name, and the point of the mode is
    that the bound is the configured gate and nothing else. What it does refuse is a value that
    is not a number at all -- a non-finite pre-cap field (already refused by ``masked_rms``,
    which validates its own input), a sequence with zero masked activation RMS, and a steered
    activation that overflows the activation dtype. Each fails closed with a message rather than
    handing NaN to the score model.
    """

    def __init__(
        self,
        width: int,
        property_name: str,
        model_family: str,
        *,
        hidden_width: int = 64,
        level_conditioning: str = "concat",
        level_gain: float = 1.0,
        site_conditioning: str | None = None,
        hidden_layers: int = 2,
        rho_max: float = 0.1,
        residual_cap: str = DEFAULT_RESIDUAL_CAP,
        eps: float = 1e-8,
        variant: AdapterVariant | str = AdapterVariant.PRIMARY,
        target_kind: TargetKind | str = TargetKind.ABSOLUTE_SEQUENCE,
        selected_feature_index: int | None = None,
        sae_checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (width, hidden_width, hidden_layers)
        ):
            raise ValueError("width, hidden_width, and hidden_layers must be positive integers")
        if not isinstance(property_name, str) or not property_name.strip():
            raise ValueError("property_name must be non-empty")
        family_value = getattr(model_family, "value", model_family)
        if family_value not in ("bioemu", "alphafold3"):
            raise ValueError("model_family must be bioemu or alphafold3")
        if isinstance(rho_max, bool) or not isinstance(rho_max, Real) or not 0 < rho_max <= 1:
            raise ValueError("rho_max must be in (0, 1]")
        if residual_cap not in RESIDUAL_CAP_MODES:
            raise ValueError(f"residual_cap must be one of {RESIDUAL_CAP_MODES}")
        if isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        try:
            variant = AdapterVariant(variant)
        except ValueError as error:
            raise ValueError("unsupported adapter variant") from error
        try:
            target_kind = TargetKind(target_kind)
        except ValueError as error:
            raise ValueError("unsupported target kind") from error
        if variant is AdapterVariant.SAE_DECODER_DIRECTION:
            if sae_checkpoint_path is None:
                raise ValueError("decoder-direction adapters require an SAE checkpoint path")
            if selected_feature_index is None:
                raise ValueError("selected_feature_index is required for SAE feature ablations")
            decoder_binding = bind_sae_decoder_direction(
                sae_checkpoint_path, selected_feature_index
            )
        else:
            if sae_checkpoint_path is not None:
                raise ValueError("SAE checkpoint path is valid only for decoder-direction adapters")
            decoder_binding = None
        feature_variant = variant in (
            AdapterVariant.SELECTED_SAE_FEATURE,
            AdapterVariant.SAE_DECODER_DIRECTION,
        )
        if feature_variant != (selected_feature_index is not None):
            raise ValueError("selected_feature_index is required only for SAE feature ablations")
        if selected_feature_index is not None and (
            not isinstance(selected_feature_index, int)
            or isinstance(selected_feature_index, bool)
            or selected_feature_index < 0
        ):
            raise ValueError("selected_feature_index must be a nonnegative integer")
        direction_variant = variant is AdapterVariant.SAE_DECODER_DIRECTION
        if decoder_binding is not None:
            direction = decoder_binding.direction
            if direction.shape != (width,):
                raise ValueError("bound SAE decoder direction width does not match adapter")
            if _direction_fingerprint(direction) != decoder_binding.direction_fingerprint:
                raise ValueError("bound SAE decoder direction fingerprint is invalid")
        else:
            direction = None

        if level_conditioning not in LEVEL_CONDITIONING_MODES:
            raise ValueError(f"level_conditioning must be one of {LEVEL_CONDITIONING_MODES}")
        branch = level_conditioning == "branch"
        #: ``none``: the level reaches nothing. Kept as a separate local from ``branch`` because
        #: the two agree only on "no level column in the first layer" and disagree on everything
        #: else -- ``branch`` builds a level pathway and a LayerNorm, ``none`` builds neither.
        unconditioned = level_conditioning == "none"
        if (
            isinstance(level_gain, bool)
            or not isinstance(level_gain, Real)
            or not math.isfinite(level_gain)
            or level_gain <= 0
        ):
            raise ValueError("level_gain must be positive and finite")
        if not branch and float(level_gain) != 1.0:
            # A knob that is accepted, stored, recorded and never applied is exactly the defect
            # ``lambda_l`` had, so a non-default gain outside its one mode is an error, not a
            # silently ignored value.
            raise ValueError("level_gain applies only to level_conditioning='branch'")
        if branch and hidden_layers < 2:
            raise ValueError("level_conditioning='branch' needs at least two hidden layers")
        if site_conditioning is not None:
            if site_conditioning not in SITE_CONDITIONING_MODES:
                raise ValueError(f"site_conditioning must be one of {SITE_CONDITIONING_MODES}")
            if not branch:
                raise ValueError(
                    "site_conditioning requires level_conditioning='branch': the site channel "
                    "joins at the second hidden layer, and no other mode has that join"
                )

        self.width = width
        self.property_name = property_name
        self.model_family = str(family_value)
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.level_conditioning = level_conditioning
        self.level_gain = float(level_gain)
        self.site_conditioning = site_conditioning
        self.architecture = _approved_architecture(
            width,
            hidden_width,
            hidden_layers,
            float(rho_max),
            variant,
            target_kind,
            level_conditioning,
            site_conditioning,
            str(residual_cap),
        )
        self.rho_max = float(rho_max)
        #: ``tanh`` or ``none``; see the class docstring. Identity-bearing: recorded in the
        #: checkpoint's adapter attributes (only when it is not ``tanh``, so no existing
        #: checkpoint's key set moves) and in the portable architecture payload.
        self.residual_cap = str(residual_cap)
        self.eps = float(eps)
        self.variant = variant.value
        self.target_kind = target_kind.value
        self.selected_feature_index = selected_feature_index
        self.sae_checkpoint_fingerprint = (
            None if decoder_binding is None else decoder_binding.sae_checkpoint_fingerprint
        )
        self.sae_state_fingerprint = (
            None if decoder_binding is None else decoder_binding.sae_state_fingerprint
        )
        self.decoder_direction_fingerprint = (
            None if decoder_binding is None else decoder_binding.direction_fingerprint
        )
        # ``branch`` keeps the level out of the first layer, so the level column is absent from
        # the concatenated input vector: [h_i, mean(h)] rather than [h_i, mean(h), o]. ``none``
        # drops that column for the opposite reason -- there is no level anywhere to put in it --
        # and the two therefore share this width while sharing nothing else.
        self.input_width = width if (branch or unconditioned) else width + 1
        if variant is not AdapterVariant.LOCAL_TARGET:
            self.input_width += width
        if variant is AdapterVariant.SELECTED_SAE_FEATURE:
            self.input_width += 1
        output_width = 1 if direction_variant else width

        hidden: list[nn.Module] = []
        in_features = self.input_width
        for index in range(hidden_layers):
            if branch and index == 1:
                # The second layer consumes [normed activation state, level branch], and for
                # cryptic arms A/B the site branch as well: 2H -> 3H. This is the whole
                # architecture change those arms are, and it is why ``site_conditioning`` is
                # gated rather than defaulted -- an adapter built without it cannot load a
                # checkpoint trained with it, and the shape mismatch is what says so.
                in_features = (3 if site_conditioning is not None else 2) * hidden_width
            hidden.extend((nn.Linear(in_features, hidden_width), nn.GELU()))
            in_features = hidden_width
        self.hidden = nn.Sequential(*hidden)
        # The requested level is one column of a ``(hidden_width, input_width)`` matrix, so
        # at width 512 it carries ~1/31 of the first layer's weight norm and its gradient is
        # diluted by the 1024 activation columns sharing that layer. Measured consequence:
        # the level-dependent part of the residual is ~0.2% of the residual magnitude and
        # *shrinks* with training, which is the adapter learning a large level-INDEPENDENT
        # intervention. These modes give the level an undiluted pathway; ``concat`` keeps the
        # original geometry exactly, including parameter names, so it stays the default.
        self.level_embedding = (
            nn.Linear(1, hidden_width) if level_conditioning == "embedding" else None
        )
        self.level_film = nn.Linear(1, 2 * hidden_width) if level_conditioning == "film" else None
        # ``None`` in every other mode, so ``concat``/``embedding``/``film`` stay numerically
        # identical to the pre-existing adapter down to the parameter names.
        # THE NORM IS ABOUT THE ACTIVATION PATHWAY, NOT THE LEVEL, so `none` keeps it (D-101).
        # `branch` introduced it because the activations arrive at RMS ~167 against a level in
        # [-1, 1]; that motivation is about SCALE, and removing the level does not remove the
        # scale. Building `none` without it -- which is what `if branch else None` did -- left the
        # hidden stack unnormalised, and MEASURED on domain_motion_uncond_m1: geometry_loss
        # spiked to 6.27 against a median of 0.00097, total_loss to 126 against 0.385, and
        # direct_metric_angstrom to 36.5 Angstrom, i.e. occasional draws with broken backbones.
        # Nine of them tripped the peptide-bond gate ("median C(i)-N(i+1) distance of 1.702,
        # outside (1.0, 1.7)") in 37 minutes and the run gave up; the level-conditioned parent
        # produced ZERO such refusals in 21 hours. lambda_g = 20 prices the damage after the fact
        # but cannot stop the draw being generated and refused.
        self.activation_norm = nn.LayerNorm(hidden_width) if (branch or unconditioned) else None
        self.level_branch = nn.Linear(1, hidden_width) if branch else None
        # The site channel's pathway, built exactly like the level's and for the same measured
        # reason: a scalar concatenated into the first layer beside 2W activation columns of RMS
        # about 167 is diluted to nothing (measured on the level: level-dependent emission stayed
        # at 0.01x the level-independent part even with the level owning 64 of 128 columns).
        #
        # Deliberately linear and at unit gain. Linear because a GELU rectifies the channel, which
        # is the failure ``branch`` exists to avoid; unit gain because ``level_gain`` earned its
        # knob from a measurement (the usable window 1-3, past which the response goes even) and
        # the site channel has no such measurement yet. A gain accepted, recorded and unjustified
        # is the ``lambda_l`` hole in a new place. Standardising ``s`` is what makes unit gain the
        # right default: the channel arrives at deviation 1, the activation pathway arrives
        # LayerNormed, and the level branch reads a scalar in [-1, 1], so all three enter the
        # second layer on the same scale without a tuning constant.
        self.site_branch = nn.Linear(1, hidden_width) if site_conditioning is not None else None
        if self.level_branch is not None:
            with torch.no_grad():
                # No activation is applied to this branch's output anywhere: a GELU on the level
                # rectifies it, which is the failure mode ``branch`` exists to avoid.
                self.level_branch.weight.mul_(self.level_gain)
                self.level_branch.bias.zero_()
        if self.level_film is not None:
            # Start as the identity modulation, gamma = 1 and beta = 0, so an untrained FiLM
            # adapter is exactly the concat adapter and the comparison is not confounded by a
            # different starting function.
            nn.init.zeros_(self.level_film.weight)
            with torch.no_grad():
                self.level_film.bias[:hidden_width].fill_(1.0)
                self.level_film.bias[hidden_width:].zero_()
        self.output = nn.Linear(hidden_width, output_width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("decoder_direction", direction)
        #: Set by :meth:`warm_start_from_direction`; ``None`` means the ordinary zero start.
        #: Recorded in the checkpoint's adapter identity, so a warm-started run and a
        #: cold-started one are never interchangeable.
        self.warm_start_direction_fingerprint: str | None = None

    def warm_start_dose_pathway(
        self, direction: torch.Tensor, *, scale: float, fingerprint: str
    ) -> None:
        """Start the adapter emitting ``direction * level``, off the initialisation saddle.

        The output layer is zero-initialised, so at step zero ``dr/do = W_out . d(hidden)/do``
        is zero *and* every parameter upstream of ``W_out`` -- including the level column, the
        ``embedding`` projection and the ``film`` modulator -- receives a gradient of exactly
        ``0``, because that gradient is proportional to ``W_out``. Measured on all three
        conditioning modes: ``|grad level column| = 0.000e+00``. Level-dependence therefore
        cannot begin to grow until ``W_out`` has already been shaped by the level-*independent*
        error, and :meth:`warm_start_from_direction` does not help because a bias does not
        multiply the hidden layer, leaving ``dr/do`` at zero.

        This writes a dose response directly. ``GELU(x) - GELU(-x) = x Phi(x) + x Phi(-x) = x``
        exactly, so one pair of hidden units per layer carries the level through untouched::

            layer 1   u0 = GELU(+s o),   u1 = GELU(-s o)     -> u0 - u1 = s o
            layer k   v0 = GELU(+t . ),  v1 = GELU(-t . )    -> v0 - v1 = t s o
            output    r  = (unit / (t s)) (v0 - v1)          = unit o

        Unlike a gated architecture, nothing here *forces* proportionality: these are ordinary
        weights and training is free to remove them. That is the point -- it converts "can the
        optimiser reach a dose response" into "does the objective keep one", which D2's ban on
        architectural gating leaves open.
        """
        self._write_dose_pathway(direction, scale=scale)
        self.warm_start_direction_fingerprint = _warm_start_fingerprint(fingerprint, scale)

    def warm_start_rank_one(
        self, direction: torch.Tensor, *, scale: float, fingerprint: str
    ) -> None:
        """Break the initialisation saddle by writing the direction prior into ``W_out``.

        :meth:`warm_start_dose_pathway` also leaves the saddle, but it does so by hand-building a
        ``GELU(x) - GELU(-x) = x`` circuit into two hidden units, which zeroes every other row of
        every hidden layer. Measured at both ``o = +1`` and ``o = -1``: **1 of 64 rows per hidden
        layer had any gradient**. The network then has to regrow the 63 dead rows from nothing.

        This writes a rank-one output layer instead::

            v = ones(H) / sqrt(H)        u = direction / ||direction||
            W_out = scale * outer(u, v)  b_out = 0

        Because ``dr/d(hidden_j) = scale * v_j * u`` and every ``v_j`` is nonzero, every hidden
        unit is on a gradient path from step 0 -- no unit is zeroed. ``v`` is deliberately the
        uniform vector rather than a basis vector: a basis vector would put the whole output on
        one hidden unit and recreate exactly the dead-unit problem this exists to remove. It is
        also deterministic, so the start is reproducible from the direction and scale alone.

        Unlike the dose pathway this carries **no** level dependence at initialisation (the level
        still has to be learned, per D2); what it changes is that the gradient which grows level
        dependence is no longer identically zero.

        Caveat, measured, on the ``branch`` mode at ``scale = 1e-2``: the *uniform* ``v`` makes
        the initial response **even** in the level. ``GELU(x) + GELU(-x) = x (2 Phi(x) - 1) >= 0``
        for every ``x``, so an all-positive readout sums 64 non-negative even parts coherently
        (~H) while the odd parts, whose signs come from the random second-layer weights, cancel
        as a random walk (~sqrt(H)) -- an ~8x bias toward the even component. ``cos(odd)`` between
        ``r(+1) - r(0)`` and ``-(r(-1) - r(0))`` is -0.74 at ``level_gain`` 1 and -0.97 at 3, even
        though the hidden state feeding the readout is still odd (+0.85 and +0.40) and a
        random-sign ``v`` reads +0.98 and +0.93 out of those same adapters. A deterministic
        alternating-sign ``v`` -- same uniform magnitude on every unit, same 64/64 gradient rows --
        measures +0.96 and +0.85 instead. This start is therefore not odd at step 0; it is a
        saddle-breaking start, and the level's sign structure is left to training.
        """
        self._write_rank_one(direction, scale=scale)
        # The mode is part of the identity: a rank-one start and a dose start from the same
        # direction file at the same gain are different initialisations and different functions,
        # so they must not collide on one fingerprint.
        self.warm_start_direction_fingerprint = _warm_start_fingerprint(
            f"rank_one:{fingerprint}", scale
        )

    def _write_rank_one(self, direction: torch.Tensor, *, scale: float) -> None:
        if not isinstance(direction, torch.Tensor) or direction.shape != self.output.bias.shape:
            raise ValueError(
                "the rank-one direction must be a tensor shaped like the adapter's output bias "
                f"{tuple(self.output.bias.shape)}"
            )
        if not torch.isfinite(direction).all():
            raise ValueError("the rank-one direction must be finite")
        precise = direction.to(torch.float64)
        norm = torch.linalg.vector_norm(precise)
        if float(norm) == 0.0:
            raise ValueError("the rank-one direction must be non-zero")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("the rank-one scale must be positive and finite")
        with torch.no_grad():
            unit = precise / norm
            # Alternating signs, NOT a uniform vector. GELU(x) + GELU(-x) = x(2*Phi(x) - 1)
            # is non-negative for every x, so the even-in-o part of every hidden unit has the
            # same sign. An all-positive readout sums those 64 even parts coherently (~H)
            # while the odd parts, whose signs come from the random second-layer weights,
            # cancel as a random walk (~sqrt(H)) -- an ~8x structural bias toward an EVEN
            # response. Measured at gain 1: cos(odd) = -0.744 with a uniform readout and
            # +0.961 with this one, i.e. the uniform vector inverts the very property the
            # warm start exists to supply. Alternating preserves the reason for spreading
            # (deterministic, |v_j| identical on every unit, so no unit is dropped) without it.
            spread = torch.full((self.hidden_width,), self.hidden_width**-0.5, dtype=torch.float64)
            spread[1::2] *= -1.0
            self.output.weight.copy_(
                (torch.outer(unit, spread) * float(scale)).to(self.output.weight.dtype)
            )
            self.output.bias.zero_()

    def _write_dose_pathway(self, direction: torch.Tensor, *, scale: float) -> None:
        if self.level_conditioning == "branch":
            # The construction writes the level into ``linears[0].weight[:, -1]``, and in
            # ``branch`` mode that column is an activation column, not the level.
            raise ValueError("the dose pathway is incompatible with level_conditioning='branch'")
        if not isinstance(direction, torch.Tensor) or direction.shape != self.output.bias.shape:
            raise ValueError(
                "the dose direction must be a tensor shaped like the adapter's output bias "
                f"{tuple(self.output.bias.shape)}"
            )
        if not torch.isfinite(direction).all():
            raise ValueError("the dose direction must be finite")
        root_mean_square = direction.to(torch.float64).square().mean().sqrt()
        if float(root_mean_square) == 0.0:
            raise ValueError("the dose direction must be non-zero")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("the dose scale must be positive and finite")
        linears = [layer for layer in self.hidden if isinstance(layer, nn.Linear)]
        if not linears or self.hidden_width < 2:
            raise ValueError("the dose pathway needs at least one hidden layer of width 2")
        # The gain is chosen so the output columns end up with unit norm, which matters far
        # more than it looks. ``GELU(x) - GELU(-x) = x`` holds for every ``x``, so any gain is
        # exact on paper; but the gain also fixes the output weights at ``scale / gain``, and
        # those multiply *whatever* the dose units carry. With a small gain the output columns
        # reach norm ~90-226, the dose units acquire a little activation weight, and the site
        # activations -- RMS about 167 here -- push ``w . h`` far past the level drive. Measured
        # at gain 4 after 745 units: the emitted field went from exactly odd in ``o`` to
        # ``cos(r(+1), -r(-1)) = -0.93``, i.e. level-independent, with ``||r(o=0)||`` at 2623
        # instead of 0. A unit-norm output column keeps the level drive dominant and stops the
        # construction amplifying stray hidden activity.
        first_gain = float(scale) * float(self.width) ** 0.5
        later_gain = 1.0
        with torch.no_grad():
            for layer in linears:
                layer.weight.zero_()
                layer.bias.zero_()
            # The level is the final input column: [h_i, mean(h), o].
            linears[0].weight[0, -1] = first_gain
            linears[0].weight[1, -1] = -first_gain
            for layer in linears[1:]:
                layer.weight[0, 0] = later_gain
                layer.weight[0, 1] = -later_gain
                layer.weight[1, 0] = -later_gain
                layer.weight[1, 1] = later_gain
            product = first_gain * (later_gain ** (len(linears) - 1))
            unit = (direction.to(torch.float64) / root_mean_square).to(self.output.weight.dtype)
            self.output.weight.zero_()
            self.output.bias.zero_()
            self.output.weight[:, 0] = unit * (scale / product)
            self.output.weight[:, 1] = -unit * (scale / product)

    def warm_start_from_direction(
        self, direction: torch.Tensor, *, scale: float, fingerprint: str
    ) -> None:
        """Start the adapter emitting a known Rg direction instead of nothing.

        The output layer is zero-initialised in both weight and bias, so the adapter begins as
        an exact no-op and has to discover a useful direction by descent. The oracle probe
        showed that a direction clearing the requested dose by 4.5x exists and that descent
        does not find it; this asks the separate question of whether descent can *hold* one.

        Only the bias moves. The weight stays zero, so at initialisation the emitted field is
        the same 512-vector at every residue and carries **no** level dependence -- that must
        still be learned through the weight, exactly as before. Writing the level into the
        initialisation instead would gate dose on the architecture, which D2 forbids on the
        relative path because it hollows out ``mean_monotonicity``.

        ``scale`` is the bias RMS. The cap is relative to the activation's own RMS, which is
        not known until generation, so the realised initial fraction has to be read back from
        the first journal rows rather than set exactly here.
        """
        if not isinstance(direction, torch.Tensor) or direction.shape != self.output.bias.shape:
            raise ValueError(
                "the warm-start direction must be a tensor shaped like the adapter's output "
                f"bias {tuple(self.output.bias.shape)}"
            )
        if not torch.isfinite(direction).all():
            raise ValueError("the warm-start direction must be finite")
        root_mean_square = direction.to(torch.float64).square().mean().sqrt()
        if float(root_mean_square) == 0.0:
            raise ValueError("the warm-start direction must be non-zero")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("the warm-start scale must be positive and finite")
        with torch.no_grad():
            unit = (direction.to(torch.float64) / root_mean_square).to(self.output.bias.dtype)
            self.output.bias.copy_(unit * scale)
        self.warm_start_direction_fingerprint = _warm_start_fingerprint(fingerprint, scale)

    def _conditioned_hidden(
        self,
        features: torch.Tensor,
        level: torch.Tensor,
        site: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the hidden stack, injecting the level on its own pathway when configured.

        ``concat`` walks ``self.hidden`` exactly as ``nn.Sequential`` would, so it stays
        numerically identical to the pre-existing adapter.
        """
        if self.level_conditioning == "concat":
            return self.hidden(features)
        if self.level_conditioning == "none":
            # An unconditioned adapter is a plain stack over the activations -- ``level`` is
            # accepted and deliberately unused, so the schedule still records which direction the
            # arm was trained for even though the network cannot see it -- but it keeps
            # ``branch``'s LayerNorm on the activation pathway, applied at the SAME seam: after
            # the first hidden GELU, before the rest of the stack. See the constructor for the
            # measured reason.
            if self.activation_norm is None:  # pragma: no cover - constructor invariant.
                raise RuntimeError("unconditioned conditioning is missing its activation norm")
            state = self.activation_norm(self.hidden[1](self.hidden[0](features)))
            for module in self.hidden[2:]:
                state = module(state)
            return state
        if self.level_branch is not None:
            # ``branch``: normalise the activation pathway so it stops out-scaling the level by
            # two orders of magnitude, then concatenate an unrectified level branch and let the
            # second layer mix them. ``features`` here is [h_i, mean(h)] with no level column.
            if self.activation_norm is None:  # pragma: no cover - constructor invariant.
                raise RuntimeError("branch conditioning is missing its activation norm")
            state = self.activation_norm(self.hidden[1](self.hidden[0](features)))
            branches = [state, self.level_branch(level)]
            if self.site_branch is not None:
                if site is None:  # pragma: no cover - forward validates this first.
                    raise RuntimeError("site conditioning is missing its standardized field")
                # Third branch, hence 2H -> 3H. The join is at the same layer as the level's,
                # so the site channel is never diluted by the activation columns and the two
                # conditioning inputs are mixed together rather than in sequence.
                branches.append(self.site_branch(site))
            state = torch.cat(branches, dim=-1)
            for module in self.hidden[2:]:
                state = module(state)
            return state
        # ``level`` arrives shaped like the concatenated column, [batch, residues, 1].
        state = features
        linear_index = 0
        for module in self.hidden:
            state = module(state)
            if not isinstance(module, nn.Linear):
                continue
            if self.level_embedding is not None and linear_index == 0:
                state = state + self.level_embedding(level)
            linear_index += 1
        if self.level_film is not None:
            # Applied once after the stack rather than per layer: a single multiplicative
            # gate is enough to make the level non-ignorable, and per-layer modulation would
            # add hidden_layers x 2H parameters for no tested benefit.
            gamma, beta = self.level_film(level).chunk(2, dim=-1)
            state = gamma * state + beta
        return state

    def _site_values(
        self,
        site_field: torch.Tensor | None,
        mask: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor | None:
        """Validate and standardize the per-residue site channel, or refuse it.

        Fails closed in **both** directions, which is the shape every gated input in this
        repository has had to be given eventually. A conditioned adapter handed no field is an
        arm A/B checkpoint silently generating as arm C, and the two would be reported as the
        same experiment; an unconditioned adapter handed a field would compute it, drop it, and
        record nothing about the mask it was given. Both are errors here.

        The range check is per source, so a mislabelled arm is refused rather than trained. A
        strict indicator, a Gaussian field and a probability are all in ``[0, 1]``, but only the
        first takes exactly two values, and only the first two are pinned to 1 at the annotated
        residues. Without that, ``pocket_smoothed`` and ``predicted_probability`` would validate
        identically and an arm B field could be trained under an ``A_smooth`` identity -- which
        is exactly the confound D-21 REVISED sets ``A_smooth`` up to remove.
        """
        if self.site_conditioning is None:
            if site_field is not None:
                raise ValueError(
                    "site_field is only valid for an adapter declaring site_conditioning; this "
                    "adapter has no site channel and would silently ignore it"
                )
            return None
        if (
            not isinstance(site_field, torch.Tensor)
            or site_field.shape != mask.shape
            or not site_field.is_floating_point()
            or site_field.device != activations.device
            or site_field.dtype != activations.dtype
        ):
            raise ValueError(
                f"site_conditioning={self.site_conditioning!r} requires a floating "
                "[batch, residues] site_field matching the activations"
            )
        observed = site_field[mask]
        if not torch.isfinite(observed).all():
            raise ValueError("site_field must be finite on valid residues")
        if torch.any(site_field[~mask] != 0):
            raise ValueError("site_field padding must be zero")
        if torch.any((observed < 0) | (observed > 1)):
            raise ValueError("site_field must lie in [0, 1] on valid residues")
        if self.site_conditioning == "pocket_binary" and torch.any(
            (observed != 0) & (observed != 1)
        ):
            raise ValueError(
                "site_conditioning='pocket_binary' requires a strict {0, 1} indicator; a soft "
                "field under this name would be A_smooth wearing A_binary's identity"
            )
        if self.site_conditioning in ("pocket_binary", "pocket_smoothed") and not bool(
            torch.all(site_field.masked_fill(~mask, 0.0).amax(dim=1) == 1)
        ):
            # Both A variants come from the SAME annotation, and both are 1 exactly at an
            # annotated residue: ``exp(-0 / 2 sigma^2) == 1``. A predicted probability is not,
            # which is what separates the arm B channel from its sharpness-matched control.
            raise ValueError(
                f"site_conditioning={self.site_conditioning!r} requires the annotated residues "
                "to carry exactly 1.0; a field that never reaches 1 is a prediction, not the "
                "annotation this arm claims"
            )
        return _standardized_site_field(site_field, mask)

    @classmethod
    def from_sae_decoder_direction(
        cls,
        width: int,
        property_name: str,
        model_family: str,
        *,
        sae_checkpoint_path: str | Path,
        selected_feature_index: int,
        hidden_width: int = 64,
        hidden_layers: int = 2,
        rho_max: float = 0.1,
        eps: float = 1e-8,
    ) -> BoundedResidualAdapter:
        """Construct the constrained ablation from a verified SAE checkpoint."""
        return cls(
            width,
            property_name,
            model_family,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            rho_max=rho_max,
            eps=eps,
            variant=AdapterVariant.SAE_DECODER_DIRECTION,
            sae_checkpoint_path=sae_checkpoint_path,
            selected_feature_index=selected_feature_index,
        )

    def architecture_payload(self) -> dict[str, object] | None:
        """Serialize the portable shared architecture when this adapter is canonical."""
        return None if self.architecture is None else self.architecture.payload()

    def forward(
        self,
        activations: torch.Tensor,
        standardized_target: torch.Tensor,
        mask: torch.Tensor,
        *,
        selected_sae_activation: torch.Tensor | None = None,
        site_field: torch.Tensor | None = None,
    ) -> AdapterOutput:
        _validate_masked_values(activations, mask, "activations")
        if activations.shape[-1] != self.width:
            raise ValueError("activation width does not match adapter")
        directional = self.target_kind == TargetKind.DIRECTIONAL_RESIDUE.value
        relative = self.target_kind == TargetKind.RELATIVE_SEQUENCE.value
        if directional:
            if (
                standardized_target.shape != mask.shape
                or not standardized_target.is_floating_point()
                or standardized_target.device != activations.device
                or standardized_target.dtype != activations.dtype
            ):
                raise ValueError("directional target must be floating [batch, residues]")
            if not torch.isfinite(standardized_target[mask]).all():
                raise ValueError("directional target must be finite on valid residues")
            if torch.any((standardized_target[mask] < -1) | (standardized_target[mask] > 1)):
                raise ValueError("directional target must lie in [-1, 1]")
            if torch.any(standardized_target[~mask] != 0):
                raise ValueError("directional target padding must be zero")
        else:
            _validate_scalar(
                standardized_target,
                "requested_level" if relative else "standardized_target",
                activations.shape[0],
                activations,
            )
            if relative and torch.any((standardized_target < -1) | (standardized_target > 1)):
                raise ValueError("requested_level must lie in [-1, 1]")
        uses_selected_feature = self.variant == AdapterVariant.SELECTED_SAE_FEATURE.value
        if uses_selected_feature:
            if (
                not isinstance(selected_sae_activation, torch.Tensor)
                or selected_sae_activation.shape != mask.shape
                or not selected_sae_activation.is_floating_point()
                or selected_sae_activation.device != activations.device
                or selected_sae_activation.dtype != activations.dtype
                or not torch.isfinite(selected_sae_activation[mask]).all()
            ):
                raise ValueError(
                    "selected_sae_activation must be finite floating [batch, residues] "
                    "and match activations"
                )
        elif selected_sae_activation is not None:
            raise ValueError("selected_sae_activation is only valid for its declared ablation")
        site_values = self._site_values(site_field, mask, activations)

        valid = mask.unsqueeze(-1)
        clean_activations = torch.where(valid, activations, torch.zeros_like(activations))
        parts = [clean_activations]
        if self.variant != AdapterVariant.LOCAL_TARGET.value:
            parts.append(
                masked_mean(activations, mask).unsqueeze(1).expand(-1, activations.shape[1], -1)
            )
        target_values = (
            standardized_target.unsqueeze(-1)
            if directional
            else standardized_target[:, None, None].expand(-1, activations.shape[1], 1)
        )
        if self.level_conditioning not in ("branch", "none"):
            # ``branch`` routes the level through its own linear pathway into the second hidden
            # layer instead, so it never enters the first layer's input vector. ``none`` appends
            # nothing anywhere: this is the single line that makes the adapter unconditioned, and
            # dropping the column here is what guarantees no even component can form (D-101).
            parts.append(target_values)
        if uses_selected_feature:
            parts.append(
                torch.where(mask, selected_sae_activation, 0.0).unsqueeze(-1)  # type: ignore[arg-type]
            )
        projected = self.output(
            self._conditioned_hidden(torch.cat(parts, dim=-1), target_values, site_values)
        )
        if self.variant == AdapterVariant.SAE_DECODER_DIRECTION.value:
            if self.decoder_direction is None:  # pragma: no cover - constructor invariant.
                raise RuntimeError("decoder direction is missing")
            raw = projected * self.decoder_direction.to(projected)  # [B, R, 1] * [W]
        else:
            raw = projected
        if directional:
            # Local-Q alone gates on its residue control so that u = 0 is an exact
            # no-op. TargetKind.RELATIVE_SEQUENCE must never be gated (D2): that
            # would build dose proportionality into the architecture and hollow out
            # the learned mean_monotonicity metric.
            raw = raw * standardized_target.unsqueeze(-1)
        raw = raw * valid

        raw_rms = masked_rms(raw.to(dtype=torch.float64), mask)
        activation_rms = masked_rms(activations.detach(), mask)
        if self.residual_cap == "none":
            # No cap: the request IS the residual, and ``rho_max`` bounds nothing (class
            # docstring). Deliberately NOT ``raw * ones``: multiplying by a computed unit scale
            # would leave a squash-shaped hole for a future edit to fill, and the whole content
            # of this mode is that there is no multiplier between the projection and the model.
            #
            # A zero activation RMS is the one case the capped path handles by collapsing the
            # residual to zero, which is honest there because the emitted field really is zero.
            # Here there is nothing to collapse to: ``raw`` would go out at full size while
            # ``residual_fraction`` divided by zero and reported 0.0 -- a dose no recorded number
            # describes. Refused rather than clamped; a clamp here would be the cap under
            # another name.
            if not bool((activation_rms > 0).all()):
                raise ValueError(
                    "residual_cap='none' was given a sequence whose masked activation RMS is "
                    "zero, so residual_fraction = RMS(raw)/RMS(h) is undefined and would be "
                    "reported as 0.0 beside a residual that is not zero. The capped mode has a "
                    "cap to collapse to and this one does not, so it refuses instead"
                )
            residual = raw
        else:
            cap = (self.rho_max * activation_rms).to(dtype=torch.float64)
            safe_cap = torch.where(cap > 0, cap, torch.ones_like(cap))
            ratio_squared = (raw_rms / safe_cap).square()
            positive_cap_scale = _tanhc_from_squared(ratio_squared)
            scale = torch.where(
                cap > 0, positive_cap_scale, torch.zeros_like(positive_cap_scale)
            ).to(dtype=raw.dtype)
            residual = raw * scale[:, None, None]
        residual_rms = masked_rms(residual, mask)
        modified = activations + residual
        if self.residual_cap == "none" and not bool(torch.isfinite(modified).all()):
            # The LAST place a non-finite value can appear, and the only one not already
            # covered. ``masked_rms`` validates its own input, so a non-finite ``raw`` and a
            # non-finite ``residual`` have both already been refused above by
            # ``valid values elements must be finite``. What survives that is a finite residual
            # whose SUM with the activation overflows the activation dtype -- reachable here and
            # not under ``tanh``, whose scale is at most ``rho_max * RMS(h) / RMS(raw)`` and
            # therefore shrinks a large request instead of passing it on. Checked on the tensor
            # the model is actually handed, and only in this mode, so the capped path costs
            # nothing. It raises rather than clamping, for the usual reason.
            raise ValueError(
                "residual_cap='none' produced a non-finite steered activation: the residual is "
                "finite but overflows when added to the activation, and there is no cap to "
                "bound it. This mode deliberately adds no clamp; lower the learning rate, price "
                "the residual penalty, or train under residual_cap='tanh'"
            )
        residual_fraction = torch.where(
            activation_rms > 0,
            residual_rms / activation_rms,
            torch.zeros_like(residual_rms),
        )
        return AdapterOutput(
            modified=modified,
            residual=residual,
            residual_rms=residual_rms,
            residual_fraction=residual_fraction,
            raw_residual_fraction=torch.where(
                activation_rms > 0,
                raw_rms.to(dtype=residual_rms.dtype) / activation_rms,
                torch.zeros_like(residual_rms),
            ),
        )


def _approved_architecture(
    width: int,
    hidden_width: int,
    hidden_layers: int,
    rho_max: float,
    variant: AdapterVariant,
    target_kind: TargetKind,
    level_conditioning: str,
    site_conditioning: str | None = None,
    residual_cap: str = DEFAULT_RESIDUAL_CAP,
) -> AdapterArchitecture | None:
    """Expose canonical architectures for shared runtime/checkpoint serialization.

    Legacy experimental adapters retain their existing arbitrary hidden widths;
    only the primary and capacity shapes have a portable AF3/JAX representation.

    ``hidden_widths`` deliberately does not move for ``branch``: the stack is still two 64-unit
    layers. What does change is the first layer's fan-in and the second layer's, so the mode has
    to travel with the portable payload or a ``branch`` checkpoint and a ``concat`` one would
    share a ``portable_architecture`` and be interchangeable there. ``AdapterArchitecture``
    emits the key only when it is not ``concat``, so every payload written before the field
    existed stays byte-identical and no trained checkpoint's identity moves.

    ``site_conditioning`` travels the same way and for a stronger reason: it *does* move the
    second layer's fan-in, ``2H -> 3H``, so a payload that omitted it would describe a shape
    the adapter cannot build. It is likewise emitted only when present, so arm C and every
    radius-of-gyration payload are unchanged.

    ``residual_cap`` travels the same way again, and for the reason ``rho_max`` is in the
    payload at all: the payload is what a *reimplementation* of this adapter reads, and a
    reimplementation that applied ``rho_max * tanh(.)`` to an uncapped checkpoint would emit a
    different intervention at an unchanged digest. Emitted only when it is not ``tanh``, so
    every payload written before the mode existed stays byte-identical -- and an uncapped
    payload is then REFUSED by the AF3 protocol and the standalone JAX adapter, which compare
    payload keys with frozenset equality. That refusal is correct: neither implements this mode.
    """
    if variant is not AdapterVariant.PRIMARY:
        return None
    hidden_widths = (hidden_width,) * hidden_layers
    if hidden_widths in ((64, 64), (128, 128, 128, 128)):
        return AdapterArchitecture(
            width,
            hidden_widths,
            rho_max,
            target_kind=target_kind.value,
            level_conditioning=level_conditioning,
            site_conditioning=site_conditioning,
            residual_cap=residual_cap,
        )
    return None
