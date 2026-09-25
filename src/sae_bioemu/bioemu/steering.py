"""Gradient-preserving, selected-call steering at a BioEmu score-model layer.

Three incompatible score-call indexing conventions coexist in this repository (see
``docs/steering/DESIGN_DECISIONS.md`` D10):

* the *analysis ordinal* ``0..9`` used by ensemble analysis sidecars and steering manifests;
* the *capture index*, a one-based absolute call ``1..98`` used by activation capture
  (:func:`sae_bioemu.bioemu.extractor.record_score_call_indices`);
* the *score call*, a zero-based absolute call ``0..97``, which is what :class:`SteeringSite`
  and the denoiser hooks in this module count.

Copying an analysis ordinal straight into ``score_calls`` therefore steers the wrong DPM
iteration with every validator green. Configuration must instead request a diffusion
*timestep*, which :func:`resolve_bioemu_intervention_site` converts once, and
:class:`BioEmuSteeringHook` asserts the observed timestep against the resolved one.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sae_bioemu.bioemu.evaluation_archive import EVALUATION_LAYER_TARGETS
from sae_bioemu.bioemu.extractor import (
    SeededPriorSampling,
    dpm_gradient_steps_for_score_calls,
    expected_bioemu_score_calls,
    pad_member_coordinates,
    record_score_call_indices,
    resolve_module_path,
)
from sae_bioemu.bioemu.precision import (
    GenerationPrecision,
    generation_precision_scope,
    measurement_coordinates,
    resolve_generation_precision,
)
from sae_bioemu.bioemu.steering_checkpointing import checkpointed_encoder_layers
from sae_bioemu.core.config import TraceConfig
from sae_bioemu.steering.adapter import AdapterOutput, BoundedResidualAdapter, TargetKind
from sae_bioemu.steering.predictor import masked_rms
from sae_bioemu.steering.training import freeze_for_steering

BIOEMU_COSINE_SCHEDULE_S = 0.008
"""Default cosine-schedule offset of ``bioemu.sde_lib.CosineVPSDE``."""

BIOEMU_TIMESTEP_TOLERANCE = 1.0e-4
"""Absolute tolerance for asserting an observed score-call timestep.

The recorded ten-call grid separates neighbouring calls by about 0.11 and the tightest
separation between *any* two adjacent DPM score calls is about 7.4e-3, while recomputing the
grid in float32 instead of float64 moves a timestep by at most about 1.8e-6. 1e-4 therefore
distinguishes every adjacent call while leaving roughly fifty times the observed float32
drift as headroom.
"""

MAX_REQUESTED_TIMESTEP_DISTANCE = 0.06
"""How far a requested timestep may sit from the recorded call it resolves to.

Slightly more than half the ~0.111 spacing of the ten-call production grid, so a request
placed between two recorded calls resolves to the nearer one instead of failing, while a
call index copied into the ``timestep`` field fails closed. The resolution is always audited:
both the requested and the resolved timestep are recorded in the run report and in the
intervention-site identity.
"""


@dataclass(frozen=True)
class SteeringSite:
    """One target module and zero-based outer score-model calls to intervene on.

    ``expected_timesteps`` carries the diffusion timestep each selected score call must be
    observed at, so a wrong index cannot pass silently; ``expected_score_call_total`` is the
    number of score calls the configured denoiser must make in total, so changing ``N`` in
    the denoiser config cannot silently remap every index.
    """

    module_path: str
    score_calls: tuple[int, ...]
    expected_timesteps: tuple[float, ...] | None = None
    expected_score_call_total: int | None = None
    timestep_tolerance: float = BIOEMU_TIMESTEP_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.module_path, str) or not self.module_path.strip():
            raise ValueError("module_path must be non-empty")
        if not isinstance(self.score_calls, tuple) or not self.score_calls:
            raise ValueError("score_calls must be a non-empty tuple")
        if any(
            isinstance(call, bool) or not isinstance(call, int) or call < 0
            for call in self.score_calls
        ):
            raise ValueError("score_calls must contain nonnegative zero-based integers")
        if tuple(sorted(set(self.score_calls))) != self.score_calls:
            raise ValueError("score_calls must be unique and increasing")
        if self.expected_timesteps is not None:
            if not isinstance(self.expected_timesteps, tuple) or len(
                self.expected_timesteps
            ) != len(self.score_calls):
                raise ValueError("expected_timesteps must supply one timestep per score call")
            if any(not _is_diffusion_timestep(value) for value in self.expected_timesteps):
                raise ValueError("expected_timesteps must be finite diffusion timesteps in (0,1)")
        if self.expected_score_call_total is not None:
            if (
                isinstance(self.expected_score_call_total, bool)
                or not isinstance(self.expected_score_call_total, int)
                or self.expected_score_call_total <= max(self.score_calls)
            ):
                raise ValueError(
                    "expected_score_call_total must be an integer greater than every score call"
                )
        if (
            isinstance(self.timestep_tolerance, bool)
            or not isinstance(self.timestep_tolerance, int | float)
            or not math.isfinite(self.timestep_tolerance)
            or self.timestep_tolerance <= 0
        ):
            raise ValueError("timestep_tolerance must be a positive finite number")

    def expected_timestep(self, score_call: int) -> float | None:
        """Return the timestep ``score_call`` must be observed at, if one is bound."""
        if self.expected_timesteps is None:
            return None
        return self.expected_timesteps[self.score_calls.index(score_call)]


def _is_diffusion_timestep(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and 0.0 < float(value) < 1.0
    )


@dataclass(frozen=True)
class RecordedScoreCall:
    """One recorded BioEmu score call in all three indexing conventions."""

    analysis_ordinal: int
    score_call: int
    capture_index: int
    timestep: float
    denoiser_iteration: int
    role: str


@dataclass(frozen=True)
class ResolvedInterventionSite:
    """A configured ``(timestep, layer)`` request resolved to one absolute score call."""

    site: SteeringSite
    requested_timestep: float
    requested_layer: str
    module_path: str
    recorded_call: RecordedScoreCall
    expected_score_call_total: int
    timestep_tolerance: float
    max_timestep_distance: float

    @property
    def resolved_timestep(self) -> float:
        return self.recorded_call.timestep

    @property
    def score_call(self) -> int:
        """The absolute zero-based score call the hook intervenes on."""
        return self.recorded_call.score_call

    @property
    def analysis_ordinal(self) -> int:
        """The analysis call ordinal that carried the selection evidence."""
        return self.recorded_call.analysis_ordinal

    def identity(self) -> dict[str, object]:
        """Return the auditable intervention-site record for reports and identities."""
        return {
            "module_path": self.module_path,
            "requested_layer": self.requested_layer,
            "requested_timestep": float(self.requested_timestep),
            "resolved_timestep": float(self.resolved_timestep),
            "score_calls": list(self.site.score_calls),
            "absolute_score_call_zero_based": self.score_call,
            "capture_index_one_based": self.recorded_call.capture_index,
            "analysis_call_ordinal": self.analysis_ordinal,
            "denoiser_iteration": self.recorded_call.denoiser_iteration,
            "denoiser_role": self.recorded_call.role,
            "expected_score_call_total": self.expected_score_call_total,
            "timestep_tolerance": float(self.timestep_tolerance),
            "max_timestep_distance": float(self.max_timestep_distance),
        }


@dataclass(frozen=True)
class ResolvedInterventionSites:
    """One or more ``(timestep, layer)`` requests merged into ONE injection site.

    A multi-timestep request injects the *same* adapter at several recorded score calls of a
    single generation, so it is one :class:`SteeringSite` -- one module path, one score-call
    total, one tolerance -- carrying several ``score_calls``, not several sites. The hook
    already supports exactly that (``_start_score_call`` gates on membership of
    ``score_calls`` and asserts each call's own expected timestep), so nothing about the
    intervention itself changes here; what this type adds is the resolution and the audit
    record for the plural request.

    ``resolved`` keeps the per-timestep :class:`ResolvedInterventionSite` records, ordered by
    absolute score call and aligned element-for-element with ``site.score_calls`` and
    ``site.expected_timesteps``. Ordering is enforced rather than assumed because the merged
    ``expected_timesteps`` is positional: a permuted pair would bind each call to the other's
    timestep and the hook's timestep assertion would fire at run time, deep inside a loaded
    model, instead of here.
    """

    site: SteeringSite
    resolved: tuple[ResolvedInterventionSite, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.site, SteeringSite):
            raise TypeError("site must be a SteeringSite")
        if not isinstance(self.resolved, tuple) or not self.resolved:
            raise ValueError("resolved must be a non-empty tuple of resolved sites")
        if any(not isinstance(item, ResolvedInterventionSite) for item in self.resolved):
            raise TypeError("resolved must contain ResolvedInterventionSite records")
        score_calls = tuple(item.score_call for item in self.resolved)
        if score_calls != tuple(sorted(set(score_calls))):
            raise ValueError(
                "resolved sites must be ordered by unique absolute score call so the merged "
                "score_calls and expected_timesteps stay aligned"
            )
        if self.site.score_calls != score_calls:
            raise ValueError("merged site score_calls must match the resolved sites")
        if self.site.expected_timesteps != tuple(item.resolved_timestep for item in self.resolved):
            raise ValueError("merged site expected_timesteps must match the resolved sites")
        first = self.resolved[0]
        if any(
            item.module_path != first.module_path
            or item.requested_layer != first.requested_layer
            or item.expected_score_call_total != first.expected_score_call_total
            or item.timestep_tolerance != first.timestep_tolerance
            or item.max_timestep_distance != first.max_timestep_distance
            for item in self.resolved
        ):
            raise ValueError(
                "every resolved site must share one layer, module path, score-call total, "
                "timestep tolerance and maximum requested distance"
            )
        if (
            self.site.module_path != first.module_path
            or self.site.expected_score_call_total != first.expected_score_call_total
            or self.site.timestep_tolerance != first.timestep_tolerance
        ):
            raise ValueError("merged site must agree with the resolved sites it was built from")

    @property
    def module_path(self) -> str:
        return self.site.module_path

    @property
    def score_calls(self) -> tuple[int, ...]:
        """The absolute zero-based score calls the hook intervenes on."""
        return self.site.score_calls

    @property
    def requested_timesteps(self) -> tuple[float, ...]:
        return tuple(float(item.requested_timestep) for item in self.resolved)

    @property
    def resolved_timesteps(self) -> tuple[float, ...]:
        return tuple(float(item.resolved_timestep) for item in self.resolved)

    @property
    def analysis_ordinals(self) -> tuple[int, ...]:
        """The analysis call ordinals that carried the selection evidence."""
        return tuple(item.analysis_ordinal for item in self.resolved)

    def identity(self) -> dict[str, object]:
        """Return the auditable intervention-site record for reports and identities.

        **THE IDENTITY CONTRACT, which is why this method is shaped the way it is.** This
        record is fingerprinted into the training checkpoint's identity (through
        ``OrdinarySteeringConfig.intervention_site``), so its bytes decide which checkpoints
        can be replayed as each other. For ONE requested timestep it therefore returns
        exactly what :meth:`ResolvedInterventionSite.identity` has always returned -- key for
        key and value for value, by delegating to it rather than by reconstructing an
        equivalent dict -- so every single-site config that ran before multi-site existed
        keeps its identity unchanged and its checkpoints replayable.

        For several timesteps the per-site values are emitted as lists under clearly plural
        keys, and the scalar single-site keys (``absolute_score_call_zero_based``,
        ``capture_index_one_based``, ``analysis_call_ordinal``, ``denoiser_iteration``,
        ``denoiser_role``, and the singular ``requested_timestep``/``resolved_timestep``) are
        deliberately ABSENT. A reader therefore cannot mistake a multi-site record for a
        single-site one by finding a scalar call index that describes only one of the calls
        actually intervened on -- which is the same class of silent index confusion D10
        exists to prevent. Every list is ordered by absolute score call, matching
        ``score_calls``.
        """
        if len(self.resolved) == 1:
            return self.resolved[0].identity()
        first = self.resolved[0]
        return {
            "module_path": self.module_path,
            "requested_layer": first.requested_layer,
            "requested_timesteps": [float(item.requested_timestep) for item in self.resolved],
            "resolved_timesteps": [float(item.resolved_timestep) for item in self.resolved],
            "score_calls": list(self.site.score_calls),
            "capture_indices_one_based": [
                item.recorded_call.capture_index for item in self.resolved
            ],
            "analysis_call_ordinals": [item.analysis_ordinal for item in self.resolved],
            "denoiser_iterations": [
                item.recorded_call.denoiser_iteration for item in self.resolved
            ],
            "denoiser_roles": [item.recorded_call.role for item in self.resolved],
            "expected_score_call_total": first.expected_score_call_total,
            "timestep_tolerance": float(first.timestep_tolerance),
            "max_timestep_distance": float(first.max_timestep_distance),
        }


def bioemu_score_call_timesteps(
    config: TraceConfig,
    denoiser_config: Mapping[str, object],
    *,
    cosine_s: float = BIOEMU_COSINE_SCHEDULE_S,
) -> tuple[float, ...]:
    """Return the diffusion timestep of every absolute zero-based DPM score call.

    ``bioemu.denoiser.dpm_solver`` walks ``torch.linspace(max_t, eps_t, N)`` and evaluates the
    score model twice per iteration: call ``2i`` is the predictor at ``timesteps[i]`` and call
    ``2i + 1`` is the DPM-Solver-2 corrector at ``t_lambda``, the log-SNR midpoint between
    ``timesteps[i]`` and ``timesteps[i + 1]``. Both are reproduced here in float64 from the
    denoiser configuration, never from a hardcoded table.
    """
    if not isinstance(config, TraceConfig):
        raise TypeError("config must be a TraceConfig")
    if not isinstance(denoiser_config, Mapping):
        raise TypeError("denoiser_config must be a mapping")
    target = str(denoiser_config.get("_target_", ""))
    if config.denoiser_type != "dpm" and "dpm_solver" not in target:
        raise ValueError(
            "BioEmu steering timestep resolution supports only the DPM-Solver-2 denoiser; "
            f"got denoiser_type={config.denoiser_type!r} target={target!r}"
        )
    noise = denoiser_config.get("noise", 0.0)
    if not isinstance(noise, int | float) or isinstance(noise, bool) or float(noise) != 0.0:
        raise ValueError(
            "BioEmu steering timestep resolution requires denoiser noise 0.0: a stochastic "
            "sampler evaluates the predictor at t_hat != t, so the recorded grid no longer "
            f"describes the score calls (noise={noise!r})"
        )
    n_steps = denoiser_config.get("N")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError("denoiser N must be an integer of at least 2")
    max_t = _finite_float(denoiser_config.get("max_t"), "denoiser max_t")
    eps_t = _finite_float(denoiser_config.get("eps_t"), "denoiser eps_t")
    if not 0.0 < eps_t < max_t < 1.0:
        raise ValueError("denoiser timesteps require 0 < eps_t < max_t < 1")
    schedule_s = _finite_float(cosine_s, "cosine_s")
    if schedule_s <= 0.0:
        raise ValueError("cosine_s must be positive")
    total = expected_bioemu_score_calls(config, dict(denoiser_config))
    if total != 2 * (n_steps - 1):
        raise ValueError(
            "DPM score-call count does not match the two calls per iteration this grid assumes"
        )
    step = (max_t - eps_t) / (n_steps - 1)
    timesteps: list[float] = []
    for iteration in range(n_steps - 1):
        current = max_t - iteration * step
        following = current - step
        timesteps.append(current)
        timesteps.append(_dpm_corrector_timestep(current, following, schedule_s))
    if len(timesteps) != total or any(not _is_diffusion_timestep(value) for value in timesteps):
        raise ValueError("derived BioEmu score-call timesteps are not a valid diffusion grid")
    return tuple(timesteps)


def recorded_bioemu_score_calls(
    config: TraceConfig,
    denoiser_config: Mapping[str, object],
    *,
    cosine_s: float = BIOEMU_COSINE_SCHEDULE_S,
) -> tuple[RecordedScoreCall, ...]:
    """Return the recorded score calls in analysis-ordinal order.

    The selection reuses :func:`~sae_bioemu.bioemu.extractor.record_score_call_indices`, the
    same one-based capture indices activation extraction records, so the analysis ordinal a
    manifest carries maps onto exactly the call the analysis measured.
    """
    timesteps = bioemu_score_call_timesteps(config, denoiser_config, cosine_s=cosine_s)
    selected = record_score_call_indices(len(timesteps), config.recorded_score_calls)
    capture_indices = (
        tuple(range(1, len(timesteps) + 1)) if selected is None else tuple(sorted(selected))
    )
    return tuple(
        RecordedScoreCall(
            analysis_ordinal=ordinal,
            score_call=capture_index - 1,
            capture_index=capture_index,
            timestep=timesteps[capture_index - 1],
            denoiser_iteration=(capture_index - 1) // 2,
            role="predictor" if (capture_index - 1) % 2 == 0 else "corrector",
        )
        for ordinal, capture_index in enumerate(capture_indices)
    )


def resolve_bioemu_layer(layer: object) -> str:
    """Expand a captured-layer shorthand to the dotted BioEmu module path.

    Accepts the archive keys of
    :data:`~sae_bioemu.bioemu.evaluation_archive.EVALUATION_LAYER_TARGETS` (``layer_m1``), the
    bare negative index (``-1`` as an integer or a string), or a full dotted module path. All
    four captured layers are 512-wide, so no shape guard downstream can catch a wrong layer,
    and the archive key form would otherwise fail only at first generation deep inside a
    loaded model.
    """
    shorthand: dict[str, str] = {}
    for key, module_path in EVALUATION_LAYER_TARGETS.items():
        shorthand[key] = module_path
        shorthand[module_path.rsplit(".", 1)[-1]] = module_path
    value = str(layer) if isinstance(layer, int) and not isinstance(layer, bool) else layer
    if not isinstance(value, str) or not value.strip():
        raise ValueError("site.layer must be a captured-layer shorthand or a dotted module path")
    value = value.strip()
    if value in shorthand:
        return shorthand[value]
    if "." in value:
        return value
    raise ValueError(
        "site.layer must be one of the captured-layer shorthands "
        f"({', '.join(sorted(shorthand))}) or a full dotted module path; got {layer!r}"
    )


def resolve_bioemu_intervention_site(
    *,
    timestep: object,
    layer: object,
    config: TraceConfig,
    denoiser_config: Mapping[str, object],
    cosine_s: float = BIOEMU_COSINE_SCHEDULE_S,
    timestep_tolerance: float = BIOEMU_TIMESTEP_TOLERANCE,
    max_timestep_distance: float = MAX_REQUESTED_TIMESTEP_DISTANCE,
) -> ResolvedInterventionSite:
    """Resolve a requested ``(timestep, layer)`` to one recorded absolute score call.

    Only recorded calls carry prior analysis evidence, so a request that is not near one
    fails closed rather than snapping to an arbitrary call.
    """
    if not _is_diffusion_timestep(timestep):
        raise ValueError(
            "site.timestep must be a finite diffusion timestep in (0,1); "
            f"got {timestep!r}. It is a diffusion time, not a score-call index."
        )
    requested = float(timestep)
    module_path = resolve_bioemu_layer(layer)
    distance = _finite_float(max_timestep_distance, "max_timestep_distance")
    if distance <= 0:
        raise ValueError("max_timestep_distance must be positive")
    recorded = recorded_bioemu_score_calls(config, denoiser_config, cosine_s=cosine_s)
    if not recorded:
        raise ValueError("the configured BioEmu denoiser records no score calls")
    ranked = sorted(recorded, key=lambda call: (abs(call.timestep - requested), call.score_call))
    nearest = ranked[0]
    if abs(nearest.timestep - requested) > distance:
        raise ValueError(
            f"requested diffusion timestep {requested} is not within {distance} of any recorded "
            "BioEmu score call, so no prior analysis evidence exists for it. "
            f"Available recorded timesteps: {_describe_recorded(recorded)}"
        )
    if len(ranked) > 1 and (
        abs(abs(ranked[1].timestep - requested) - abs(nearest.timestep - requested)) <= 1.0e-12
    ):
        raise ValueError(
            f"requested diffusion timestep {requested} is exactly between two recorded BioEmu "
            "score calls, so the intervention site is ambiguous. "
            f"Available recorded timesteps: {_describe_recorded(recorded)}"
        )
    total = expected_bioemu_score_calls(config, dict(denoiser_config))
    site = SteeringSite(
        module_path,
        (nearest.score_call,),
        expected_timesteps=(nearest.timestep,),
        expected_score_call_total=total,
        timestep_tolerance=timestep_tolerance,
    )
    return ResolvedInterventionSite(
        site=site,
        requested_timestep=requested,
        requested_layer=str(layer),
        module_path=module_path,
        recorded_call=nearest,
        expected_score_call_total=total,
        timestep_tolerance=float(timestep_tolerance),
        max_timestep_distance=distance,
    )


def resolve_bioemu_intervention_sites(
    *,
    timesteps: object,
    layer: object,
    config: TraceConfig,
    denoiser_config: Mapping[str, object],
    cosine_s: float = BIOEMU_COSINE_SCHEDULE_S,
    timestep_tolerance: float = BIOEMU_TIMESTEP_TOLERANCE,
    max_timestep_distance: float = MAX_REQUESTED_TIMESTEP_DISTANCE,
) -> ResolvedInterventionSites:
    """Resolve several requested timesteps at one layer into a single merged injection site.

    Each timestep goes through :func:`resolve_bioemu_intervention_site` unchanged, so the
    nearest-recorded-call search, the ``max_timestep_distance`` refusal and the exact-midpoint
    ambiguity refusal are the production ones rather than a second implementation that could
    drift from them. The per-timestep records are then merged into one :class:`SteeringSite`,
    because a multi-site request is one adapter injected at several calls of one generation.

    Two requests that land on the same recorded call are refused: the merged ``score_calls``
    must be unique (:class:`SteeringSite` enforces that too), and silently collapsing a pair
    would give a run fewer injections than its configuration asked for while reporting the
    request it did not honour.
    """
    if isinstance(timesteps, str | bytes) or not isinstance(timesteps, Sequence):
        raise ValueError(
            "site.timesteps must be a list of finite diffusion timesteps in (0,1); "
            f"got {timesteps!r}"
        )
    requested = tuple(timesteps)
    if not requested:
        raise ValueError("site.timesteps must contain at least one diffusion timestep")
    resolved: list[ResolvedInterventionSite] = []
    for value in requested:
        site = resolve_bioemu_intervention_site(
            timestep=value,
            layer=layer,
            config=config,
            denoiser_config=denoiser_config,
            cosine_s=cosine_s,
            timestep_tolerance=timestep_tolerance,
            max_timestep_distance=max_timestep_distance,
        )
        clash = next((item for item in resolved if item.score_call == site.score_call), None)
        if clash is not None:
            raise ValueError(
                "site.timesteps requests two timesteps that resolve to the same recorded BioEmu "
                f"score call {site.score_call} (t={site.resolved_timestep:.5f}): "
                f"{clash.requested_timestep} and {site.requested_timestep}. Requesting one call "
                "twice would inject once while reporting two sites"
            )
        resolved.append(site)
    # Sorted by absolute score call, which is also denoising order, so the merged
    # ``expected_timesteps`` lines up positionally with ``score_calls`` whatever order the
    # configuration listed its timesteps in.
    ordered = tuple(sorted(resolved, key=lambda item: item.score_call))
    first = ordered[0]
    merged = SteeringSite(
        first.module_path,
        tuple(item.score_call for item in ordered),
        expected_timesteps=tuple(item.resolved_timestep for item in ordered),
        expected_score_call_total=first.expected_score_call_total,
        timestep_tolerance=first.timestep_tolerance,
    )
    return ResolvedInterventionSites(site=merged, resolved=ordered)


def _describe_recorded(recorded: tuple[RecordedScoreCall, ...]) -> str:
    return "; ".join(
        f"ordinal {call.analysis_ordinal} t={call.timestep:.5f} (score call {call.score_call})"
        for call in recorded
    )


def _dpm_corrector_timestep(current: float, following: float, schedule_s: float) -> float:
    """Return ``t_lambda``, the log-SNR midpoint DPM-Solver-2 evaluates its corrector at."""
    lambda_middle = (
        _cosine_log_snr(current, schedule_s) + _cosine_log_snr(following, schedule_s)
    ) / 2.0
    if -2.0 * lambda_middle > 700.0:
        raise ValueError("DPM log-SNR midpoint overflows; the timestep grid is out of range")
    f_lambda = -0.5 * math.log(math.exp(-2.0 * lambda_middle) + 1.0)
    exponent = f_lambda + math.log(math.cos(math.pi * schedule_s / 2.0 / (1.0 + schedule_s)))
    argument = min(max(math.exp(exponent), -1.0), 1.0)
    return 2.0 * (1.0 + schedule_s) / math.pi * math.acos(argument) - schedule_s


def _cosine_log_snr(timestep: float, schedule_s: float) -> float:
    coefficient = math.cos(schedule_s / (1.0 + schedule_s) * math.pi / 2.0)
    alpha = math.cos((timestep + schedule_s) / (1.0 + schedule_s) * math.pi / 2.0) / coefficient
    alpha = min(max(alpha, 0.0), 1.0)
    sigma = math.sqrt(max(1.0 - alpha * alpha, 0.0))
    if alpha <= 0.0 or sigma <= 0.0:
        raise ValueError("cosine schedule log-SNR is undefined at this timestep")
    return math.log(alpha / sigma)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class EnsembleMember:
    """One differentiable generated member and its regularization diagnostics."""

    coordinates: torch.Tensor
    residual_penalty: torch.Tensor
    geometry_penalty: torch.Tensor


@dataclass(frozen=True)
class EnsembleBatch:
    """A seed-ordered differentiable ensemble."""

    coordinates: torch.Tensor
    residual_penalty: torch.Tensor
    geometry_penalty: torch.Tensor


@dataclass(frozen=True)
class BioEmuSteeredSample:
    """One differentiable BioEmu structure and adapter diagnostics."""

    sampled_batch: object
    coordinates: torch.Tensor
    residual_rms: torch.Tensor
    residual_fraction: torch.Tensor
    #: Pre-cap request, averaged over the intervened score calls exactly as the realised
    #: fraction is. ``None`` when any hook predates the field, so a penalty that needs it
    #: fails loudly rather than silently reverting to the saturating measure.
    raw_residual_fraction: torch.Tensor | None = None


def capture_steered_ensemble(
    generate_member: Callable[[int, torch.Tensor, torch.Tensor], EnsembleMember],
    seeds: tuple[int, ...],
    target_profile: torch.Tensor,
    residue_mask: torch.Tensor,
) -> EnsembleBatch:
    """Generate and stack one sequence's steered members in exact seed order."""
    if not callable(generate_member):
        raise TypeError("generate_member must be callable")
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
    ):
        raise ValueError("ensemble seeds must be unique nonnegative integers")
    if target_profile.ndim != 1 or not target_profile.is_floating_point():
        raise ValueError("target_profile must be floating [residues]")
    if residue_mask.dtype is not torch.bool or residue_mask.shape != target_profile.shape:
        raise ValueError("residue_mask must be boolean and match target_profile")
    members = tuple(generate_member(seed, target_profile, residue_mask) for seed in seeds)
    if any(not isinstance(member, EnsembleMember) for member in members):
        raise TypeError("generate_member must return EnsembleMember")
    coordinates = torch.stack(tuple(member.coordinates for member in members))
    expected = (len(seeds), residue_mask.numel(), 3)
    if coordinates.shape != expected or not coordinates.is_floating_point():
        raise ValueError("ensemble coordinates have invalid residue alignment")
    if coordinates.device != target_profile.device or not coordinates.requires_grad:
        raise RuntimeError("ensemble coordinates must preserve gradients on the target device")
    if not torch.isfinite(coordinates[:, residue_mask]).all():
        raise ValueError("valid ensemble coordinates must be finite")
    residual = _mean_member_diagnostic(members, "residual_penalty", coordinates)
    geometry = _mean_member_diagnostic(members, "geometry_penalty", coordinates)
    return EnsembleBatch(coordinates, residual, geometry)


def _mean_member_diagnostic(
    members: tuple[EnsembleMember, ...], name: str, coordinates: torch.Tensor
) -> torch.Tensor:
    values = tuple(getattr(member, name) for member in members)
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 0
        or value.device != coordinates.device
        or not torch.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{name} must contain finite scalar tensors on the coordinate device")
    return torch.stack(values).mean()


def _validate_site_field(
    adapter: BoundedResidualAdapter,
    site_field: torch.Tensor | None,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
) -> None:
    """Refuse a site channel that does not match the adapter, before the model is even called.

    The adapter refuses both directions itself (``BoundedResidualAdapter._site_values``), and this
    is not a second copy of that rule: it is the same rule applied one layer earlier, where the
    other conditioning input is already checked. Without it a conditioned adapter handed no field
    -- an arm A checkpoint silently generating as arm C -- fails inside the denoiser at the first
    intervened score call, after the score model has been moved to the device and 40-odd
    iterations have run. The value checks (range, the ``{0, 1}`` strictness of ``pocket_binary``,
    the ``max == 1`` of both annotation sources, the per-sequence standardization) stay in the
    adapter, which is the only place that knows which source it declared.
    """
    conditioned = adapter.site_conditioning is not None
    if not conditioned:
        if site_field is not None:
            raise ValueError(
                "a site_field was supplied for an adapter that declares no site_conditioning; it "
                "would be computed, dropped, and absent from every identity the run records"
            )
        return
    if site_field is None:
        raise ValueError(
            f"this adapter declares site_conditioning={adapter.site_conditioning!r} and no "
            "site_field was supplied, so it would generate as the unconditioned arm C while "
            "reporting a conditioned arm's identity"
        )
    if (
        not isinstance(site_field, torch.Tensor)
        or site_field.shape != residue_mask.shape
        or not site_field.is_floating_point()
        or site_field.dtype != target.dtype
        or site_field.device != residue_mask.device
    ):
        raise ValueError(
            "site_field must be a floating [batch, residues] tensor matching the residue mask's "
            "shape and device and the steering target's dtype; the adapter runs at the target's "
            "precision in every generation mode, so a channel in the model's own dtype would be "
            "refused there instead"
        )


class BioEmuSteeringHook:
    """Modify one BioEmu tensor layer at selected calls with only the steering adapter."""

    def __init__(
        self,
        score_model: nn.Module,
        *,
        site: SteeringSite,
        adapter: BoundedResidualAdapter,
        target: torch.Tensor,
        residue_mask: torch.Tensor,
        expected_property_name: str,
        expected_model_family: str,
        enabled: bool = True,
        autocast_dtype: torch.dtype | None = None,
        site_field: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(score_model, nn.Module):
            raise TypeError("score_model must be a torch module")
        if not isinstance(site, SteeringSite):
            raise TypeError("site must be a SteeringSite")
        if not isinstance(adapter, BoundedResidualAdapter) or adapter.model_family != "bioemu":
            raise TypeError("adapter must be a BioEmu BoundedResidualAdapter")
        if adapter.variant != "primary":
            raise ValueError(
                "active BioEmu steering supports only the learned bounded residual adapter"
            )
        if expected_model_family != "bioemu":
            raise ValueError("expected_model_family must be bioemu")
        if not isinstance(expected_property_name, str) or not expected_property_name.strip():
            raise ValueError("expected_property_name must be non-empty")
        if adapter.property_name != expected_property_name:
            raise ValueError("adapter property does not match expected_property_name")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        if autocast_dtype is not None and (
            not isinstance(autocast_dtype, torch.dtype)
            or not autocast_dtype.is_floating_point
            or autocast_dtype is torch.float32
        ):
            raise ValueError("autocast_dtype must be a reduced-precision floating dtype or None")
        if not isinstance(target, torch.Tensor) or not target.is_floating_point():
            raise ValueError("target must be a floating tensor")
        if residue_mask.ndim != 2 or residue_mask.dtype is not torch.bool:
            raise ValueError("residue_mask must be boolean [batch, residues]")
        expected_target_shape = (residue_mask.shape[0],)
        if adapter.target_kind == TargetKind.DIRECTIONAL_RESIDUE.value:
            expected_target_shape = residue_mask.shape
        if target.shape != expected_target_shape:
            raise ValueError("target shape does not match adapter target kind and residue mask")
        if target.device != residue_mask.device or not torch.isfinite(target).all():
            raise ValueError("target must be finite and share the residue-mask device")
        if adapter.target_kind == TargetKind.DIRECTIONAL_RESIDUE.value and (
            torch.any((target < -1) | (target > 1)) or torch.any(target[~residue_mask] != 0)
        ):
            raise ValueError("directional residue controls must be in [-1,1] with zero padding")
        _validate_site_field(adapter, site_field, target, residue_mask)

        self.score_model = score_model
        self.site = site
        self.adapter = adapter
        self.target = target
        self.residue_mask = residue_mask
        #: The cryptic arms' per-residue channel ``s_i``, or ``None`` for every unconditioned
        #: adapter. It is a per-SEQUENCE constant, not a per-call quantity: it is measured on the
        #: deposited apo structure the annotation was made on, so recomputing it from the tensor
        #: being steered would let a structure that pushes residues away from the pocket shrink
        #: its own conditioning field. Held here for the life of the generation and handed to
        #: every intervened score call unchanged.
        self.site_field = site_field
        self.enabled = enabled
        self.autocast_dtype = autocast_dtype
        self.n_score_calls = 0
        self.intervention_count = 0
        self._target_module = resolve_module_path(score_model, site.module_path)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_score_call: int | None = None
        self._intervened_current_call = False
        self.adapter_outputs: list[AdapterOutput] = []

    def __enter__(self) -> BioEmuSteeringHook:
        if self._handles:
            raise RuntimeError("BioEmuSteeringHook is already active")
        try:
            self._handles.append(self._target_module.register_forward_hook(self._steer_output))
            self._handles.append(self.score_model.register_forward_pre_hook(self._start_score_call))
            self._handles.append(
                self.score_model.register_forward_hook(self._finish_score_call, always_call=True)
            )
        except Exception:
            self._remove_hooks()
            raise
        return self

    def __exit__(self, _exception_type: object, _exception: object, _traceback: object) -> bool:
        self._remove_hooks()
        return False

    def _start_score_call(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        score_call = self.n_score_calls
        self.n_score_calls += 1
        self._intervened_current_call = False
        if not self.enabled or score_call not in self.site.score_calls:
            self._current_score_call = None
            return
        if len(inputs) < 2 or not isinstance(inputs[1], torch.Tensor):
            raise TypeError("BioEmu score model must receive a tensor timestep as its second input")
        timestep = inputs[1]
        if timestep.shape != (self.residue_mask.shape[0],) or not timestep.is_floating_point():
            raise ValueError("score-model timestep must be floating [batch]")
        if not torch.isfinite(timestep).all():
            raise ValueError("score-model timestep must be finite")
        self._assert_expected_timestep(score_call, timestep)
        self._current_score_call = score_call

    def _assert_expected_timestep(self, score_call: int, timestep: torch.Tensor) -> None:
        """Fail closed when a selected score call runs at the wrong diffusion timestep.

        This is the single check that closes the whole class of silent score-call indexing
        errors described in ``docs/steering/DESIGN_DECISIONS.md`` D10: the denoiser hands the
        hook the true timestep, so an index taken from the wrong convention lands on a call
        whose timestep does not match the site the analysis selected.
        """
        expected = self.site.expected_timestep(score_call)
        if expected is None:
            return
        observed = timestep.detach().reshape(-1).to(torch.float64)
        deviation = float((observed - float(expected)).abs().max().item())
        if deviation > self.site.timestep_tolerance:
            raise RuntimeError(
                "BioEmu steering reached the wrong diffusion timestep: zero-based score call "
                f"{score_call} was resolved at t={float(expected):.6f} but the score model was "
                f"called at t={float(observed[0].item()):.6f} "
                f"(|dt|={deviation:.3e} exceeds tolerance {self.site.timestep_tolerance:.3e}). "
                "Score-call indices are absolute and zero-based here; analysis ordinals and "
                "one-based capture indices must be converted, never copied."
            )

    def assert_expected_score_calls(self) -> None:
        """Fail closed when the denoiser did not make the configured number of score calls."""
        expected = self.site.expected_score_call_total
        if expected is None:
            return
        if self.n_score_calls != expected:
            raise RuntimeError(
                "BioEmu denoiser made "
                f"{self.n_score_calls} score calls but the resolved intervention site expects "
                f"{expected}; changing N in the denoiser config remaps every absolute score-call "
                "index and invalidates the resolved timestep"
            )

    def _finish_score_call(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: object
    ) -> None:
        del output
        self._current_score_call = None
        self._intervened_current_call = False

    def _steer_output(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: object
    ) -> torch.Tensor | None:
        if self._current_score_call is None:
            return None
        if self._intervened_current_call:
            raise RuntimeError("target BioEmu module executed more than once in one score call")
        if not isinstance(output, torch.Tensor):
            raise TypeError("target BioEmu module must have a tensor output")
        if output.ndim != 3 or output.shape[:2] != self.residue_mask.shape:
            raise ValueError(
                "target BioEmu tensor output has invalid residue alignment; expected [batch, residue, width]"
            )
        if output.device != self.target.device or not self._accepts_dtype(output.dtype):
            raise ValueError(
                "target BioEmu tensor output must match steering target device and dtype"
            )
        if output.shape[-1] != self.adapter.width:
            raise ValueError("target BioEmu tensor width must match the adapter input width")
        adapter_output = self._apply_adapter(output)
        modified = adapter_output.modified
        if modified.dtype != output.dtype:
            modified = modified.to(output.dtype)
            adapter_output = self._realised_output(adapter_output, modified, output)
        self.adapter_outputs.append(adapter_output)
        self._intervened_current_call = True
        self.intervention_count += 1
        return modified

    def _accepts_dtype(self, dtype: torch.dtype) -> bool:
        """Return whether a hooked activation dtype is one this hook declared.

        With no autocast declared this is exactly the historical strict equality, so a real
        device/dtype mismatch -- the class of error ``DESIGN_DECISIONS.md`` D7 shows the cost
        of hiding -- still fails closed. Under a declared autocast mode the score model's own
        matmuls emit the autocast dtype, so *that one additional* dtype is admitted and
        nothing else is: a float64 activation, or a bf16 activation in a run that declared no
        autocast, is still refused.
        """
        return dtype == self.target.dtype or (
            self.autocast_dtype is not None and dtype is self.autocast_dtype
        )

    def _apply_adapter(self, output: torch.Tensor) -> AdapterOutput:
        """Run the adapter at the target's precision, outside any active autocast.

        The alternative -- casting ``self.target`` down and letting the adapter run in bf16 --
        was rejected. The adapter's bounded-residual cap is ``rho_max * RMS_m(h)`` compared
        against ``RMS_m(raw)`` through a tanh (``steering/adapter.py``), and ``F_delta`` is the
        quantity D13's penalty is normalized by and D16's residual reduction squares. bf16
        carries about three significant decimal digits, so computing the cap in it would move
        the realised residual fraction by a few tenths of a percent for no speed gain: the
        adapter is a two-layer MLP over ``[B, R, 512]`` beside a full score-model call. The
        activation is therefore widened to the target's float32 losslessly, the adapter is
        applied to that, and only the value handed back to the denoiser is returned to the
        model's own dtype.

        **Scope, stated exactly, because the difference matters.** What this guarantees in
        every mode is the *cap arithmetic*: ``masked_rms``, the ``tanh``/``tanhc`` scaling,
        ``residual_rms``, and ``residual_fraction`` are reductions and elementwise ops, never
        matmuls, and they run on float32 (``raw`` is promoted to float64 before ``masked_rms``
        as it always was). So the realised residual fraction is measured exactly at ``fp32``,
        ``tf32``, and ``bf16`` alike. What it does *not* claim is that ``raw`` -- the adapter
        MLP's own two matmuls -- is bit-identical across modes: under ``tf32`` those matmuls
        follow the process TF32 flag like every other matmul in the generation, which perturbs
        the residual *direction* by order 1e-3 relative. That is a different intervention, not
        a mismeasured one, and it is why any mode other than ``fp32`` is bound into the
        checkpoint identity rather than treated as a free speed knob.
        """
        if self.autocast_dtype is None:
            return self.adapter(output, self.target, self.residue_mask, site_field=self.site_field)
        activations = output.to(self.target.dtype)
        with torch.autocast(output.device.type, enabled=False):
            return self.adapter(
                activations, self.target, self.residue_mask, site_field=self.site_field
            )

    def _realised_output(
        self, adapter_output: AdapterOutput, modified: torch.Tensor, output: torch.Tensor
    ) -> AdapterOutput:
        """Re-measure the residual the score model actually received after the cast back.

        **This is not bookkeeping; it is the difference between a reported spend and a real
        one.** bf16 has an 8-bit mantissa, so at an activation magnitude of order 1 its
        spacing is about 4e-3, while the residual fraction measured on this path is under 2%
        of ``rho_max = 0.10`` -- around 6e-4. A large share of a small intervention is
        therefore rounded away at the injection point: measured on a synthetic ``[1, 170,
        512]`` activation in that regime, a requested fraction of 6.39e-4 arrives as 4.34e-4,
        a 32% shortfall. The pre-cast number would report a spend no structure received,
        which is exactly what D13's penalty is normalized by and what D16's residual reduction
        squares *per structure* precisely because every structure is an observation of what
        was spent.

        The cast is differentiable, so the penalty keeps a straight-through gradient and the
        regularizer still shapes behaviour; what changes is that it now shapes the realised
        intervention rather than the intended one. Nothing here runs when the dtypes already
        agree, so ``fp32`` and ``tf32`` take the pre-existing path untouched.
        """
        activations = output.to(self.target.dtype)
        residual = modified.to(self.target.dtype) - activations
        residual_rms = masked_rms(residual, self.residue_mask)
        activation_rms = masked_rms(activations, self.residue_mask)
        return AdapterOutput(
            modified=modified,
            residual=residual,
            residual_rms=residual_rms,
            residual_fraction=torch.where(
                activation_rms > 0, residual_rms / activation_rms, torch.zeros_like(residual_rms)
            ),
            # The cast changes what the score model received, not what the adapter asked
            # for, so the pre-cap request passes through unmodified.
            raw_residual_fraction=adapter_output.raw_residual_fraction,
        )

    def _remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._current_score_call = None
        self._intervened_current_call = False


def capture_or_steer_from_denoiser(
    score_model: nn.Module,
    denoiser: object,
    sdes: dict,
    batch: object,
    site: SteeringSite,
    adapter: BoundedResidualAdapter,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    expected_property_name: str,
    expected_model_family: str,
    prior_seeds: tuple[int, ...] | None = None,
    precision: GenerationPrecision | str | None = None,
    site_field: torch.Tensor | None = None,
) -> BioEmuSteeredSample:
    """Run a differentiable BioEmu denoiser with only adapter parameters trainable.

    ``prior_seeds`` gives one seed per batch member, so each generated structure
    takes its initial noised state from its own seeded stream. With one seed the
    draw is bit-identical to the unpatched path; see
    :class:`~sae_bioemu.bioemu.extractor.SeededPriorSampling`.

    ``residue_mask`` is ``[B, max_residues]``. Members may differ in length: the
    per-member node counts are read from the mask, the padded dense representation
    the structure module already produces is what the hook steers, and the flat
    node-concatenated final positions are scattered back into padded rows.

    ``precision`` scopes TF32 and bf16 autocast to the denoiser call alone; ``fp32``,
    the default, touches no backend flag and is byte-identical to the pre-precision
    path (:mod:`sae_bioemu.bioemu.precision`).

    ``site_field`` is the cryptic arms' ``[batch, residues]`` channel ``s_i``, required by and
    only by an adapter that declares ``site_conditioning``. It is resolved once per protein by
    the caller and is constant for the whole generation -- it is measured on the deposited apo
    structure, not on the tensor being steered -- so it is passed here rather than recomputed in
    the hook, whose ``_apply_adapter`` runs once per intervened score call. It is deliberately
    NOT an input to ``prior_seeds`` or to anything that chooses a noise draw (D-23): the channel
    selects residues *within* a protein, while the seed decides which draws exist, and routing
    the first through the second would stop arms sharing draws and reintroduce the sd-0.0057
    re-randomisation floor a previous fix removed.
    """
    if not callable(denoiser):
        raise TypeError("denoiser must be callable")
    resolved_precision = resolve_generation_precision(precision)
    device = _module_device(score_model, target.device)
    score_model.to(device)
    adapter.to(device)
    target = target.to(device)
    residue_mask = residue_mask.to(device)
    if site_field is not None:
        if not isinstance(site_field, torch.Tensor):
            raise TypeError("site_field must be a tensor")
        # To the target's dtype as well as its device: the adapter runs at the target's precision
        # in every generation mode, so a channel left in the model's own dtype would be refused
        # there rather than here.
        site_field = site_field.to(device=device, dtype=target.dtype)
    if hasattr(batch, "to"):
        batch = batch.to(device)

    freeze_for_steering(score_model, adapter)
    denoiser_kwargs: dict[str, object] = {
        "sdes": sdes,
        "device": device,
        "batch": batch,
        "score_model": score_model,
    }
    if _accepts_keyword(denoiser, "record_grad_steps"):
        denoiser_kwargs["record_grad_steps"] = dpm_gradient_steps_for_score_calls(site.score_calls)

    member_residues = tuple(int(count) for count in residue_mask.sum(dim=1).tolist())
    if any(count < 1 for count in member_residues):
        raise ValueError("every batch member must contain at least one valid residue")

    seeded_priors: AbstractContextManager[object] = nullcontext()
    if prior_seeds is not None:
        if len(prior_seeds) != residue_mask.shape[0]:
            raise ValueError("prior_seeds must provide one seed per batch member")
        seeded_priors = SeededPriorSampling(
            sdes,
            prior_seeds,
            # One member cannot be split, so the equal-split path is used unchanged and
            # the draw stays bit-identical to the pre-batching path.
            member_nodes=member_residues if len(prior_seeds) > 1 else None,
            # Two members sharing a seed start from the same prior state and then diverge
            # only through the adapter's level-dependent residual -- which is exactly what
            # the paired dose objective needs, since that shared start is what cancels the
            # baseline in their difference. The caller has already rejected members matching
            # in sequence, seed AND level, which is the case that would really be a duplicate.
            allow_duplicate_seeds=True,
        )

    with (
        seeded_priors,
        BioEmuSteeringHook(
            score_model,
            site=site,
            adapter=adapter,
            target=target,
            residue_mask=residue_mask,
            expected_property_name=expected_property_name,
            expected_model_family=expected_model_family,
            autocast_dtype=resolved_precision.autocast_dtype,
            site_field=site_field,
        ) as hook,
    ):
        # Only the denoiser call, so nothing outside this block inherits the backend flags.
        # The seeded prior draws happen inside it but are unaffected: autocast rewrites
        # matmul-family ops, never the factory and indexing ops ``prior_sampling`` uses, so
        # the drawn state stays bit-identical and SeededPriorSampling's ambient-RNG
        # assertion still holds.
        with (
            # INSIDE the hook scope, never outside it. The scope refuses to checkpoint any
            # layer carrying a hook, and it can only see the steering hook once that hook is
            # attached; entering it earlier would let a site configured at some layer other
            # than the last one be recomputed WITHOUT its intervention. It covers the steered
            # generation alone, so the unsteered and no-grad paths are byte-identical
            # (:mod:`sae_bioemu.bioemu.steering_checkpointing`).
            checkpointed_encoder_layers(score_model, site.module_path),
            generation_precision_scope(resolved_precision, device.type),
        ):
            sampled_batch = denoiser(**denoiser_kwargs)

    hook.assert_expected_score_calls()
    if hook.intervention_count != len(site.score_calls):
        raise RuntimeError(
            "BioEmu steering intervention count did not match the selected score calls: "
            f"expected {len(site.score_calls)}, observed {hook.intervention_count}"
        )
    if isinstance(sampled_batch, tuple) or not hasattr(sampled_batch, "pos"):
        raise TypeError("BioEmu denoiser output must be one ChemGraph-like object with .pos")
    positions = sampled_batch.pos
    if (
        not isinstance(positions, torch.Tensor)
        or not positions.is_floating_point()
        or positions.ndim not in (2, 3)
        or positions.shape[-1] != 3
    ):
        raise TypeError("BioEmu denoiser output .pos must be a floating [..., 3] tensor")
    if not positions.requires_grad:
        raise RuntimeError("BioEmu denoiser final positions are not differentiable")
    if not torch.isfinite(positions).all():
        raise ValueError("BioEmu denoiser final positions must be finite")
    # The property is a measurement, not part of the generation: it is taken from a float32
    # view of whatever the denoiser produced, so a bf16 run measures its own bf16 structure
    # exactly rather than re-quantizing the answer. fp32 and tf32 never change an output
    # dtype, so this is a no-op for them.
    positions = measurement_coordinates(positions, resolved_precision)
    batch_size, width = residue_mask.shape
    if positions.numel() == batch_size * width * 3:
        # Every member fills the padded width (always true at batch size 1), so the
        # reshape of the pre-batching path is reused verbatim.
        coordinates = positions.reshape(batch_size, width, 3)
    elif positions.numel() == sum(member_residues) * 3:
        coordinates = pad_member_coordinates(positions, member_residues)
    else:
        raise ValueError("BioEmu final positions do not match the residue mask")
    residual_rms = torch.stack([item.residual_rms for item in hook.adapter_outputs]).mean(dim=0)
    residual_fraction = torch.stack([item.residual_fraction for item in hook.adapter_outputs]).mean(
        dim=0
    )
    raw_fractions = [item.raw_residual_fraction for item in hook.adapter_outputs]
    raw_residual_fraction = (
        None
        if any(value is None for value in raw_fractions)
        else torch.stack(raw_fractions).mean(dim=0)
    )
    return BioEmuSteeredSample(
        sampled_batch=sampled_batch,
        coordinates=coordinates,
        residual_rms=residual_rms,
        residual_fraction=residual_fraction,
        raw_residual_fraction=raw_residual_fraction,
    )


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    first_parameter = next(module.parameters(), None)
    return first_parameter.device if first_parameter is not None else fallback


def _accepts_keyword(function: object, name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def steering_site_from_record(spec: dict[str, object]) -> SteeringSite:
    """Rebuild the site an adapter was TRAINED at from its recorded ``intervention_site``.

    One reconstruction for every caller, so the command and its test cannot drift apart: they did
    once, when the test rebuilt the site without the two guard fields and passed while the command
    died on them. Both record shapes are accepted -- a multi-site record carries the plural
    ``resolved_timesteps`` and none of the scalar single-site keys -- and nothing is defaulted:
    ``expected_score_call_total`` and ``timestep_tolerance`` are the guards that catch a wrong call
    index, so a record without them is refused rather than run unguarded.
    """
    if "resolved_timesteps" in spec:
        expected_timesteps = tuple(float(value) for value in spec["resolved_timesteps"])
    elif "resolved_timestep" in spec:
        expected_timesteps = (float(spec["resolved_timestep"]),)
    else:
        raise KeyError(
            "intervention_site names neither resolved_timesteps nor resolved_timestep, so the "
            "timesteps the score calls must be observed at are unknown"
        )
    missing = [k for k in ("expected_score_call_total", "timestep_tolerance") if k not in spec]
    if missing:
        raise KeyError(f"intervention_site lacks {missing}; without them a wrong call index runs")
    return SteeringSite(
        module_path=str(spec["module_path"]),
        score_calls=tuple(int(value) for value in spec["score_calls"]),
        expected_timesteps=expected_timesteps,
        expected_score_call_total=int(spec["expected_score_call_total"]),
        timestep_tolerance=float(spec["timestep_tolerance"]),
    )
