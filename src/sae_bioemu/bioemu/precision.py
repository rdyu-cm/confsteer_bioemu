"""Config-controlled reduced-precision BioEmu generation.

Everything on this path runs in true FP32 today: ``seed_everything`` sets
``torch.use_deterministic_algorithms`` and ``cudnn.deterministic``, the Slurm launchers
export ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, and nothing ever enables TF32 or autocast. That
is why an H100 SXM5 benchmarked at only 1.07x an L40S (``DESIGN_DECISIONS.md`` D15): the
H100's advantage lives entirely in tensor cores that plain FP32 does not use.

TF32 and bf16 autocast are **deterministic** -- the same inputs give the same outputs on the
same hardware and software -- so they trade *precision* for speed, not reproducibility. They
must nonetheless be recorded, because they change the generated structure, and a checkpoint
trained under one may not be evaluated under another.

Three modes:

``fp32``
    The default and a strict no-op. :func:`generation_precision_scope` does not read or write
    a single backend flag, so a run is byte-identical to one made before this module existed.
``tf32``
    ``torch.backends.cuda.matmul.allow_tf32`` and ``torch.backends.cudnn.allow_tf32``, scoped
    to the generation and restored afterwards.
``bf16``
    The TF32 flags **and** ``torch.autocast(device_type, dtype=torch.bfloat16)`` around the
    denoiser call.

Two things deliberately stay at full precision under every mode, because they are
measurements rather than generation:

* the **steering adapter and its bounded-residual cap arithmetic**, which run with autocast
  explicitly disabled on a float32 view of the hooked activation
  (:class:`~sae_bioemu.bioemu.steering.BioEmuSteeringHook`). ``rho_max * RMS(h)`` in bf16
  carries about three significant decimal digits, which would move the realised residual
  fraction that D13's penalty is defined on. Note what this does *not* buy: the residual is
  still applied into a bf16 tensor, and bf16's spacing at an activation magnitude of order 1
  is about 4e-3 against a measured residual fraction near 6e-4, so a large share of a small
  intervention is rounded away at the injection point. The hook therefore **re-measures the
  residual after that cast** and reports what the model received rather than what the adapter
  intended (``BioEmuSteeringHook._realised_output``);
* the **property surrogate**, which is handed float32 coordinates and promotes to float64
  internally (:func:`~sae_bioemu.steering.surrogate.differentiable_radius_of_gyration`).
  Autocast never downcasts float64 -- ``at::autocast::is_eligible`` excludes ``kDouble`` --
  and none of the reduction ops Rg uses are autocast-listed, so the promotion holds; the
  float32 entry point is what keeps the *returned* Rg out of bf16.

The seeded prior draw is likewise unaffected: ``torch.randn`` and the SO(3) table lookups are
factory and indexing ops, which autocast does not touch, so
:class:`~sae_bioemu.bioemu.extractor.SeededPriorSampling` draws the same bits and its
ambient-RNG assertion still holds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

#: The default mode. Selecting it must leave every backend flag untouched.
DEFAULT_GENERATION_PRECISION = "fp32"

#: Every accepted ``bioemu_pytorch.precision`` value, in increasing order of approximation.
GENERATION_PRECISION_MODES: tuple[str, ...] = ("fp32", "tf32", "bf16")

#: How the steering adapter is run when generation is reduced. Recorded in the identity so a
#: checkpoint states which arithmetic produced its interventions, not only which produced its
#: structures.
ADAPTER_PRECISION_POLICY = "float32_adapter_with_autocast_disabled"

#: How the property is measured when generation is reduced: coordinates are promoted to
#: float32 before the surrogate, which then promotes to float64 internally.
PROPERTY_PRECISION_POLICY = "float32_coordinates_float64_property"

_AUTOCAST_DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16}


@dataclass(frozen=True)
class GenerationPrecision:
    """One resolved generation precision mode and everything it implies."""

    mode: str

    def __post_init__(self) -> None:
        if self.mode not in GENERATION_PRECISION_MODES:
            raise ValueError(
                "bioemu_pytorch.precision must be one of "
                f"{', '.join(GENERATION_PRECISION_MODES)}; got {self.mode!r}"
            )

    @property
    def is_default(self) -> bool:
        """True for ``fp32``, the mode that touches nothing."""
        return self.mode == DEFAULT_GENERATION_PRECISION

    @property
    def allows_tf32(self) -> bool:
        """True when the scope enables the TF32 matmul and cuDNN paths."""
        return not self.is_default

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """The autocast dtype the denoiser call runs under, or ``None`` for no autocast."""
        return _AUTOCAST_DTYPES.get(self.mode)

    def identity(self) -> dict[str, object]:
        """Return the auditable record of what this mode changes."""
        dtype = self.autocast_dtype
        return {
            "mode": self.mode,
            "allow_tf32": self.allows_tf32,
            "autocast_dtype": None if dtype is None else str(dtype),
            "adapter_policy": ADAPTER_PRECISION_POLICY,
            "property_policy": PROPERTY_PRECISION_POLICY,
            # Stated explicitly because it is the question a reader asks first: reduced
            # precision is not reduced determinism, and the deterministic-algorithm and
            # cuDNN settings ``seed_everything`` applies are untouched by every mode.
            "deterministic": True,
        }


def resolve_generation_precision(value: object) -> GenerationPrecision:
    """Resolve a configured precision value, defaulting to ``fp32`` when absent.

    ``None`` is the absent case and is not a synonym for something else: a configuration
    that omits ``precision`` is a configuration that runs in FP32.
    """
    if isinstance(value, GenerationPrecision):
        return value
    if value is None:
        return GenerationPrecision(DEFAULT_GENERATION_PRECISION)
    if not isinstance(value, str):
        raise ValueError(
            "bioemu_pytorch.precision must be a string naming one of "
            f"{', '.join(GENERATION_PRECISION_MODES)}; got {value!r}"
        )
    return GenerationPrecision(value.strip().lower())


@contextmanager
def generation_precision_scope(
    precision: GenerationPrecision, device_type: str = "cuda"
) -> Iterator[None]:
    """Apply one precision mode for the duration of a generation, then restore it.

    Scoped rather than global on purpose: the TF32 backend flags are process-wide, so
    setting them once at start-up would silently apply to every later matmul in the process
    -- including an evaluation, a diagnostic, or an SAE forward pass that recorded no such
    mode. Restoration runs in ``finally``, so a failed generation cannot leak the setting
    into the retry that D7's ``maximum_unit_attempts`` performs.

    ``fp32`` returns without reading or writing anything at all.
    """
    if not isinstance(precision, GenerationPrecision):
        raise TypeError("precision must be a GenerationPrecision")
    if precision.is_default:
        yield
        return
    if not isinstance(device_type, str) or not device_type.strip():
        raise ValueError("device_type must be a non-empty string")
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    dtype = precision.autocast_dtype
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if dtype is None:
            yield
        else:
            with torch.autocast(device_type, dtype=dtype):
                yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def measurement_coordinates(
    coordinates: torch.Tensor, precision: GenerationPrecision
) -> torch.Tensor:
    """Return generated coordinates in a dtype the property surrogate may measure.

    float32 and float64 pass through untouched, so ``fp32`` and ``tf32`` -- which never
    change an output dtype -- are byte-identical. Under autocast the denoiser can return the
    autocast dtype; promoting it to float32 here keeps the *measurement* exact while leaving
    the *generation* reduced, which is the intended trade. Any other dtype, or a reduced
    dtype under a mode that declares no autocast, fails closed rather than being promoted:
    that would be an unexplained mismatch of exactly the kind D7 shows the cost of hiding.
    """
    if not isinstance(coordinates, torch.Tensor) or not coordinates.is_floating_point():
        raise TypeError("coordinates must be a floating tensor")
    if coordinates.dtype in (torch.float32, torch.float64):
        return coordinates
    if coordinates.dtype is not precision.autocast_dtype:
        raise ValueError(
            f"BioEmu generated coordinates have dtype {coordinates.dtype}, which precision "
            f"mode {precision.mode!r} does not produce"
        )
    return coordinates.to(torch.float32)
