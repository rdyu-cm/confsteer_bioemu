"""Shared, serializable architecture for bounded steering adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real

_STANDARDIZATION_KEYS = frozenset({"target_mean", "target_scale"})
_BASE_PAYLOAD_KEYS = frozenset({"activation_width", "hidden_widths", "rho_max", "target_kind"})
#: Absolute and directional payloads keep their exact historical key set, because the
#: AF3 protocol and the standalone JAX adapter compare payload keys with frozenset
#: equality. A relative-sequence adapter takes an already dimensionless input (D1), so
#: it omits the two standardization constants entirely rather than carrying them pinned
#: to the identity; AF3 correctly rejects that payload today.
_PAYLOAD_KEYS = _BASE_PAYLOAD_KEYS | _STANDARDIZATION_KEYS
_RELATIVE_PAYLOAD_KEYS = _BASE_PAYLOAD_KEYS
_TARGET_KINDS = frozenset({"absolute_sequence", "directional_residue", "relative_sequence"})
_RELATIVE_TARGET_KIND = "relative_sequence"

#: How the requested level reaches the hidden stack. Declared here rather than in
#: :mod:`sae_bioemu.steering.adapter` because that module imports this one; the adapter
#: keeps the runtime copy used to build its layers. ``branch`` genuinely changes the tensor
#: shapes -- the level gets its own linear pathway concatenated into the second hidden
#: layer -- so it is not interchangeable with the three that only change the arithmetic.
#: ``none`` is the UNCONDITIONED mode (D-101). The level reaches no part of the network, so the
#: adapter's only input is the hidden activations and one arm is trained PER DIRECTION. It exists
#: because every conditioned mode measured EVEN in the level once trained -- ``||odd||/||even||``
#: 0.0133 on cryptic and 0.0099 on domain motion, i.e. ~1% of the response is directional -- and
#: because forcing oddness by multiplying the residual by the level is what D2 forbids on
#: ``RELATIVE_SEQUENCE``. Like ``branch`` it changes tensor shapes: it drops the level column from
#: the first layer, so an adapter built with it cannot load a ``concat`` checkpoint.
LEVEL_CONDITIONING_MODES = ("concat", "embedding", "film", "branch", "none")
#: The original geometry, and the only value an existing payload can have implied.
DEFAULT_LEVEL_CONDITIONING = "concat"
#: Keys a payload may carry beyond its exact required set. Each is emitted only when it is
#: not the default, so every payload written before the field existed stays byte-identical
#: and its ``portable_architecture`` checkpoint identity does not move -- which is the whole
#: reason the set exists, given that AF3 and the standalone JAX adapter compare payload keys
#: with frozenset equality.
#:
#: ``site_conditioning`` was RESERVED here before it existed, precisely so that landing arms
#: A/B needed no widening of this frozenset while radius-of-gyration arms were running. It is
#: now IMPLEMENTED: :class:`AdapterArchitecture` parses and emits it, and the adapter builds
#: the third hidden-layer branch it names. It is still emitted only when it is not ``None``,
#: so every payload written before the field existed stays byte-identical.
_OPTIONAL_PAYLOAD_KEYS = frozenset({"level_conditioning", "site_conditioning", "residual_cap"})

#: Optional keys that are declared but not yet parseable. Reserving a key and implementing it
#: are two different acts; conflating them is how a payload comes to mean less than it says.
#: Empty since ``site_conditioning`` graduated. The mechanism stays because the next
#: architecture will want the same two-step, and because an empty set here is what makes
#: "declared and buildable" auditable rather than assumed.
_RESERVED_PAYLOAD_KEYS: frozenset[str] = frozenset()

#: Where the per-residue site channel ``s_i`` comes from, for cryptic-pocket arms A and B.
#: These are DATA sources, not shapes: all three build the identical ``2H -> 3H`` second
#: hidden layer and differ only in what fills the channel. They are nonetheless part of the
#: architecture identity, because two checkpoints trained on a binary mask and on a predicted
#: probability are different experiments and must never write byte-identical identities --
#: the hole ``lambda_l`` and ``level_gain`` each sat in.
#:
#: * ``pocket_binary``         arm A_binary: the annotation as a strict ``{0, 1}`` indicator.
#: * ``pocket_smoothed``       arm A_smooth: the same annotation as a Gaussian spatial field.
#: * ``predicted_probability`` arm B: PocketMiner's per-residue probability, unthresholded.
#:
#: ``sae_bioemu.cryptic_steering.site`` is the single producer keyed by these same names, and
#: a test asserts the two sets are equal: a name the adapter accepts and nothing can build is
#: an arm that cannot be run, and a name a builder emits and the adapter refuses is worse.
SITE_CONDITIONING_MODES = ("pocket_binary", "pocket_smoothed", "predicted_probability")

#: How the adapter's pre-cap request becomes the emitted residual.
#:
#: * ``tanh``  the historical bounded residual: ``residual_fraction = rho_max * tanh(r)`` for a
#:   pre-cap ratio ``r = RMS_m(raw) / (rho_max * RMS_m(h))``. Bounded by construction, and the
#:   DEFAULT -- every adapter ever trained is this one and stays byte-identical under it.
#: * ``none``  no cap at all: ``residual = raw`` and ``residual_fraction = RMS_m(raw)/RMS_m(h)``,
#:   unbounded. ``rho_max`` stops bounding anything (it keeps its OTHER job, normalising the
#:   ``fraction_of_rho_max`` parsimony term), and the only thing left holding the dose down is
#:   ``bioemu_pytorch.validation_residual_fraction_maximum``, which is therefore REQUIRED under
#:   this mode rather than optional.
#:
#: Why an "off" exists at all: ``rho_max * tanh(r)`` approaches ``rho_max`` asymptotically and its
#: derivative dies with it. At the realised fraction 0.09999966 against ``rho_max = 0.10`` that two
#: paired arms trained to, ``d(fraction)/dr`` is attenuated about 47,000x relative to the small-r
#: slope, so the adapter was on a flat shelf and the optimizer had nothing to descend. Turning the
#: cap OFF is a different experiment from raising ``rho_max``, and the two must never be confused,
#: which is why this is an identity-bearing mode and not a number.
RESIDUAL_CAP_MODES = ("tanh", "none")
#: The historical behaviour, and the only value an existing payload can have implied.
DEFAULT_RESIDUAL_CAP = "tanh"


#: Keys a workflow config's ``adapter`` block must NOT carry, and where each belongs instead.
#: That block is not key-set-checked -- every radius-of-gyration config predates such a check and
#: adding one would refuse them all -- so an unknown key there is silently ignored. That is
#: tolerable for a typo in a knob whose absence raises anyway, and NOT tolerable here: the site
#: source lives in the ARM config, so writing it in the runtime block would look configured, be
#: dropped, and run the unconditioned arm C under an arm A or B name.
_ADAPTER_BLOCK_ELSEWHERE = {
    "site_conditioning": (
        "the cryptic arm config's 'site.source'. Which channel the adapter is conditioned on is "
        "a property of the experiment rather than of a GPU job, and the arm config is also what "
        "resolves the fields themselves; declaring it in both files would create a pair that can "
        "disagree, and the conditioned arms share every tensor shape so nothing would catch it"
    ),
}


def reject_misplaced_adapter_keys(values: Mapping[str, object]) -> None:
    """Fail closed on an ``adapter`` block key that belongs to another file."""
    for key, destination in sorted(_ADAPTER_BLOCK_ELSEWHERE.items()):
        if key in values:
            raise ValueError(
                f"adapter.{key} is not read from the workflow config; it belongs in {destination}"
            )


def payload_keys(target_kind: str) -> frozenset[str]:
    """Return the exact required payload key set for one target kind.

    ``level_conditioning`` is not included: it is an optional key that a payload carries
    only when it is not ``concat`` (see :meth:`AdapterArchitecture.payload`).
    """
    if target_kind not in _TARGET_KINDS:
        raise ValueError("target_kind is not an approved adapter target kind")
    return _RELATIVE_PAYLOAD_KEYS if target_kind == _RELATIVE_TARGET_KIND else _PAYLOAD_KEYS


@dataclass(frozen=True)
class AdapterArchitecture:
    """The approved MLP shape and training-only target normalization."""

    activation_width: int
    hidden_widths: tuple[int, ...] = (64, 64)
    rho_max: float = 0.10
    target_mean: float = 0.0
    target_scale: float = 1.0
    target_kind: str = "absolute_sequence"
    #: Last, so every existing positional construction keeps its meaning.
    level_conditioning: str = DEFAULT_LEVEL_CONDITIONING
    #: ``None`` is arm C and every radius-of-gyration adapter: no site channel at all. A name
    #: from :data:`SITE_CONDITIONING_MODES` adds the third second-layer branch and says which
    #: source fills it. Appended after ``level_conditioning`` for the same reason that one was
    #: appended last.
    site_conditioning: str | None = None
    #: ``tanh`` is every adapter ever trained: ``rho_max`` is a genuine bound. ``none`` removes
    #: the cap entirely, which makes ``rho_max`` above a parsimony-term scale rather than a
    #: bound -- so it travels with the payload, appended last for the same reason the two above
    #: were. A consumer that reimplements the cap (the AF3 protocol, the standalone JAX adapter)
    #: compares payload keys with frozenset equality and will therefore REFUSE an uncapped
    #: payload outright, which is the correct answer: it cannot run one.
    residual_cap: str = DEFAULT_RESIDUAL_CAP

    def __post_init__(self) -> None:
        if (
            not isinstance(self.activation_width, int)
            or isinstance(self.activation_width, bool)
            or self.activation_width <= 0
        ):
            raise ValueError("activation_width must be a positive integer")
        if self.hidden_widths not in ((64, 64), (128, 128, 128, 128)):
            raise ValueError("hidden_widths must be an approved primary or capacity shape")
        if any(
            not isinstance(width, int) or isinstance(width, bool) for width in self.hidden_widths
        ):
            raise ValueError("hidden_widths must contain integers")
        if not _finite_real(self.rho_max) or not 0 < self.rho_max <= 1:
            raise ValueError("rho_max must be finite and in (0, 1]")
        if not _finite_real(self.target_mean):
            raise ValueError("target_mean must be finite")
        if not _finite_real(self.target_scale) or self.target_scale <= 0:
            raise ValueError("target_scale must be positive and finite")
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError(
                "target_kind must be absolute_sequence, directional_residue, or relative_sequence"
            )
        if self.target_kind == _RELATIVE_TARGET_KIND and (
            self.target_mean != 0.0 or self.target_scale != 1.0
        ):
            raise ValueError(
                "relative_sequence target_kind applies no normalization constants, so "
                "target_mean and target_scale must stay at the identity and are omitted "
                "from its payload"
            )
        if self.level_conditioning not in LEVEL_CONDITIONING_MODES:
            raise ValueError(f"level_conditioning must be one of {LEVEL_CONDITIONING_MODES}")
        if self.site_conditioning is not None:
            if self.site_conditioning not in SITE_CONDITIONING_MODES:
                raise ValueError(f"site_conditioning must be one of {SITE_CONDITIONING_MODES}")
            if self.level_conditioning != "branch":
                # The site channel exists only as a second-layer branch, so outside ``branch``
                # there is no place to put it. Accepting the name anyway would record an
                # architecture the adapter did not build.
                raise ValueError(
                    "site_conditioning requires level_conditioning='branch': the site channel "
                    "is a second-layer branch and no other mode has a second-layer join"
                )
        if self.residual_cap not in RESIDUAL_CAP_MODES:
            raise ValueError(f"residual_cap must be one of {RESIDUAL_CAP_MODES}")

    @classmethod
    def primary(cls, activation_width: int = 1, **kwargs: float) -> AdapterArchitecture:
        """Return the two-layer primary architecture."""
        return cls(activation_width=activation_width, hidden_widths=(64, 64), **kwargs)

    @classmethod
    def capacity_ablation(cls, activation_width: int = 1, **kwargs: float) -> AdapterArchitecture:
        """Return the approved four-layer capacity ablation."""
        return cls(activation_width=activation_width, hidden_widths=(128, 128, 128, 128), **kwargs)

    @classmethod
    def from_payload(cls, payload: object) -> AdapterArchitecture:
        """Parse the exact primitive JSON representation used by AF3 workers."""
        if not isinstance(payload, Mapping):
            raise ValueError("adapter architecture payload keys are invalid")
        kind = payload.get("target_kind")
        if not isinstance(kind, str) or kind not in _TARGET_KINDS:
            raise ValueError("adapter architecture payload keys are invalid")
        if frozenset(payload) - _OPTIONAL_PAYLOAD_KEYS != payload_keys(kind):
            raise ValueError("adapter architecture payload keys are invalid")
        reserved = sorted(frozenset(payload) & _RESERVED_PAYLOAD_KEYS)
        if reserved:
            raise ValueError(
                f"adapter architecture payload carries reserved key(s) {reserved} that this "
                "class cannot yet build; parsing it would silently return an adapter without "
                "the conditioning the payload declares"
            )
        site_conditioning = payload.get("site_conditioning")
        if site_conditioning is not None and not isinstance(site_conditioning, str):
            raise ValueError("site_conditioning must be a source name or absent")
        # Absent means ``tanh``: a payload written before the cap could be turned off can only
        # have described the capped adapter, so every checkpoint on disk stays readable and its
        # ``portable_architecture`` digest does not move.
        residual_cap = payload.get("residual_cap", DEFAULT_RESIDUAL_CAP)
        if not isinstance(residual_cap, str):
            raise ValueError("residual_cap must be a mode name or absent")
        hidden_widths = payload["hidden_widths"]
        if not isinstance(hidden_widths, list):
            raise ValueError("hidden_widths must use the canonical JSON list representation")
        standardization = (
            {}
            if kind == _RELATIVE_TARGET_KIND
            else {
                "target_mean": payload["target_mean"],
                "target_scale": payload["target_scale"],
            }
        )
        try:
            return cls(
                activation_width=payload["activation_width"],
                hidden_widths=tuple(hidden_widths),
                rho_max=payload["rho_max"],
                target_kind=kind,
                # Absent means "concat": a payload written before the field existed can only
                # have described the original geometry, so old checkpoints stay readable.
                level_conditioning=payload.get("level_conditioning", DEFAULT_LEVEL_CONDITIONING),
                # Absent means "no site channel", which is what every payload written before
                # arms A/B existed described, and what arm C describes today.
                site_conditioning=site_conditioning,
                residual_cap=residual_cap,
                **standardization,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("adapter architecture payload is invalid") from error

    def payload(self) -> dict[str, object]:
        """Return primitive values suitable for checkpoint/runtime serialization.

        A relative-sequence payload omits ``target_mean``/``target_scale`` entirely:
        they are never applied on that path, and carrying them would invite a stale
        standardization constant to ride along in the checkpoint identity.

        ``level_conditioning`` appears only when it is not ``concat``, and
        ``site_conditioning`` only when it is not ``None``, for the same reason. Emitting
        either always would rewrite every historical payload and, through
        ``portable_architecture``, move the checkpoint identity of every adapter already
        trained; it would also be rejected by the AF3 protocol and the standalone JAX adapter,
        which compare payload keys with frozenset equality against the historical set.
        """
        keys = payload_keys(self.target_kind)
        payload = {key: value for key, value in asdict(self).items() if key in keys}
        payload["hidden_widths"] = list(self.hidden_widths)
        if self.level_conditioning != DEFAULT_LEVEL_CONDITIONING:
            payload["level_conditioning"] = self.level_conditioning
        if self.site_conditioning is not None:
            payload["site_conditioning"] = self.site_conditioning
        if self.residual_cap != DEFAULT_RESIDUAL_CAP:
            payload["residual_cap"] = self.residual_cap
        return payload


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
