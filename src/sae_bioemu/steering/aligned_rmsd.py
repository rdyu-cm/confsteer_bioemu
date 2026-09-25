"""A differentiable, superposition-invariant RMSD: the one kernel two programmes will share.

The cryptic and domain-motion direct-metric objectives both want to say "move THIS structure
towards THAT one" in Angstrom, and both have to say it about a quantity that does not change
when the generated structure is translated or rotated. BioEmu's denoiser produces coordinates
in an arbitrary global frame, so a bare coordinate MSE against a reference is dominated by the
rigid-body offset between the two frames -- a gradient that spends its budget rotating the whole
protein rather than changing its shape. The contrast case in
``tests/test_steering_aligned_rmsd.py`` measures exactly that on a 214-residue system: 99% of an
unaligned MSE's gradient energy lies in the six rigid-body directions when the two frames
differ, and 41% even when they already agree and the only difference is a domain hinge.

This module is the fix, and it is deliberately ONE file with no property's data in it, hashed in
its own code group (``aligned_rmsd_shared``) so that a change to the kernel re-keys the two
properties that execute it and no others.

WHY THE ROTATION MAY BE DETACHED
--------------------------------

``kabsch_rotation`` computes ``R*`` and ``t*`` on ``mobile.detach()`` and ``reference.detach()``,
so no gradient flows through the SVD. That is not an approximation, and the reason is the
envelope theorem (Danskin). Write

    F(X) = min over (R, t) in SE(3) of g(X, R, t),
    g(X, R, t) = (1/W) * sum_i w_i * || R x_i + t - y_i ||^2 .

``(R*, t*)`` is an interior minimiser of ``g`` over the group, so ``dg/d(R, t) = 0`` there along
every tangent direction, and the total derivative collapses to the partial one:

    dF/dX = partial-g/partial-X evaluated at (R*, t*).

Differentiating with ``R*`` and ``t*`` held constant therefore yields the EXACT gradient of the
minimised quantity, not a truncation of it. The test measures this rather than assuming it: it
compares the detached-rotation gradient against autograd taken straight through
``torch.linalg.svd`` and finds a relative difference at the float64 floor.

The surviving gradient is

    dF/dX_i = (w_i / (W * RMSD)) * R*^T r_i,   r_i = R* x_i + t* - y_i,

and it is orthogonal to the rigid-body tangent space of X for two independent reasons, each of
which is one half of the Kabsch optimality condition:

* **Translation optimality** gives ``sum_i w_i r_i = 0``, so ``sum_i dF/dX_i = 0``. An
  infinitesimal translation of the whole structure changes nothing: there is no net force.
* **Rotation optimality** gives ``sum_i w_i (R* x_i) x r_i = 0``. Since ``R^T(u x v) =
  (R^T u) x (R^T v)`` for ``R`` in SO(3), this is the same statement as
  ``sum_i w_i x_i' x (R*^T r_i) = 0``: no net torque about the weighted centroid.

So the gradient points purely along shape change. That is the property the direct-metric
objectives need, and ``rigid_body_leak`` is what measures it.

THE ALIGNMENT SET AND THE METRIC SET MUST BE THE SAME SET
---------------------------------------------------------

The envelope argument above needs ``(R*, t*)`` to minimise THE SAME functional that is being
differentiated. If a caller superposes on one atom set and measures on another, ``R*`` minimises
the alignment set's residual while the metric reads the other set's, the stationarity condition
does not apply to the quantity being differentiated, and both conclusions fail at once:

* detaching ``R*`` becomes WRONG rather than free -- the true derivative now has a term through
  ``R*(X)``; and
* that term places gradient on atoms that carry no metric term at all. Measured on the same
  214-residue system, aligning on residues 1-128 while measuring on residues 150-214 puts 53% of
  the gradient norm on residues the objective never scores, leaves 99% of the energy inside the
  metric region pointing along that region's own rigid-body directions, and makes the detached
  rotation wrong by 61% rather than by 3e-15.

``aligned_rmsd`` therefore REFUSES a distinct alignment set by name rather than silently
returning a quantity whose gradient means something else. ``alignment_weights`` exists only so
that refusal has somewhere to fire: the legal value is the metric weight set itself. A caller
that genuinely wants a superposition on one region and a measurement on another is asking for a
different estimator and must build it, and argue for it, somewhere else.

WHAT THE EPSILON GUARDS COVER
-----------------------------

Two degeneracies are possible in a 3x3 Kabsch solve, and they are not the same problem.

1. **A numerically zero determinant.** The reflection correction is applied as a ``where`` on the
   sign rather than as ``sign(det)``, which would return 0 for a determinant at the rounding
   floor and hand back a singular matrix that is not a rotation at all.
2. **Coincident singular values.** When the two smallest singular values of the covariance
   coincide, the minimising rotation is NOT UNIQUE: there is a one-parameter family of rotations
   achieving the same optimum, and which member the SVD returns depends on the LAPACK driver.
   The RMSD VALUE is still correct and still the true minimum, and because ``R*`` is detached the
   gradient computed with any member of that family is a valid subgradient of the (here
   non-differentiable) minimum. Autograd THROUGH the SVD is what would blow up in this case, by
   dividing by the vanishing singular-value gap; detaching is what makes it safe. This is
   reported by :func:`rotation_degeneracy` rather than refused, because a refusal would reject a
   configuration whose value and subgradient are both correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "DEGENERACY_EPS",
    "RigidBodyLeak",
    "aligned_rmsd",
    "kabsch_rotation",
    "rigid_body_leak",
    "rotation_degeneracy",
]

#: Relative floor for the weight-sum, determinant-sign and singular-gap guards. The superposition
#: is solved in float64 whatever the input dtype, so this sits far above the arithmetic floor and
#: far below any separation a real structure produces.
DEGENERACY_EPS = 1e-12

#: Dtype every superposition is solved in. A 3x3 SVD costs nothing next to the denoiser, and
#: solving it in float32 would put the rotation's own error above the shape signal the objective
#: is trying to read.
_WORK_DTYPE = torch.float64


def _is_batched(coordinates: Tensor, name: str) -> bool:
    """Return whether ``coordinates`` carries a leading batch dimension, type-checking first."""
    if not isinstance(coordinates, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, not {type(coordinates).__name__}")
    return coordinates.ndim == 3


def _as_batch(coordinates: Tensor, name: str) -> Tensor:
    """Return ``coordinates`` as ``(B, N, 3)``, validating shape and finiteness."""
    if not isinstance(coordinates, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, not {type(coordinates).__name__}")
    if coordinates.ndim not in (2, 3) or coordinates.shape[-1] != 3:
        raise ValueError(
            f"{name} must be (N, 3) Cartesian coordinates or a (B, N, 3) batch of them; got "
            f"{tuple(coordinates.shape)}"
        )
    if coordinates.shape[-2] < 3:
        raise ValueError(
            f"{name} must carry at least three atoms for a superposition to be determined; got "
            f"{coordinates.shape[-2]}"
        )
    if not torch.isfinite(coordinates.detach()).all():
        raise ValueError(f"{name} carries non-finite coordinates")
    return coordinates if coordinates.ndim == 3 else coordinates.unsqueeze(0)


def _as_weights(weights: Tensor | None, batch: int, atoms: int, like: Tensor) -> Tensor:
    """Return per-atom weights as a finite, non-negative ``(B, N)`` tensor in the work dtype.

    ``None`` means every atom counts once, which is the ordinary whole-chain case. A boolean
    mask is accepted and is the natural spelling of "measure over this region".
    """
    if weights is None:
        return torch.ones((batch, atoms), dtype=_WORK_DTYPE, device=like.device)
    if not isinstance(weights, Tensor):
        raise TypeError(f"weights must be a torch.Tensor or None, not {type(weights).__name__}")
    values = weights.detach().to(dtype=_WORK_DTYPE, device=like.device)
    if values.ndim == 1:
        values = values.unsqueeze(0).expand(batch, -1)
    if values.shape != (batch, atoms):
        raise ValueError(
            f"weights must be ({atoms},) or ({batch}, {atoms}) to match the coordinates; got "
            f"{tuple(weights.shape)}"
        )
    if not torch.isfinite(values).all():
        raise ValueError("weights carry non-finite entries")
    if bool((values < 0.0).any()):
        raise ValueError("weights must be non-negative")
    return values


def _output_dtype(mobile: Tensor, reference: Tensor) -> torch.dtype:
    """Return the dtype the rotation is handed back in.

    Promoting the two inputs keeps the caller's arithmetic in the caller's precision. Half
    precision is promoted to float32 because returning a float16 rotation would quantise the
    superposition itself, which is a property of the pair and not of the denoiser's storage.
    """
    promoted = torch.promote_types(mobile.dtype, reference.dtype)
    return torch.float32 if promoted in (torch.float16, torch.bfloat16) else promoted


def _centred(
    mobile: Tensor, reference: Tensor, weights: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the weighted centroids and the centred, detached coordinate sets."""
    total = weights.sum(dim=-1)
    if bool((total <= eps).any()):
        raise ValueError(
            "the weights of at least one structure sum to zero, so neither the superposition nor "
            "the metric is defined over any atom"
        )
    if int((weights > 0.0).sum(dim=-1).min()) < 3:
        raise ValueError(
            "a Kabsch superposition needs at least three atoms with positive weight; fewer leaves "
            "the rotation undetermined about the remaining axis"
        )
    mobile_centroid = torch.einsum("bn,bni->bi", weights, mobile) / total.unsqueeze(-1)
    reference_centroid = torch.einsum("bn,bni->bi", weights, reference) / total.unsqueeze(-1)
    return (
        mobile - mobile_centroid.unsqueeze(-2),
        reference - reference_centroid.unsqueeze(-2),
        mobile_centroid,
        reference_centroid,
    )


def kabsch_rotation(
    mobile: Tensor,
    reference: Tensor,
    weights: Tensor | None = None,
    *,
    eps: float = DEGENERACY_EPS,
) -> tuple[Tensor, Tensor]:
    """Return the rotation and translation that superpose ``mobile`` onto ``reference``.

    ``R, t = kabsch_rotation(X, Y)`` minimise ``sum_i w_i || R x_i + t - y_i ||^2`` over SE(3),
    with ``t = centroid(Y) - R centroid(X)`` so that ``R x + t`` maps the weighted centroid of
    ``X`` onto that of ``Y``.

    **Both returned tensors are detached and carry no gradient**, and that is the point rather
    than a limitation: see the module docstring. The superposition is solved on
    ``mobile.detach()`` and ``reference.detach()`` in float64 and returned in the promoted input
    dtype.

    The reflection case is handled the way Kabsch's paper handles it -- the sign of the singular
    vector belonging to the SMALLEST singular value is flipped, which is the cheapest departure
    from the improper optimum -- and the sign is taken through a comparison rather than through
    ``sign``, so a determinant at the rounding floor cannot produce a singular matrix. When the
    two smallest singular values coincide the minimiser is not unique; the value is unaffected
    and :func:`rotation_degeneracy` reports the gap.

    Accepts ``(N, 3)`` or ``(B, N, 3)`` and answers in kind.
    """
    batched = _is_batched(mobile, "mobile")
    x = _as_batch(mobile, "mobile").detach().to(_WORK_DTYPE)
    y = _as_batch(reference, "reference").detach().to(_WORK_DTYPE)
    if x.shape != y.shape:
        raise ValueError(
            f"mobile and reference must have the same shape; got {tuple(mobile.shape)} and "
            f"{tuple(reference.shape)}. An aligned RMSD is defined between corresponding atoms"
        )
    w = _as_weights(weights, x.shape[0], x.shape[1], x)
    mobile_centred, reference_centred, mobile_centroid, reference_centroid = _centred(x, y, w, eps)

    covariance = torch.einsum("bn,bni,bnj->bij", w, mobile_centred, reference_centred)
    left, _, right_transpose = torch.linalg.svd(covariance)
    # ``R = V diag(1, 1, d) U^T`` maximises ``tr(R H)`` over SO(3) for ``H = U S V^T``.
    right = right_transpose.transpose(-2, -1)
    proper = right @ left.transpose(-2, -1)
    determinant = torch.linalg.det(proper)
    # A ``where`` rather than ``sign``: ``sign(0.0)`` is 0 and would return a singular matrix.
    flip = torch.where(
        determinant < 0.0, -torch.ones_like(determinant), torch.ones_like(determinant)
    )
    correction = torch.eye(3, dtype=_WORK_DTYPE, device=x.device).expand(x.shape[0], 3, 3).clone()
    correction[:, 2, 2] = flip
    rotation = right @ correction @ left.transpose(-2, -1)
    translation = reference_centroid - torch.einsum("bij,bj->bi", rotation, mobile_centroid)

    dtype = _output_dtype(mobile, reference)
    rotation = rotation.to(dtype)
    translation = translation.to(dtype)
    if batched:
        return rotation, translation
    return rotation.squeeze(0), translation.squeeze(0)


def rotation_degeneracy(
    mobile: Tensor,
    reference: Tensor,
    weights: Tensor | None = None,
    *,
    eps: float = DEGENERACY_EPS,
) -> Tensor:
    """Return ``(s1 - s2) / s0``, the relative gap that decides whether ``R*`` is unique.

    A gap at or below :data:`DEGENERACY_EPS` means the two smallest singular values of the
    covariance coincide and a one-parameter family of rotations attains the same optimum. The
    RMSD value is unaffected, and with the rotation detached the gradient is a valid subgradient
    of that minimum; this is reported so a caller can say so, not refused.
    """
    batched = _is_batched(mobile, "mobile")
    x = _as_batch(mobile, "mobile").detach().to(_WORK_DTYPE)
    y = _as_batch(reference, "reference").detach().to(_WORK_DTYPE)
    if x.shape != y.shape:
        raise ValueError(
            f"mobile and reference must have the same shape; got {tuple(mobile.shape)} and "
            f"{tuple(reference.shape)}"
        )
    w = _as_weights(weights, x.shape[0], x.shape[1], x)
    mobile_centred, reference_centred, _, _ = _centred(x, y, w, eps)
    covariance = torch.einsum("bn,bni,bnj->bij", w, mobile_centred, reference_centred)
    singular = torch.linalg.svdvals(covariance)
    gap = (singular[:, 1] - singular[:, 2]) / singular[:, 0].clamp_min(eps)
    return gap if batched else gap.squeeze(0)


def _require_one_set(weights: Tensor | None, alignment_weights: Tensor | None) -> None:
    """Refuse an alignment set that differs from the metric set.

    Not caution: the envelope argument that makes the detached rotation exact, and the
    orthogonality that makes the gradient pure shape change, are both statements about a single
    functional. Two sets and the gradient acquires a term through ``R*(X)`` that lands on atoms
    the objective never scores. See the module docstring for the measured size of that leak.
    """
    if alignment_weights is None:
        return
    resolved = (
        torch.ones_like(alignment_weights, dtype=torch.float64)
        if weights is None
        else weights.detach().to(torch.float64)
    )
    candidate = alignment_weights.detach().to(torch.float64)
    if resolved.shape != candidate.shape or not bool(torch.equal(resolved, candidate)):
        raise ValueError(
            "aligned_rmsd refuses an alignment set that differs from the metric set. The "
            "superposition and the measurement must be the same weighted atom set: otherwise "
            "the rotation minimises one residual while the metric reads another, the envelope "
            "theorem no longer applies, detaching the rotation stops being exact, and the "
            "gradient places weight on atoms carrying no metric term. Pass the metric weights "
            "as alignment_weights, or omit it"
        )


def aligned_rmsd(
    mobile: Tensor,
    reference: Tensor,
    weights: Tensor | None = None,
    *,
    alignment_weights: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Return ``sqrt(weighted mean over atoms of || R* x + t* - y ||^2)``, differentiable in X.

    The superposition is solved on the SAME weighted atom set the metric is read over -- see
    :func:`_require_one_set` and the module docstring -- and ``R*``/``t*`` are detached, which
    the envelope theorem makes exact rather than approximate.

    ``eps`` is added inside the square root so that the derivative stays finite when the two
    structures coincide. At an RMSD of 1 Angstrom the default shifts the value by 5e-13
    Angstrom; at an RMSD of exactly 0 it reports 1e-6 Angstrom instead of 0.

    Accepts ``(N, 3)`` or ``(B, N, 3)`` and returns a scalar or a ``(B,)`` vector to match.
    """
    if not isinstance(eps, float) or eps < 0.0:
        raise ValueError("eps must be a non-negative float")
    _require_one_set(weights, alignment_weights)
    rotation, translation = kabsch_rotation(mobile, reference, weights, eps=DEGENERACY_EPS)

    batched = _is_batched(mobile, "mobile")
    x = _as_batch(mobile, "mobile")
    y = _as_batch(reference, "reference")
    if not batched:
        rotation = rotation.unsqueeze(0)
        translation = translation.unsqueeze(0)
    dtype = _output_dtype(mobile, reference)
    w = _as_weights(weights, x.shape[0], x.shape[1], x).to(dtype)

    placed = torch.einsum("bij,bnj->bni", rotation, x.to(dtype)) + translation.unsqueeze(-2)
    squared = (placed - y.to(dtype)).square().sum(dim=-1)
    mean_squared = (w * squared).sum(dim=-1) / w.sum(dim=-1)
    value = torch.sqrt(mean_squared + eps)
    return value if batched else value.squeeze(0)


@dataclass(frozen=True)
class RigidBodyLeak:
    """How much of a gradient points along rigid-body motion rather than shape change.

    Every entry is dimensionless and scale free, so the numbers compare across structures and
    across the arbitrary overall factor a loss weight applies.
    """

    #: Fraction of ``||g||^2`` lying in the six-dimensional rigid-body tangent space at the
    #: structure the gradient was taken at. 0 means pure shape change; 1 means the gradient does
    #: nothing but move the structure as a rigid body.
    energy_fraction: float
    #: ``|| sum_i g_i || / sum_i ||g_i||``: the net force, normalised so that a gradient pushing
    #: every atom the same way scores 1 and a balanced one scores 0.
    net_force: float
    #: ``|| sum_i (x_i - xbar) x g_i || / sum_i ||x_i - xbar|| ||g_i||``: the net torque about the
    #: centroid, normalised by the largest torque those lever arms and forces could produce.
    net_torque: float


def rigid_body_leak(gradient: Tensor, coordinates: Tensor) -> RigidBodyLeak:
    """Measure the rigid-body content of ``gradient`` at ``coordinates``.

    This is the instrument the scientific claim in the module docstring is checked with, and it
    is shipped rather than left in the test because the two objectives that will use this kernel
    should be able to report it beside their losses.

    The six rigid-body tangent directions at ``X`` are the three translations ``e_k`` and the
    three rotations ``e_k x (x_i - xbar)``. They are orthonormalised by QR rather than assumed
    orthogonal, because the rotation generators are not mutually orthogonal for a general
    structure and are orthogonal to the translations only about the unweighted centroid.
    """
    g = _as_batch(gradient, "gradient")
    x = _as_batch(coordinates, "coordinates")
    if g.shape != x.shape:
        raise ValueError(
            f"gradient and coordinates must have the same shape; got {tuple(gradient.shape)} and "
            f"{tuple(coordinates.shape)}"
        )
    if g.shape[0] != 1:
        raise ValueError("rigid_body_leak measures one structure at a time; got a batch")
    g = g[0].detach().to(_WORK_DTYPE)
    x = x[0].detach().to(_WORK_DTYPE)

    norm = torch.linalg.vector_norm(g)
    if float(norm) == 0.0:
        raise ValueError("a zero gradient has no direction to decompose")
    centred = x - x.mean(dim=0, keepdim=True)
    atoms = x.shape[0]
    eye = torch.eye(3, dtype=_WORK_DTYPE, device=x.device)
    translations = eye.unsqueeze(0).expand(atoms, 3, 3).reshape(atoms * 3, 3)
    rotations = torch.stack(
        [torch.linalg.cross(eye[axis].expand(atoms, 3), centred) for axis in range(3)], dim=-1
    ).reshape(atoms * 3, 3)
    basis = torch.cat([translations, rotations], dim=-1)
    orthonormal, _ = torch.linalg.qr(basis, mode="reduced")

    flat = g.reshape(-1)
    projected = orthonormal.transpose(0, 1) @ flat
    energy_fraction = float((projected.square().sum() / flat.square().sum()).clamp(0.0, 1.0))

    magnitudes = torch.linalg.vector_norm(g, dim=-1)
    force_scale = float(magnitudes.sum())
    net_force = float(torch.linalg.vector_norm(g.sum(dim=0)) / force_scale)
    lever = torch.linalg.vector_norm(centred, dim=-1)
    torque_scale = float((lever * magnitudes).sum())
    torque = torch.linalg.cross(centred, g).sum(dim=0)
    net_torque = float(torch.linalg.vector_norm(torque) / torque_scale) if torque_scale > 0 else 0.0
    return RigidBodyLeak(
        energy_fraction=energy_fraction, net_force=net_force, net_torque=net_torque
    )
