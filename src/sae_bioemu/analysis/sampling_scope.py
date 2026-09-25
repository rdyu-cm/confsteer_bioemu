"""Typed structure-sampling semantics and scientific safety guards."""

from __future__ import annotations

from enum import Enum


class StructureSamplingScope(str, Enum):
    """Coordinate or target scope attached to structural analyses."""

    BIOEMU_32 = "bioemu_32"
    BIOEMU_128 = "bioemu_128"
    ALPHAFOLD3_STOCHASTIC = "alphafold3_stochastic"
    SINGLE_STRUCTURE = "single_structure"
    EXTERNAL_TARGET = "external_target"


def require_equilibrium_source(scope: StructureSamplingScope) -> None:
    """Reject coordinate sources that cannot define equilibrium properties."""

    if scope is StructureSamplingScope.EXTERNAL_TARGET:
        raise ValueError("an external target does not provide coordinates")
    if scope not in {StructureSamplingScope.BIOEMU_32, StructureSamplingScope.BIOEMU_128}:
        raise ValueError(f"{scope.value} is not an equilibrium ensemble")
