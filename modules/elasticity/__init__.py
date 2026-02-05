"""
Linear Elasticity Module
========================

Discontinuous Galerkin (DG) solver for linear elasticity problems.

This module implements a P1 DG solver for:
    -div(σ(u)) = f

where σ(u) is the Cauchy stress tensor defined by Lamé's law:
    σ(u) = λ(div u)I + 2μ ε(u)

with λ, μ being the Lamé parameters and ε(u) the strain tensor:
    ε(u) = 0.5(∇u + ∇u^T)

Uses SIPG (Symmetric Interior Penalty Galerkin) method for discretization.

Classes
-------
P1DGLinearElasticitySolver
    Main solver class for linear elasticity with DG method
"""

from .elasticity_DG import P1DGLinearElasticitySolver

__all__ = ['P1DGLinearElasticitySolver']
