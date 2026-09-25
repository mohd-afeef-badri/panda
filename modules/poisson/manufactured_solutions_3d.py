"""Manufactured Poisson problems on the unit cube.

Every factory returns ``(u_exact, f, g, name)`` and satisfies
``-Delta u_exact = f`` with Dirichlet data ``g = u_exact``.  This mirrors the
2D :mod:`manufactured_solutions` interface.
"""

import numpy as np


def affine():
    """An exactly representable affine solution (patch test)."""

    def u_exact(x, y, z):
        return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z

    def f(x, y, z):
        return 0.0

    return u_exact, f, u_exact, "3D affine patch"


def smooth_sin():
    """Smooth trigonometric solution with homogeneous boundary data."""

    def u_exact(x, y, z):
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

    def f(x, y, z):
        return 3.0 * np.pi**2 * u_exact(x, y, z)

    return u_exact, f, u_exact, "3D smooth sine"


def polynomial():
    """Smooth degree-six product polynomial, zero on the unit-cube boundary."""

    def factor(value):
        return value * (1.0 - value)

    def u_exact(x, y, z):
        return factor(x) * factor(y) * factor(z)

    def f(x, y, z):
        return 2.0 * (
            factor(y) * factor(z)
            + factor(x) * factor(z)
            + factor(x) * factor(y)
        )

    return u_exact, f, u_exact, "3D polynomial"


def quadratic():
    """A non-homogeneous quadratic used to verify the P1 convergence rate."""

    def u_exact(x, y, z):
        return x**2 + y**2 + z**2

    def f(x, y, z):
        return -6.0

    return u_exact, f, u_exact, "3D quadratic"


def gaussian_peak(alpha=20.0, center=(0.5, 0.5, 0.5)):
    """A smooth localized Gaussian peak."""
    x0, y0, z0 = center

    def u_exact(x, y, z):
        radius_squared = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
        return np.exp(-alpha * radius_squared)

    def f(x, y, z):
        radius_squared = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
        return (6.0 * alpha - 4.0 * alpha**2 * radius_squared) * np.exp(
            -alpha * radius_squared
        )

    return u_exact, f, u_exact, "3D Gaussian peak"


def multiple_peaks(alpha=25.0):
    """Three Gaussian peaks at different locations in the cube."""
    centers = ((0.3, 0.3, 0.3), (0.7, 0.7, 0.65), (0.3, 0.7, 0.75))

    def u_exact(x, y, z):
        return sum(
            np.exp(-alpha * ((x - x0)**2 + (y - y0)**2 + (z - z0)**2))
            for x0, y0, z0 in centers
        )

    def f(x, y, z):
        result = 0.0
        for x0, y0, z0 in centers:
            radius_squared = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
            result += (6.0 * alpha - 4.0 * alpha**2 * radius_squared) * np.exp(
                -alpha * radius_squared
            )
        return result

    return u_exact, f, u_exact, "3D multiple peaks"


def boundary_layer(epsilon=0.08):
    """A steep x-directed layer modulated smoothly in y and z."""

    def u_exact(x, y, z):
        return (
            np.tanh((x - 0.15) / epsilon)
            * np.sin(np.pi * y)
            * np.sin(np.pi * z)
        )

    def f(x, y, z):
        scaled_x = (x - 0.15) / epsilon
        tanh_x = np.tanh(scaled_x)
        sech_squared = 1.0 / np.cosh(scaled_x)**2
        yz = np.sin(np.pi * y) * np.sin(np.pi * z)
        return (
            2.0 / epsilon**2 * sech_squared * tanh_x * yz
            + 2.0 * np.pi**2 * tanh_x * yz
        )

    return u_exact, f, u_exact, "3D boundary layer"


def internal_layer(epsilon=0.05):
    """A steep planar layer crossing the unit cube.

    The transition is centered on ``x + y + z = 1.5``.  Increasing
    ``epsilon`` makes the layer smoother and easier to resolve on coarse
    meshes; the default intentionally provides a challenging test.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    def u_exact(x, y, z):
        return np.tanh((x + y + z - 1.5) / epsilon)

    def f(x, y, z):
        scaled_distance = (x + y + z - 1.5) / epsilon
        tanh_term = np.tanh(scaled_distance)
        sech_squared = 1.0 / np.cosh(scaled_distance)**2
        # Each of the three pure second derivatives is
        # -2 sech(s)^2 tanh(s) / epsilon^2.
        return 6.0 / epsilon**2 * sech_squared * tanh_term

    return u_exact, f, u_exact, "3D internal layer (diagonal plane)"


# Descriptive aliases consistent with the 2D module's naming style.
smooth_sin_cos = smooth_sin
corner_peak = gaussian_peak
