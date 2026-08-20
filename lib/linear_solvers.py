"""Linear-system solver backends shared by PANDA's PDE modules."""

from dataclasses import dataclass
import inspect
from time import perf_counter
import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import (
    LinearOperator,
    bicgstab,
    cg,
    gmres,
    minres,
    spilu,
    spsolve,
)


@dataclass(frozen=True)
class LinearSolveInfo:
    """Diagnostics from the most recent linear solve."""

    method: str
    converged: bool
    iterations: int
    residual_norm: float
    elapsed_seconds: float


class LinearSolverError(RuntimeError):
    """Raised when an iterative method breaks down or does not converge."""


def _shifted_matrix(A, relative_shift):
    """Shift the preconditioner matrix without changing the solved operator."""
    row_scale = np.asarray(abs(A).sum(axis=1)).ravel()
    positive_scale = row_scale[row_scale > 0.0]
    fallback_scale = float(np.median(positive_scale)) if positive_scale.size else 1.0
    row_scale[row_scale == 0.0] = fallback_scale
    return csc_matrix(A + diags(relative_shift * row_scale))


def _make_ilu_preconditioner(A, options):
    options = dict(options or {})
    explicit_shift = options.pop("shift", None)
    auto_shift = options.pop("auto_shift", True)
    shift_sequence = options.pop("shift_sequence", (1e-12, 1e-10, 1e-8, 1e-6))

    factor_matrix = (
        _shifted_matrix(A, float(explicit_shift))
        if explicit_shift is not None
        else csc_matrix(A)
    )

    try:
        ilu = spilu(factor_matrix, **options)
        return LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)
    except RuntimeError as error:
        if not auto_shift or "singular" not in str(error).lower():
            raise
        original_error = error

    # Coupled saddle-point matrices often need a symmetric ordering and no
    # diagonal-pivot preference, even when their diagonal blocks factor well.
    robust_options = dict(options)
    robust_options.setdefault("permc_spec", "MMD_AT_PLUS_A")
    robust_options.setdefault("diag_pivot_thresh", 0.0)
    if robust_options != options:
        try:
            ilu = spilu(factor_matrix, **robust_options)
        except RuntimeError as error:
            if "singular" not in str(error).lower():
                raise
        else:
            warnings.warn(
                "ILU encountered a zero pivot; using symmetric minimum-degree "
                "ordering and relaxed diagonal pivoting",
                RuntimeWarning,
                stacklevel=3,
            )
            return LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)

    attempted_shifts = []
    for shift in shift_sequence:
        shift = float(shift)
        attempted_shifts.append(shift)
        try:
            ilu = spilu(_shifted_matrix(A, shift), **robust_options)
        except RuntimeError as error:
            if "singular" not in str(error).lower():
                raise
            continue

        warnings.warn(
            "ILU encountered a zero pivot; using a scale-aware diagonal "
            f"shift of {shift:.0e} for the preconditioner only",
            RuntimeWarning,
            stacklevel=3,
        )
        return LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)

    attempted = ", ".join(f"{shift:.0e}" for shift in attempted_shifts)
    raise LinearSolverError(
        "ILU factorization remained singular after diagonal-shift retries "
        f"({attempted}). Try a larger fill_factor or preconditioner='jacobi'."
    ) from original_error


def _make_preconditioner(A, preconditioner, options=None):
    if isinstance(preconditioner, LinearOperator):
        return preconditioner
    if preconditioner is None or str(preconditioner).lower() == "none":
        return None

    name = str(preconditioner).lower()
    if name == "jacobi":
        diagonal = np.asarray(A.diagonal(), dtype=float)
        threshold = np.finfo(float).eps
        if np.any(np.abs(diagonal) <= threshold):
            raise ValueError("Jacobi preconditioning requires a nonzero matrix diagonal")
        return LinearOperator(A.shape, matvec=lambda x: x / diagonal, dtype=A.dtype)

    if name == "ilu":
        return _make_ilu_preconditioner(A, options)

    raise ValueError(
        f"Unknown preconditioner {preconditioner!r}; expected None, 'jacobi', or 'ilu'"
    )


def _tolerance_kwargs(method, rtol, atol):
    """Support both old SciPy ``tol`` and current ``rtol`` APIs."""
    parameters = inspect.signature(method).parameters
    kwargs = {"rtol": rtol} if "rtol" in parameters else {"tol": rtol}
    if "atol" in parameters:
        kwargs["atol"] = atol
    return kwargs


def solve_linear_system(
    A,
    b,
    *,
    method="direct",
    rtol=1e-8,
    atol=0.0,
    maxiter=None,
    preconditioner=None,
    preconditioner_options=None,
    x0=None,
    restart=None,
):
    """Solve ``A x = b`` using a direct or Krylov method.

    Parameters are intentionally SciPy-like.  ``method`` may be ``direct``,
    ``cg``, ``minres``, ``gmres``, or ``bicgstab``.  The optional
    preconditioners are ``jacobi`` and ``ilu``. ``preconditioner_options`` is
    forwarded to SciPy's ILU factorization and also accepts ``shift``,
    ``auto_shift``, and ``shift_sequence``.

    Returns
    -------
    solution, info : (numpy.ndarray, LinearSolveInfo)
        The solution and convergence diagnostics.
    """
    A = csr_matrix(A)
    b = np.asarray(b)
    method_name = str(method).lower()
    started_at = perf_counter()

    if method_name == "direct":
        solution = spsolve(A, b)
        if not np.all(np.isfinite(solution)):
            raise LinearSolverError("The direct solver returned non-finite values")
        residual = float(np.linalg.norm(A @ solution - b))
        elapsed = perf_counter() - started_at
        return solution, LinearSolveInfo(method_name, True, 0, residual, elapsed)

    methods = {
        "cg": cg,
        "minres": minres,
        "gmres": gmres,
        "bicgstab": bicgstab,
    }
    if method_name not in methods:
        available = ", ".join(("direct", *methods))
        raise ValueError(f"Unknown linear solver {method!r}; expected one of: {available}")

    krylov_method = methods[method_name]
    M = _make_preconditioner(A, preconditioner, preconditioner_options)
    iterations = 0

    def count_iteration(_):
        nonlocal iterations
        iterations += 1

    kwargs = _tolerance_kwargs(krylov_method, rtol, atol)
    kwargs.update({"maxiter": maxiter, "M": M, "x0": x0, "callback": count_iteration})
    if method_name == "gmres":
        kwargs["restart"] = restart
        if "callback_type" in inspect.signature(gmres).parameters:
            kwargs["callback_type"] = "legacy"

    solution, status = krylov_method(A, b, **kwargs)
    residual = float(np.linalg.norm(A @ solution - b))
    if status != 0:
        if status > 0:
            message = (
                f"{method_name} did not converge after {status} iterations "
                f"(residual {residual:.6e})"
            )
        else:
            message = f"{method_name} failed with breakdown code {status}"
        raise LinearSolverError(message)

    elapsed = perf_counter() - started_at
    info = LinearSolveInfo(method_name, True, iterations, residual, elapsed)
    return solution, info
