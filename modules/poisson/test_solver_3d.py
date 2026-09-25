"""Three-dimensional manufactured-solution tests for the SIPG solver."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from .poisson_DG import P1DGPoissonSolver
    from . import manufactured_solutions_3d
except ImportError:
    from poisson_DG import P1DGPoissonSolver
    import manufactured_solutions_3d

from panda.lib import boundary_conditions, polygonal_mesh


def _solve(problem, n, penalty=12.0):
    mesh = polygonal_mesh.create_cube_mesh(n=n)
    u_exact, source, boundary_value, _ = problem()
    conditions = boundary_conditions.BoundaryConditionManager(mesh)
    conditions.add_bc_to_all_boundaries("dirichlet", boundary_value)
    solver = P1DGPoissonSolver(mesh, conditions, penalty_param=penalty)
    solution = solver.solve(source)
    errors = np.array([
        solver.evaluate_solution(solution, mesh.cell_centroid(cell_id), cell_id)
        - u_exact(*mesh.cell_centroid(cell_id))
        for cell_id in range(mesh.n_cells)
    ])
    return solver, solution, errors


def test_cube_mesh_geometry_and_connectivity():
    mesh = polygonal_mesh.create_box_mesh(
        length=2.0, width=3.0, height=4.0, nx=2, ny=3, nz=4
    )
    assert mesh.dimension == 3
    assert mesh.n_cells == 24
    assert len(mesh.boundary_faces) == 2 * (2 * 3 + 2 * 4 + 3 * 4)
    assert np.isclose(sum(mesh.cell_volume(i) for i in range(mesh.n_cells)), 24.0)
    for face_id in mesh.boundary_faces:
        face = mesh.faces[face_id]
        cell_id = mesh.face_to_cells[tuple(sorted(face))][0]
        direction = mesh.face_centroid(face_id) - mesh.cell_centroid(cell_id)
        assert np.dot(mesh.face_normal(face_id, cell_id), direction) > 0.0

    tetrahedra = polygonal_mesh.create_tetrahedral_cube_mesh(n=2)
    assert tetrahedra.n_cells == 48
    assert len(tetrahedra.boundary_faces) == 48
    assert sum(tetrahedra.cell_volume(i) for i in range(tetrahedra.n_cells)) == pytest.approx(1.0)


def test_3d_affine_patch_is_exact():
    solver, solution, errors = _solve(manufactured_solutions_3d.affine, n=2)
    assert solver.n_dofs_per_cell == 4
    assert np.max(np.abs(errors)) < 2.0e-12
    for cell_id in range(solver.mesh.n_cells):
        point = solver.mesh.vertices[solver.mesh.cells[cell_id][0]]
        numerical = solver.evaluate_solution(solution, point, cell_id)
        exact, _, _, _ = manufactured_solutions_3d.affine()
        assert numerical == pytest.approx(exact(*point), abs=3.0e-12)


@pytest.mark.parametrize(
    "problem,n,tolerance",
    [
        (manufactured_solutions_3d.smooth_sin, 5, 1.0e-1),
        (manufactured_solutions_3d.polynomial, 4, 2.0e-3),
        (manufactured_solutions_3d.gaussian_peak, 5, 7.0e-2),
        (manufactured_solutions_3d.multiple_peaks, 5, 1.1e-1),
    ],
)
def test_3d_manufactured_solution_accuracy(problem, n, tolerance):
    _, _, errors = _solve(problem, n=n)
    assert np.sqrt(np.mean(errors**2)) < tolerance


def test_3d_quadratic_solution_converges_at_second_order():
    errors = []
    for n in (2, 3, 4):
        _, _, point_errors = _solve(
            manufactured_solutions_3d.quadratic, n=n, penalty=20.0
        )
        errors.append(np.sqrt(np.mean(point_errors**2)))
    rate = np.log(errors[0] / errors[-1]) / np.log(4.0 / 2.0)
    assert errors[0] > errors[1] > errors[2]
    assert rate > 1.7


def test_3d_boundary_layer_manufactured_data_are_finite():
    u_exact, source, boundary_value, _ = manufactured_solutions_3d.boundary_layer()
    sample_points = ((0.0, 0.2, 0.3), (0.15, 0.5, 0.5), (1.0, 0.8, 0.7))
    for point in sample_points:
        assert np.isfinite(u_exact(*point))
        assert np.isfinite(source(*point))
        assert boundary_value(*point) == u_exact(*point)


def test_3d_internal_layer_source_and_solve():
    epsilon = 0.2
    problem = lambda: manufactured_solutions_3d.internal_layer(epsilon=epsilon)
    u_exact, source, boundary_value, _ = problem()

    point = np.array([0.35, 0.45, 0.55])
    step = 1.0e-4
    numerical_laplacian = 0.0
    for direction in range(3):
        offset = np.zeros(3)
        offset[direction] = step
        numerical_laplacian += (
            u_exact(*(point + offset))
            - 2.0 * u_exact(*point)
            + u_exact(*(point - offset))
        ) / step**2

    assert source(*point) == pytest.approx(-numerical_laplacian, rel=2.0e-6)
    assert boundary_value(*point) == u_exact(*point)

    _, _, errors = _solve(problem, n=5, penalty=12.0)
    assert np.sqrt(np.mean(errors**2)) < 2.5e-1


def test_3d_mixed_dirichlet_neumann_affine_patch():
    mesh = polygonal_mesh.create_cube_mesh(n=2)
    exact = lambda x, y, z: 1.0 + x + 2.0 * y + 3.0 * z
    conditions = boundary_conditions.BoundaryConditionManager(mesh)
    boundary_data = (
        (lambda x, y, z: np.isclose(x, 0.0), "dirichlet", exact),
        (lambda x, y, z: np.isclose(x, 1.0), "neumann", 1.0),
        (lambda x, y, z: np.isclose(y, 0.0), "neumann", -2.0),
        (lambda x, y, z: np.isclose(y, 1.0), "neumann", 2.0),
        (lambda x, y, z: np.isclose(z, 0.0), "neumann", -3.0),
        (lambda x, y, z: np.isclose(z, 1.0), "neumann", 3.0),
    )
    for index, (region, bc_type, value) in enumerate(boundary_data):
        conditions.add_bc_by_function(region, bc_type, value, name=f"face_{index}")

    solver = P1DGPoissonSolver(mesh, conditions, penalty_param=12.0)
    solution = solver.solve(lambda x, y, z: 0.0)
    errors = [
        abs(solver.evaluate_solution(solution, mesh.cell_centroid(cell_id), cell_id)
            - exact(*mesh.cell_centroid(cell_id)))
        for cell_id in range(mesh.n_cells)
    ]
    assert max(errors) < 3.0e-12
