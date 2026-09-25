import sys
from pathlib import Path

import numpy as np

# Make the panda package importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from panda.lib import boundary_conditions, med_io, vtk_io
from poisson_DG import P1DGPoissonSolver
import manufactured_solutions_3d


INPUT_MESH = Path("mesh/cube.med")
OUTPUT_BASE = Path("solution/poisson_3d")


# 1. Load the 3D MED mesh
mesh = med_io.load_med_mesh_mc(str(INPUT_MESH))

if mesh.dimension != 3:
    raise ValueError(f"Expected a 3D mesh, got dimension {mesh.dimension}")


# 2. Select a manufactured solution
u_exact, source, dirichlet_value, name = (
    manufactured_solutions_3d.polynomial()
)

# Other available choices:
# manufactured_solutions_3d.affine()
# manufactured_solutions_3d.quadratic()
# manufactured_solutions_3d.polynomial()
# manufactured_solutions_3d.gaussian_peak()
# manufactured_solutions_3d.multiple_peaks()
# manufactured_solutions_3d.boundary_layer()


# 3. Apply exact Dirichlet data to the whole boundary
bc = boundary_conditions.BoundaryConditionManager(mesh)
bc.add_bc_to_all_boundaries(
    bc_type="dirichlet",
    value_func=dirichlet_value,
)


# 4. Solve
# solver = P1DGPoissonSolver(
#     mesh,
#     bc,
#     penalty_param=12.0,
# )

solver = P1DGPoissonSolver(
    mesh,
    bc,
    linear_solver="cg",
    solver_options={
        "preconditioner": "jacobi",
        "rtol": 1e-8,
        "maxiter": 1000,
        "verbose": True,
    },
    penalty_param=12.0,
)
print(f"Solving 3D Poisson problem with {solver.n_dofs} DOFs...")
u_dofs = solver.solve(source)


# 5. Evaluate numerical solution and error at cell centroids
centroids = np.array([
    mesh.cell_centroid(cell_id)
    for cell_id in range(mesh.n_cells)
])

u_numerical = np.array([
    solver.evaluate_solution(u_dofs, point, cell_id)
    for cell_id, point in enumerate(centroids)
])

u_reference = np.array([
    u_exact(*point)
    for point in centroids
])

error = u_numerical - u_reference
absolute_error = np.abs(error)

volumes = np.array([
    mesh.cell_volume(cell_id)
    for cell_id in range(mesh.n_cells)
])

l2_error = np.sqrt(
    np.sum(error**2 * volumes) / np.sum(volumes)
)

print(f"Problem:   {name}")
print(f"Cells:     {mesh.n_cells}")
print(f"DOFs:      {solver.n_dofs}")
print(f"L2 error:  {l2_error:.6e}")
print(f"Max error: {absolute_error.max():.6e}")


# 6. Export through the same library APIs used by the 2D solver.
cell_fields = {
    "u": {
        "type": "scalar",
        "components": [0],
        "projection": "cell",
        "gradient": True,
        "gradient_magnitude": True,
        "zz_estimator": True,
    }
}

vtk_io.export_solution(
    solver,
    u_dofs,
    filename="solution/poisson_3d_cells.vtk",
    fields=cell_fields,
)
med_io.export_solution(
    solver,
    u_dofs,
    filename="solution/poisson_3d_cells.med",
    fields=cell_fields,
)

node_fields = {
    "u": {
        **cell_fields["u"],
        "projection": "nodes",
    }
}

vtk_io.export_solution(
    solver,
    u_dofs,
    filename="solution/poisson_3d_nodes.vtk",
    fields=node_fields,
)
med_io.export_solution(
    solver,
    u_dofs,
    filename="solution/poisson_3d_nodes.med",
    fields=node_fields,
)

print("Open the files in solution/ with ParaView.")
