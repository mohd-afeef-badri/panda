import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import os
import numpy as np
import plotter

from panda.lib import med_io
from panda.lib import boundary_conditions
from panda.lib import polygonal_mesh
from panda.lib import vtk_io
from elasticity_DG import *

if __name__ == "__main__":
    # 1. Create mesh
    # mesh = polygonal_mesh.create_rectangle_mesh(length=5.0, height=1.0, nx=30, ny=15)
    mesh = med_io.load_med_mesh_mc("./mesh/bar_poly.med")

    # 2. Select test case
    # Gravity acting downward (negative y direction). Uniform distributed load
    def f_body_force(x, y):
        return 0.0, -0.8

    # 3. Set up boundary conditions
    bc_manager = boundary_conditions.BoundaryConditionManager(mesh)

    # Cantilever beam: FIXED LEFT END, FREE EVERYWHERE ELSE
    # Fixed left end (x ≈ 0): fully clamped
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: x < 0.1,
        bc_type="dirichlet",
        value_func=(0.0, 0.0),  # Fully fixed
        name="left_fixed",
        is_vector=True
    )

    # Right end (x ≈ 5): free (zero traction = natural BC)
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: x >= 0.1,
        bc_type="neumann",
        value_func=(0.0, 0.0),  # Zero traction (free edge)
        name="not_left",
        is_vector=True
    )

    # 4. Create solver and solve
    solver = P1DGLinearElasticitySolver(
        mesh,
        bc_manager,
        lame_lambda=1.0,
        lame_mu=1.0,
        penalty_param=50.0
    )

    print("Assembling system...")
    u_dofs = solver.solve(f_body_force)
    print("Solve complete!")

    # Summary
    print("\n" + "="*60)
    print(f"Test case:")
    print(f"Mesh: {mesh.n_cells} cells, {mesh.n_vertices} vertices")
    print(f"Lamé parameters: λ={solver.lam}, μ={solver.mu}")
    print(f"SIPG penalty: γ={solver.penalty}")
    print("="*60)

    # 5. Visualization
    print("\n" + "="*60)
    print("Plotting results...")
    plotter.plot_results(mesh, solver, u_dofs)

    # Export to VTK for visualization in ParaView
    print("\n" + "="*60)
    print("Exporting to VTK format...")
    vtk_io.export_solution(
        solver,
        u_dofs,
        filename="./solution/elasticity_solution_P0.vtk",
        fields={"displacement": {"type": "vector", "components": [0, 1]}}
    )

    vtk_io.export_solution(
        solver,
        u_dofs,
        filename="./solution/elasticity_solution_P1_vertex.vtk",
        fields={"displacement": {"type": "vector", "components": [0, 1], "projection": "nodes"}}
    )

    # Export to MED for visualization in ParaView
    print("\n" + "="*60)
    print("Exporting to MED format...")
    med_io.export_solution(
        solver,
        u_dofs,
        filename="./solution/elasticity_solution_P1_vertex.med",
        fields={"displacement": {"type": "vector", "components": [0, 1], "projection": "nodes"}}
    )

    med_io.export_solution(
        solver,
        u_dofs,
        filename="./solution/elasticity_solution_P0.med",
        fields={"displacement": {"type": "vector", "components": [0, 1]}}
    )