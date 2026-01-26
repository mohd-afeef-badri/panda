import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

import manufactured_solutions as manufactured_solutions
from panda.lib import med_io
from panda.lib import vtk_writer
from panda.lib import boundary_conditions
from panda.lib import polygonal_mesh
from DG_P1 import *

if __name__ == "__main__":

    # Define the mesh of the domain
    mesh = polygonal_mesh.create_square_mesh(n=10)
    # mesh_name = "./mesh/square_poly.med"
    # mesh = med_io.load_med_mesh_mc(mesh_name)

    # Select test case exact solution and corresponding f, g
        # smooth_sin_cos   # extreme_corner
        # circular_layer   # sharp_front
        # multiple_peaks   # corner_peak
        # internal_layer   # boundary_layer
    u_exact, f, g, name = manufactured_solutions.smooth_sin_cos()

    # Set up DG Poisson solver with boundary conditions
    # Dirichlet BCs on group "boundary"
    # edge_groups = extract_edge_groups_from_med(mesh_name)
    # bc_manager = BoundaryConditionManager(mesh, edge_groups)
    # bc_manager.add_bc_by_group("boundary", "dirichlet", lambda x, y: g(x, y))

    # Set up DG Poisson solver with boundary conditions
    # Dirichlet BCs on all boundaries no needed to specify groups
    bc_manager = boundary_conditions.BoundaryConditionManager(mesh)
    bc_manager.add_bc_to_all_boundaries( bc_type="dirichlet", value_func=lambda x, y: g(x, y) )

    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param = 10.0)
    u_dofs = solver.solve(f)

    fig, axes = plt.subplots(2, 1, figsize=(5, 10))

    # Get cell values
    cell_values = []
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        cell_values.append(u_val)
    
    cell_values = np.array(cell_values)
    
    # Plot solution
    ax = axes[0]
    for cell_id, cell in enumerate(mesh.cells):
        verts = mesh.vertices[cell]
        val_norm = (cell_values[cell_id] - cell_values.min()) / (cell_values.max() - cell_values.min() + 1e-10)
        poly = plt.Polygon(verts, facecolor=plt.cm.viridis(val_norm),  edgecolor='black', linewidth=0.3)
        ax.add_patch(poly)
    
    all_verts = mesh.vertices
    ax.set_xlim(all_verts[:, 0].min()-0.05, all_verts[:, 0].max()+0.05)
    ax.set_ylim(all_verts[:, 1].min()-0.05, all_verts[:, 1].max()+0.05)
    ax.set_aspect('equal')
    ax.set_title(f'Solution (γ={solver.penalty})')
    
    # Plot error
    ax = axes[1]
    errors = []
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        u_num = solver.evaluate_solution(u_dofs, cent, cell_id)
        u_exact_val = u_exact(cent[0], cent[1])
        errors.append(abs(u_num - u_exact_val))
    
    max_error = max(errors) if max(errors) > 0 else 1.0
    for cell_id, cell in enumerate(mesh.cells):
        verts = mesh.vertices[cell]
        poly = plt.Polygon(verts, facecolor=plt.cm.hot(errors[cell_id]/max_error), edgecolor='black', linewidth=0.3)
        ax.add_patch(poly)
    
    ax.set_xlim(all_verts[:, 0].min()-0.05, all_verts[:, 0].max()+0.05)
    ax.set_ylim(all_verts[:, 1].min()-0.05, all_verts[:, 1].max()+0.05)
    ax.set_aspect('equal')
    ax.set_title(f'Error (max={max(errors):.3e})')
    
    print(f"\nSIPG Penalty γ = {solver.penalty}:")
    print(f"  Max error: {max(errors):.6e}")
    print(f"  Mean error: {np.mean(errors):.6e}")
    print(f"  L2 error: {np.sqrt(np.mean(np.array(errors)**2)):.6e}")
    
    plt.tight_layout()
    plt.show()
    
    # Export solution to VTK | MED
    print("\n" + "="*60)
    print("Exporting to VTK | MED formats visualization:")
    print("="*60)
    
    # Export to triangular mesh (VTK) values are projected to triangular mesh
    # vtk_writer.project_and_export_to_triangular_mesh_vtk(
    #     solver,
    #     u_dofs, 
    #     tria_mesh_file="./mesh/square_tria.med",
    #     output_file="./solution/solution_triangular_poisson.vtk",
    #     fields={"u": {"type": "scalar", "components": [0]}}
    # )

    # # Export to triangular mesh
    # med_io.project_and_export_to_triangular_mesh_med(
    #     solver,
    #     u_dofs, 
    #     tria_mesh_file="./mesh/mesh_tria_0.med",
    #     output_file="./solution/solution_triangular.med",
    #     field_name="u"
    # )

    # Export u using cell filed
    vtk_writer.export_to_vtk(solver, u_dofs, "./solution/solution_p0.vtk", "u", method="P0")
    # we can also specify fields as a dictionary
    # vtk_writer.export_to_vtk(solver, u_dofs, "./solution/poisson.vtk", 
    #           fields={"u": {"type": "scalar", "components": [0]}}, method="P0")

    # Export u using vertex field
    vtk_writer.export_to_vtk(solver, u_dofs, "./solution/solution_p1_vertex.vtk", "u", method="P1_vertex")
    # we can also specify fields as a dictionary
    # vtk_writer.export_to_vtk(solver, u_dofs, "./solution/poisson_p1.vtk", 
    #           fields={"u": {"type": "scalar", "components": [0]}}, method="P1_vertex")
    print("Open these files in ParaView to visualize the solution!\n")

    # Export to MED format
    med_io.export_to_med(solver, u_dofs, "./solution/solution_p0.med", "u", method="P0")
    med_io.export_to_med(solver, u_dofs, "./solution/solution_p1_vertex.med", "u", method="P1_vertex")
    print("Open these files in SALOME ParaVis to visualize the solution!\n")
