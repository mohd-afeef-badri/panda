import numpy as np
import matplotlib.pyplot as plt

from med_reader import *
from poly_test import *
from med_writer import *
from vtk_writer import *
from DG_P1 import P1DGPoissonSolver

if __name__ == "__main__":
    # Create mesh
    # mesh = create_square_mesh(n=20)
    mesh = load_med_mesh_mc("./mesh/mesh.med")

    # smooth_sin_cos        # test_extreme_corner
    # test_circular_layer   # test_sharp_front
    # test_multiple_peaks   # test_corner_peak
    # test_internal_layer   # test_boundary_layer
    u_exact, f, g, name = test_extreme_corner()

    # Solve with different penalty parameters to verify robustness
    penalties = [2.0, 5.0, 10.0]

    fig, axes = plt.subplots(2, len(penalties), figsize=(15, 10))

    for idx, penalty in enumerate(penalties):
        solver = P1DGPoissonSolver(mesh, penalty_param=penalty)
        u_dofs = solver.solve(f, g)

        # Get cell values
        cell_values = []
        for cell_id in range(mesh.n_cells):
            cent = mesh.cell_centroid(cell_id)
            u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
            cell_values.append(u_val)
        
        cell_values = np.array(cell_values)
        
        # Plot solution
        ax = axes[0, idx]
        for cell_id, cell in enumerate(mesh.cells):
            verts = mesh.vertices[cell]
            val_norm = (cell_values[cell_id] - cell_values.min()) / (cell_values.max() - cell_values.min() + 1e-10)
            poly = plt.Polygon(verts, facecolor=plt.cm.viridis(val_norm), 
                             edgecolor='black', linewidth=0.3)
            ax.add_patch(poly)
        
        all_verts = mesh.vertices
        ax.set_xlim(all_verts[:, 0].min()-0.05, all_verts[:, 0].max()+0.05)
        ax.set_ylim(all_verts[:, 1].min()-0.05, all_verts[:, 1].max()+0.05)
        ax.set_aspect('equal')
        ax.set_title(f'Solution (γ={penalty})')
        
        # Plot error
        ax = axes[1, idx]
        errors = []
        for cell_id in range(mesh.n_cells):
            cent = mesh.cell_centroid(cell_id)
            u_num = solver.evaluate_solution(u_dofs, cent, cell_id)
            u_exact_val = u_exact(cent[0], cent[1])
            errors.append(abs(u_num - u_exact_val))
        
        max_error = max(errors) if max(errors) > 0 else 1.0
        for cell_id, cell in enumerate(mesh.cells):
            verts = mesh.vertices[cell]
            poly = plt.Polygon(verts, facecolor=plt.cm.hot(errors[cell_id]/max_error), 
                             edgecolor='black', linewidth=0.3)
            ax.add_patch(poly)
        
        ax.set_xlim(all_verts[:, 0].min()-0.05, all_verts[:, 0].max()+0.05)
        ax.set_ylim(all_verts[:, 1].min()-0.05, all_verts[:, 1].max()+0.05)
        ax.set_aspect('equal')
        ax.set_title(f'Error (max={max(errors):.3e})')
        
        print(f"\nPenalty γ = {penalty}:")
        print(f"  Max error: {max(errors):.6e}")
        print(f"  Mean error: {np.mean(errors):.6e}")
        print(f"  L2 error: {np.sqrt(np.mean(np.array(errors)**2)):.6e}")
    
    plt.tight_layout()
    plt.show()
    
    # Export solution to VTK | MED
    print("\n" + "="*60)
    print("Exporting to VTK | MED formats visualization:")
    print("="*60)
    
    # Use the last solver (penalty=20)
    u_exact_func = u_exact #lambda x, y: np.exp(-50 * ((x - .5)**2 + (y - .5)**2))

    # # Export to triangular mesh
    # project_and_export_to_triangular_mesh_vtk(
    #     solver,
    #     u_dofs, 
    #     tria_mesh_file="./mesh/mesh_tria_0.med",
    #     output_file="./solution/solution_triangular.vtk",
    #     field_name="u",
    #     u_exact_func=u_exact
    # )

    # # Export to triangular mesh
    # project_and_export_to_triangular_mesh_med(
    #     solver,
    #     u_dofs, 
    #     tria_mesh_file="./mesh/mesh_tria_0.med",
    #     output_file="./solution/solution_triangular.med",
    #     field_name="u",
    #     u_exact_func=u_exact
    # )

    # Export using different methods
    export_to_vtk(solver, u_dofs, "./solution/solution_p0.vtk", "u", u_exact, method="P0")
    export_to_vtk(solver, u_dofs, "./solution/solution_p1_vertex.vtk", "u", u_exact, method="P1_vertex")
    print("Open these files in ParaView to visualize the solution!\n")

    # Export to MED format
    export_to_med(solver, u_dofs, "./solution/solution_p0.med", "u", u_exact, method="P0")
    export_to_med(solver, u_dofs, "./solution/solution_p1_vertex.med", "u", u_exact, method="P1_vertex")
    print("Open these files in SALOME ParaVis to visualize the solution!\n")
