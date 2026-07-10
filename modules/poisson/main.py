import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import plotter

import manufactured_solutions as manufactured_solutions
from panda.lib import med_io
from panda.lib import vtk_io
from panda.lib import boundary_conditions
from panda.lib import polygonal_mesh
from poisson_DG import *

if __name__ == "__main__":

    # Define the mesh of the domain
    # mesh = polygonal_mesh.create_square_mesh(n=10)
    mesh = med_io.load_med_mesh_mc("./mesh/square_poly.med")

    # Select test case exact solution and corresponding f, g
        # smooth_sin_cos   # extreme_corner
        # circular_layer   # sharp_front
        # multiple_peaks   # corner_peak
        # internal_layer   # boundary_layer
    u_exact, f, g, name = manufactured_solutions.multiple_peaks()

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

    plotter.plot_results(mesh, solver, u_dofs, u_exact)
    
    # Export solution to VTK | MED
    print("\n" + "="*60)
    print("Exporting to VTK | MED formats visualization:")
    print("="*60)
    
    # Export to triangular mesh (VTK) values are projected to triangular mesh
    # vtk_io.project_and_export_to_triangular_mesh_vtk(solver, u_dofs, 
    #     tria_mesh_file="./mesh/square_tria.med",
    #     output_file="./solution/solution_triangular_poisson.vtk",
    #     fields={"u": {"type": "scalar", "components": [0]}}
    # )

    # # Export to triangular mesh
    # med_io.project_and_export_to_triangular_mesh_med(solver, u_dofs, 
    #     tria_mesh_file="./mesh/mesh_tria_0.med",
    #     output_file="./solution/solution_triangular.med",
    #     fields={"u": {"type": "scalar", "components": [0]}}
    # )

    # Export u using cell field (P0 projection)
    vtk_io.export_solution(solver, u_dofs, filename="./solution/solution_p0.vtk", fields="u")  # String defaults to scalar on cells
    
    # Export with explicit field specification
    # vtk_io.export_solution(solver, u_dofs, filename="./solution/poisson.vtk", 
    #           fields={"u": {"type": "scalar", "components": [0], "projection": "cell"}})

    # Export u using vertex field (P1_vertex projection)
    vtk_io.export_solution(solver, u_dofs, filename="./solution/solution_nodes.vtk",
                          fields={"u": {"type": "scalar", "components": [0], "projection": "nodes"}})
    
    # Export with gradients
    # vtk_io.export_solution(solver, u_dofs, filename="./solution/poisson_p1_grad.vtk", 
    #           fields={"u": {"type": "scalar", "components": [0], "projection": "nodes", "gradient": True}})
    print("Open these files in ParaView to visualize the solution!\n")

    # Export to MED format
    med_io.export_solution(solver, u_dofs, filename="./solution/solution_cells.med", fields="u")
    med_io.export_solution(solver, u_dofs, filename="./solution/solution_nodes.med",
                          fields={"u": {"type": "scalar", "components": [0], "projection": "nodes"}})
    med_io.export_solution(solver, u_dofs, filename="./solution/solution_cells_with_gradient.med",
        fields={
            "u": {
                "type": "scalar",
                "components": [0],
                "projection": "cell",  # or "nodes"
                "gradient": True,
                "gradient_magnitude": True
            }
        }
    )
