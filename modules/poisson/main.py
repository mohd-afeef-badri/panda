import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import plotter

import manufactured_solutions as manufactured_solutions
from panda.lib import med_io
from panda.lib import vtk_writer
from panda.lib import boundary_conditions
from panda.lib import polygonal_mesh
from poisson_DG import *

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

    plotter.plot_results(mesh, solver, u_dofs, u_exact)
    
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
