import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import plotter

from panda.lib import med_io
from panda.lib import vtk_writer
from panda.lib import boundary_conditions
from panda.lib import polygonal_mesh
from stokes_DG import *

if __name__ == "__main__":

    # 1. Define the mesh of the domain and associated boundary conditions manager
    mesh = polygonal_mesh.create_square_mesh(n=10)
    bc_manager = boundary_conditions.BoundaryConditionManager(mesh)

    # Alternatively, load from MED file with groups 
    # mesh_name = "./../poisson/mesh/square_poly.med"
    # mesh = med_io.load_med_mesh_mc(mesh_name)
    # edge_groups = med_io.extract_edge_groups_from_med(mesh_name)
    # bc_manager = boundary_conditions.BoundaryConditionManager(mesh, edge_groups)

    # 2. Define Problem (Lid Driven Cavity)
    def f_source(x, y): return 0.0, 0.0

    # Lid-driven cavity: top wall moving, others no-slip via the med mesh groups
    # bc_manager.add_bc_by_group("top", "dirichlet", lambda x, y: (1.0, 0.0), is_vector=True)
    # bc_manager.add_bc_by_group("bottom", "dirichlet", (0.0, 0.0), is_vector=True)
    # bc_manager.add_bc_by_group("left", "dirichlet", (0.0, 0.0), is_vector=True)
    # bc_manager.add_bc_by_group("right", "dirichlet", (0.0, 0.0), is_vector=True)

    # Lid-driven cavity: top wall moving, others no-slip using function-based regions
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: (y > (1-1e-10)),
        bc_type="dirichlet",
        value_func=(1.0, 0.0),
        name="top",
        is_vector=True
    )
    
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: (y <= (1-1e-10)),
        bc_type="dirichlet",
        value_func=(0.0, 0.0),
        name="wall",
        is_vector=True
    )

    # 3. Solve
    solver = P1DGStokesSolver(mesh, bc_manager, viscosity=0.1, penalty_u=40.0, penalty_p=0.5)
    u_dofs = solver.solve(f_source)

    # 4. Postprocess
    if u_dofs is not None:
        print("Solve successful! Visualizing...")
        
        plotter.plot_results(mesh, solver, u_dofs)
        # Export discontinuous cell data (Raw DG result)
        vtk_writer.export_to_vtk(solver, u_dofs, 
            filename="./solution/stokes_P0.vtk", 
            fields={
                "velocity": {"type": "vector", "components": [0, 1]},
                "pressure": {"type": "scalar", "components": [2]}
            }, 
            method="P0")

        # Export smoothed vertex data (Better for streamlines)
        vtk_writer.export_to_vtk(solver, u_dofs,
            filename="./solution/stokes_P1.vtk", 
            fields={
              "velocity": {"type": "vector", "components": [0, 1]},
              "pressure": {"type": "scalar", "components": [2]}
            },
            method="P1_vertex")

        # Export to triangular mesh (VTK) values are projected to triangular mesh
        vtk_writer.project_and_export_to_triangular_mesh_vtk(solver, u_dofs, 
            tria_mesh_file="./../poisson/mesh/square_tria.med",
            output_file="./solution/stokes_P1_tria_new.vtk",
            fields={
              "velocity": {"type": "vector", "components": [0, 1]},
              "pressure": {"type": "scalar", "components": [2]}
            })

        med_io.export_to_med(solver, u_dofs, 
            filename="./solution/stokes_P0.med", 
            fields={
                "velocity": {"type": "vector", "components": [0, 1]},
                "pressure": {"type": "scalar", "components": [2]}
            }, 
            method="P0")

        # Also export P1 vertex version for smoother visualization
        med_io.export_to_med(solver, u_dofs,
            filename="./solution/stokes_P1.med",
            fields={
                "velocity": {"type": "vector", "components": [0, 1]},
                "pressure": {"type": "scalar", "components": [2]}
            },
            method="P1_vertex")
        print("Open these files in SALOME ParaVis to visualize the solution!\n")
