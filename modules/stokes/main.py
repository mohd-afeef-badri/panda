import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from panda.lib import med_io
from panda.lib import vtk_writer
from panda.lib import boundary_conditions
from panda.lib import polygonal_mesh
from stokes_DG import *

if __name__ == "__main__":

    # 1. Define the mesh of the domain
    # mesh = polygonal_mesh.create_square_mesh(n=10)
    mesh_name = "./../poisson/mesh/square_poly.med"
    mesh = med_io.load_med_mesh_mc(mesh_name)

    # 2. Define Problem (Lid Driven Cavity)
    def f_source(x, y): return 0.0, 0.0
    def g_boundary(x, y):
        # Velocity 1.0 on top lid (y=1), 0.0 elsewhere
        if y > 0.99: return 1.0, 0.0
        return 0.0, 0.0

    # 3. Initialize Solver
    # Note: penalty_u=40.0 is robust for P1 DG.
    solver = P1DGStokesSolver(mesh, viscosity=0.1, penalty_u=40.0, penalty_p=0.5)

    # 4. Solve
    u_dofs = solver.solve(f_source, g_boundary)

    # 3. Postprocess
    if u_dofs is not None:
        print("Solve successful! Visualizing...")
        
        # 5. Extract Cell Averages for Plotting
        u_avg, v_avg, p_avg, cents = [], [], [], []
        
        for i in range(mesh.n_cells):
            idx = solver.get_indices(i)
            # Centroid values (since basis is [1, x-xc, y-yc], the first DOF is the average)
            u_val = u_dofs[idx['u'][0]]
            v_val = u_dofs[idx['v'][0]]
            p_val = u_dofs[idx['p'][0]]
            
            u_avg.append(u_val)
            v_avg.append(v_val)
            p_avg.append(p_val)
            cents.append(mesh.cell_centroid(i))

        cents = np.array(cents)
        u_avg = np.array(u_avg)
        v_avg = np.array(v_avg)
        p_avg = np.array(p_avg)
        vel_mag = np.sqrt(u_avg**2 + v_avg**2)

        # 6. Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Velocity Magnitude
        sc1 = ax1.tripcolor(cents[:,0], cents[:,1], vel_mag, shading='gouraud', cmap='viridis')
        ax1.set_title("Velocity Magnitude |u|")
        ax1.set_aspect('equal')
        plt.colorbar(sc1, ax=ax1)
        
        # Streamlines (Quiver)
        skip = (slice(None, None, 1))
        ax1.quiver(cents[skip,0], cents[skip,1], u_avg[skip], v_avg[skip], 
                   color='white', alpha=0.6)

        # Pressure
        sc2 = ax2.tripcolor(cents[:,0], cents[:,1], p_avg, shading='gouraud', cmap='RdBu_r')
        ax2.set_title("Pressure Field")
        ax2.set_aspect('equal')
        plt.colorbar(sc2, ax=ax2)

        plt.tight_layout()
        plt.show()
        
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

    # # Export to MED format
    # med_io.export_to_med(solver, u_dofs, "./solution/solution_p0.med", "u", method="P0")
    # med_io.export_to_med(solver, u_dofs, "./solution/solution_p1_vertex.med", "u", method="P1_vertex")
    # print("Open these files in SALOME ParaVis to visualize the solution!\n")
