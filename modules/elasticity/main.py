import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from panda.lib import med_io
from panda.lib import boundary_conditions
from panda.lib import polygonal_mesh
from panda.lib import vtk_writer
from elasticity_DG import *

if __name__ == "__main__":
    # 1. Create mesh
    mesh = polygonal_mesh.create_rectangle_mesh(length=5.0, height=1.0, nx=30, ny=15)
    mesh = med_io.load_med_mesh_mc("./mesh/bar_poly.med")

    # 2. Select test case
    def f_body_force(x, y):
        # Gravity acting downward (negative y direction)
        # Uniform distributed load
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

    # 5. Postprocess: Extract solution at cell centroids
    cell_values_ux = []
    cell_values_uy = []
    cell_values_vmag = []
    cents = []

    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        ux_num, uy_num = solver.evaluate_solution(u_dofs, cent, cell_id)
        
        cell_values_ux.append(ux_num)
        cell_values_uy.append(uy_num)
        cell_values_vmag.append(np.sqrt(ux_num**2 + uy_num**2))
        cents.append(cent)

    cell_values_ux = np.array(cell_values_ux)
    cell_values_uy = np.array(cell_values_uy)
    cell_values_vmag = np.array(cell_values_vmag)
    cents = np.array(cents)

    # 6. Visualization
    # Create 2x2 plot: u_x, u_y, |u|, and warped displaced bar
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    all_verts = mesh.vertices
    xlim = (all_verts[:, 0].min()-0.05, all_verts[:, 0].max()+0.05)
    ylim = (all_verts[:, 1].min()-0.05, all_verts[:, 1].max()+0.05)

    # Helper to draw scalar per-cell
    def draw_cell_scalar(ax, scalar_vals, cmap=plt.cm.viridis, title=""):
        norm = (scalar_vals - scalar_vals.min()) / (scalar_vals.max() - scalar_vals.min() + 1e-12)
        for cell_id, cell in enumerate(mesh.cells):
            verts = mesh.vertices[cell]
            poly = plt.Polygon(verts, facecolor=cmap(norm[cell_id]), edgecolor='black', linewidth=0.3)
            ax.add_patch(poly)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_title(title)

    # Top-left: u_x
    ax = axes[0, 0]
    draw_cell_scalar(ax, cell_values_ux, cmap=plt.cm.viridis,
                     title=f'u_x (min={cell_values_ux.min():.3e}, max={cell_values_ux.max():.3e})')

    # Top-right: u_y
    ax = axes[0, 1]
    draw_cell_scalar(ax, cell_values_uy, cmap=plt.cm.viridis,
                     title=f'u_y (min={cell_values_uy.min():.3e}, max={cell_values_uy.max():.3e})')

    # Bottom-left: |u|
    ax = axes[1, 0]
    draw_cell_scalar(ax, cell_values_vmag, cmap=plt.cm.viridis,
                     title=f'|u| (max={cell_values_vmag.max():.3e})')

    # Bottom-right: warped displaced bar (color by |u|)
    ax = axes[1, 1]
    # Compute vertex-wise displacement by evaluating solution at vertices
    n_vertices = mesh.n_vertices
    vert_disp = np.zeros((n_vertices, 2))
    vert_counts = np.zeros((n_vertices,))
    for v_id in range(n_vertices):
        x, y = mesh.vertices[v_id]
        # find a cell that contains this vertex
        # mesh.cells are lists of vertex indices
        found = False
        vals = []
        for cell_id, cell in enumerate(mesh.cells):
            if v_id in cell:
                ux_v, uy_v = solver.evaluate_solution(u_dofs, (x, y), cell_id)
                vals.append((ux_v, uy_v))
                found = True
        if found and vals:
            vals = np.array(vals)
            vert_disp[v_id, 0] = vals[:, 0].mean()
            vert_disp[v_id, 1] = vals[:, 1].mean()

    # Choose warp factor (scale displacement for visualization)
    warp_factor = .005
    displaced_verts = mesh.vertices.copy() + warp_factor * vert_disp

    # Color by cell |u| (same as cell_values_vmag)
    vmag_norm = (cell_values_vmag - cell_values_vmag.min()) / (cell_values_vmag.max() - cell_values_vmag.min() + 1e-12)
    for cell_id, cell in enumerate(mesh.cells):
        verts = displaced_verts[cell]
        poly = plt.Polygon(verts, facecolor=plt.cm.viridis(vmag_norm[cell_id]), edgecolor='black', linewidth=0.3)
        ax.add_patch(poly)

    # Adjust limits to include displaced vertices
    all_disp = displaced_verts
    xlim_disp = (all_disp[:, 0].min() - 0.05, all_disp[:, 0].max() + 0.05)
    ylim_disp = (all_disp[:, 1].min() - 0.05, all_disp[:, 1].max() + 0.05)
    ax.set_xlim(*xlim_disp)
    ax.set_ylim(*ylim_disp)
    ax.set_aspect('equal')
    ax.set_title(f'Warped displaced mesh (factor={warp_factor})')

    plt.tight_layout()

    # Show plots interactively when a display is available; do not save PNGs
    backend = matplotlib.get_backend().lower()
    has_display = ('display' in os.environ and os.environ.get('DISPLAY')) or ('qt' in backend) or ('tk' in backend)
    if has_display:
        plt.show()
    else:
        print("Display not available; plots not shown. Run in an interactive environment to view them.")

    # 7. Summary
    print("\n" + "="*60)
    print(f"Test case:")
    print(f"Mesh: {mesh.n_cells} cells, {mesh.n_vertices} vertices")
    print(f"Lamé parameters: λ={solver.lam}, μ={solver.mu}")
    print(f"SIPG penalty: γ={solver.penalty}")
    print("="*60)

    # Export to VTK for visualization in ParaView
    print("\n" + "="*60)
    print("Exporting to VTK format...")
    vtk_writer.export_to_vtk(
        solver,
        u_dofs,
        "./solution/elasticity_solution.vtk",
        fields={
            "displacement": {"type": "vector", "components": [0, 1]}
        },
        method="P0"
    )