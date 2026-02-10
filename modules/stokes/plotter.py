import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata

def plot_results(mesh, solver, u_dofs):
    """
    Visualizes the results using matplotlib.
    """
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Normalize for colormap
    denom_vel = vel_mag.max() - vel_mag.min()
    if abs(denom_vel) < 1e-10: denom_vel=1e-10
    vel_norm = (vel_mag - vel_mag.min()) / denom_vel
    
    denom_p = p_avg.max() - p_avg.min()
    if abs(denom_p) < 1e-10: denom_p=1e-10
    p_norm = (p_avg - p_avg.min()) / denom_p

    # --- Plot 1: Velocity Magnitude with mesh ---
    for i, cell in enumerate(mesh.cells):
        verts = mesh.vertices[cell]
        poly = plt.Polygon(verts, facecolor=plt.cm.viridis(vel_norm[i]), 
                          edgecolor='black', linewidth=0.3)
        ax1.add_patch(poly)

    ax1.set_xlim(mesh.vertices[:, 0].min()-0.05, mesh.vertices[:, 0].max()+0.05)
    ax1.set_ylim(mesh.vertices[:, 1].min()-0.05, mesh.vertices[:, 1].max()+0.05)
    ax1.set_aspect('equal')
    ax1.set_title("Velocity Magnitude |u|")

    sm1 = plt.cm.ScalarMappable(cmap='viridis', 
                                 norm=plt.Normalize(vmin=vel_mag.min(), vmax=vel_mag.max()))
    sm1.set_array([])
    plt.colorbar(sm1, ax=ax1)

    # Quiver overlay
    skip = slice(None, None, 1)
    ax1.quiver(cents[skip,0], cents[skip,1], u_avg[skip], v_avg[skip], 
               color='white', alpha=0.6)

    # --- Plot 2: Pressure with mesh ---
    for i, cell in enumerate(mesh.cells):
        verts = mesh.vertices[cell]
        poly = plt.Polygon(verts, facecolor=plt.cm.RdBu_r(p_norm[i]), 
                          edgecolor='black', linewidth=0.3)
        ax2.add_patch(poly)

    ax2.set_xlim(mesh.vertices[:, 0].min()-0.05, mesh.vertices[:, 0].max()+0.05)
    ax2.set_ylim(mesh.vertices[:, 1].min()-0.05, mesh.vertices[:, 1].max()+0.05)
    ax2.set_aspect('equal')
    ax2.set_title("Pressure Field")

    sm2 = plt.cm.ScalarMappable(cmap='RdBu_r', 
                                 norm=plt.Normalize(vmin=p_avg.min(), vmax=p_avg.max()))
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax2)

    # --- Plot 3: Streamlines ---
    # Create a regular grid for interpolation
    x_min, x_max = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    y_min, y_max = mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()

    # Create grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Interpolate velocity components onto grid
    grid_u = griddata(cents, u_avg, (grid_x, grid_y), method='linear', fill_value=0)
    grid_v = griddata(cents, v_avg, (grid_x, grid_y), method='linear', fill_value=0)
    grid_speed = np.sqrt(grid_u**2 + grid_v**2)

    # Plot velocity magnitude as background
    contour = ax3.contourf(grid_x, grid_y, grid_speed, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax3, label='Velocity Magnitude')

    # Add streamlines
    ax3.streamplot(grid_x, grid_y, grid_u, grid_v, 
                  color='black', linewidth=1, density=1.5, arrowsize=1.5)

    # Add mesh edges
    for cell in mesh.cells:
        verts = mesh.vertices[cell]
        verts_closed = np.vstack([verts, verts[0]])
        ax3.plot(verts_closed[:, 0], verts_closed[:, 1], 
                'k-', linewidth=0.3, alpha=0.3)

    ax3.set_xlim(x_min-0.05, x_max+0.05)
    ax3.set_ylim(y_min-0.05, y_max+0.05)
    ax3.set_aspect('equal')
    ax3.set_title("Streamlines")

    plt.tight_layout()
    
    # Show plots interactively when a display is available; do not save PNGs
    backend = matplotlib.get_backend().lower()
    has_display = ('display' in os.environ and os.environ.get('DISPLAY')) or ('qt' in backend) or ('tk' in backend)
    if has_display:
        plt.show()
    else:
        print("Display not available; plots not shown. Run in an interactive environment to view them.")
