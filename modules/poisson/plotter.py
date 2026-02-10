import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def plot_results(mesh, solver, u_dofs, u_exact):
    """
    Visualizes the results using matplotlib.
    """
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
    # Avoid division by zero
    denom = cell_values.max() - cell_values.min()
    if abs(denom) < 1e-10:
        denom = 1e-10
    
    for cell_id, cell in enumerate(mesh.cells):
        verts = mesh.vertices[cell]
        val_norm = (cell_values[cell_id] - cell_values.min()) / denom
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
    
    # Show plots interactively when a display is available; do not save PNGs
    backend = matplotlib.get_backend().lower()
    has_display = ('display' in os.environ and os.environ.get('DISPLAY')) or ('qt' in backend) or ('tk' in backend)
    if has_display:
        plt.show()
    else:
        print("Display not available; plots not shown. Run in an interactive environment to view them.")
