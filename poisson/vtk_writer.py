import numpy as np
from med_io import load_med_mesh_mc

def export_to_vtk(solver, u_dofs, filename="solution.vtk", field_name="u",
                  u_exact_func=None, method="P0"):
    if method == "P0":
        _export_vtk_p0(solver, u_dofs, filename, field_name, u_exact_func)
    elif method == "P1_vertex":
        _export_vtk_p1_vertex(solver, u_dofs, filename, field_name, u_exact_func)
    else:
        raise ValueError(f"Unknown export method: {method}")


def _export_vtk_p0(solver, u_dofs, filename, field_name, u_exact_func):
    mesh = solver.mesh
    # Evaluate at cell centroids
    u_cells = np.zeros(mesh.n_cells)
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        u_cells[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)

    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("P1 DG Solution (P0 projection)\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {mesh.n_vertices} double\n")
        for v in mesh.vertices:
            f.write(f"{v[0]} {v[1]} 0.0\n")

        # Cells
        total_size = sum(len(cell) + 1 for cell in mesh.cells)
        f.write(f"\nCELLS {mesh.n_cells} {total_size}\n")
        for cell in mesh.cells:
            f.write(f"{len(cell)} " + " ".join(map(str, cell)) + "\n")

        # Cell types
        f.write(f"\nCELL_TYPES {mesh.n_cells}\n")
        for cell in mesh.cells:
            n_nodes = len(cell)
            if n_nodes == 3:
                cell_type = 5  # VTK_TRIANGLE
            elif n_nodes == 4:
                cell_type = 9  # VTK_QUAD
            else:
                cell_type = 7  # VTK_POLYGON
            f.write(f"{cell_type}\n")

        # Cell data
        f.write(f"\nCELL_DATA {mesh.n_cells}\n")
        f.write(f"SCALARS {field_name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in u_cells:
            f.write(f"{val}\n")

        # Error field if available
        if u_exact_func is not None:
            f.write(f"\nSCALARS error double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for cell_id in range(mesh.n_cells):
                cent = mesh.cell_centroid(cell_id)
                error = abs(u_cells[cell_id] - u_exact_func(cent[0], cent[1]))
                f.write(f"{error}\n")

    print(f"P0 projection exported to: {filename}")


def _export_vtk_p1_vertex(solver, u_dofs, filename, field_name, u_exact_func):
    mesh = solver.mesh
    # Interpolate to vertices using averaging from adjacent cells
    u_vertices = np.zeros(mesh.n_vertices)
    vertex_count = np.zeros(mesh.n_vertices)

    for cell_id, cell in enumerate(mesh.cells):
        for vertex_id in cell:
            vertex_pos = mesh.vertices[vertex_id]
            u_val = solver.evaluate_solution(u_dofs, vertex_pos, cell_id)
            u_vertices[vertex_id] += u_val
            vertex_count[vertex_id] += 1

    # Average values at vertices shared by multiple cells
    u_vertices /= np.maximum(vertex_count, 1)

    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("P1 DG Solution (vertex interpolation)\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {mesh.n_vertices} double\n")
        for v in mesh.vertices:
            f.write(f"{v[0]} {v[1]} 0.0\n")

        # Cells
        total_size = sum(len(cell) + 1 for cell in mesh.cells)
        f.write(f"\nCELLS {mesh.n_cells} {total_size}\n")
        for cell in mesh.cells:
            f.write(f"{len(cell)} " + " ".join(map(str, cell)) + "\n")

        # Cell types
        f.write(f"\nCELL_TYPES {mesh.n_cells}\n")
        for cell in mesh.cells:
            n_nodes = len(cell)
            if n_nodes == 3:
                cell_type = 5  # VTK_TRIANGLE
            elif n_nodes == 4:
                cell_type = 9  # VTK_QUAD
            else:
                cell_type = 7  # VTK_POLYGON
            f.write(f"{cell_type}\n")

        # Point data (smooth visualization)
        f.write(f"\nPOINT_DATA {mesh.n_vertices}\n")
        f.write(f"SCALARS {field_name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in u_vertices:
            f.write(f"{val}\n")

        # Error field if available
        if u_exact_func is not None:
            f.write(f"\nSCALARS error double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for vertex_id in range(mesh.n_vertices):
                vertex_pos = mesh.vertices[vertex_id]
                error = abs(u_vertices[vertex_id] - u_exact_func(vertex_pos[0], vertex_pos[1]))
                f.write(f"{error}\n")

    print(f"P1 vertex interpolation exported to: {filename}")

def project_and_export_to_triangular_mesh_vtk(solver, u_dofs, tria_mesh_file, 
                                  output_file="solution_tria.vtk", 
                                  field_name="u", 
                                  u_exact_func=None):
    """
    Export P1 DG solution to a triangular mesh where triangular vertices 
    correspond to polygonal mesh cell centroids.
    
    Parameters:
    -----------
    u_dofs : array
        Solution DOF array from the polygonal mesh
    tria_mesh_file : str
        Path to the triangular mesh file (e.g., "mesh_tria.med")
    output_file : str
        Output VTK filename
    field_name : str
        Name for the solution field
    u_exact_func : callable, optional
        Exact solution function for error computation
    """    
    # Load triangular mesh
    print(f"Loading triangular mesh from {tria_mesh_file}...")
    tria_mesh = load_med_mesh_mc(tria_mesh_file)

    # Check that triangular vertices match polymesh centroids
    if tria_mesh.n_vertices != solver.mesh.n_cells:
        print(f"WARNING: Triangular mesh has {tria_mesh.n_vertices} vertices "
              f"but polymesh has {solver.mesh.n_cells} cells!")
        print("Proceeding anyway, but results may be incorrect.")

    # Evaluate DG solution at each polymesh cell centroid
    # These values correspond to triangular mesh vertices
    u_tria_vertices = np.zeros(tria_mesh.n_vertices)

    for cell_id in range(min(solver.mesh.n_cells, tria_mesh.n_vertices)):
        # Get polymesh cell centroid
        cent = solver.mesh.cell_centroid(cell_id)
        # Evaluate DG solution at this point
        u_tria_vertices[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)

    # Write VTK file
    with open(output_file, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("P1 DG Solution on Triangular Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {tria_mesh.n_vertices} double\n")
        for v in tria_mesh.vertices:
            f.write(f"{v[0]} {v[1]} 0.0\n")

        # Cells
        total_size = sum(len(cell) + 1 for cell in tria_mesh.cells)
        f.write(f"\nCELLS {tria_mesh.n_cells} {total_size}\n")
        for cell in tria_mesh.cells:
            f.write(f"{len(cell)} " + " ".join(map(str, cell)) + "\n")

        # Cell types (should be all triangles)
        f.write(f"\nCELL_TYPES {tria_mesh.n_cells}\n")
        for cell in tria_mesh.cells:
            n_nodes = len(cell)
            if n_nodes == 3:
                cell_type = 5  # VTK_TRIANGLE
            elif n_nodes == 4:
                cell_type = 9  # VTK_QUAD
            else:
                cell_type = 7  # VTK_POLYGON
            f.write(f"{cell_type}\n")

        # Point data (the solution values)
        f.write(f"\nPOINT_DATA {tria_mesh.n_vertices}\n")
        f.write(f"SCALARS {field_name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in u_tria_vertices:
            f.write(f"{val}\n")

        # Error field if available
        if u_exact_func is not None:
            f.write(f"\nSCALARS error double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for vertex_id in range(tria_mesh.n_vertices):
                vertex_pos = tria_mesh.vertices[vertex_id]
                error = abs(u_tria_vertices[vertex_id] - 
                          u_exact_func(vertex_pos[0], vertex_pos[1]))
                f.write(f"{error}\n")

    print(f"Solution exported to triangular mesh: {output_file}")
    print(f"  - Triangular mesh vertices: {tria_mesh.n_vertices}")
    print(f"  - Triangular mesh cells: {tria_mesh.n_cells}")

    # Compute errors on triangular mesh if exact solution provided
    if u_exact_func is not None:
        errors_tria = []
        for vertex_id in range(tria_mesh.n_vertices):
            vertex_pos = tria_mesh.vertices[vertex_id]
            error = abs(u_tria_vertices[vertex_id] - 
                       u_exact_func(vertex_pos[0], vertex_pos[1]))
            errors_tria.append(error)

        print(f"  - Max error: {max(errors_tria):.6e}")
        print(f"  - Mean error: {np.mean(errors_tria):.6e}")
        print(f"  - L2 error: {np.sqrt(np.mean(np.array(errors_tria)**2)):.6e}")
