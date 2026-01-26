"""VTK file export utilities for scalar and vector fields."""

from pathlib import Path
import numpy as np
from .med_io import load_med_mesh_mc


def export_to_vtk(solver, u_dofs, filename="solution.vtk", fields=None, method="P0"):
    """
    Export solution to VTK format with flexible field specification.
    
    Parameters:
    -----------
    solver : Solver object
        The solver containing mesh and evaluate_solution method
    u_dofs : array
        Solution DOF array
    filename : str
        Output VTK filename
    fields : dict or str
        Field specification. Can be:
        - str: single scalar field name (e.g., "u" for Poisson)
        - dict: {"field_name": {"type": "scalar"|"vector", "components": [indices]}}
        
        Examples:
        - Poisson: fields="u" or fields={"u": {"type": "scalar", "components": [0]}}
        - Stokes: fields={
                      "velocity": {"type": "vector", "components": [0, 1]},
                      "pressure": {"type": "scalar", "components": [2]}
                  }
        - Elasticity: fields={
                          "displacement": {"type": "vector", "components": [0, 1]},
                          "stress_xx": {"type": "scalar", "components": [2]},
                          "stress_yy": {"type": "scalar", "components": [3]}
                      }
    method : str
        Export method: "P0" or "P1_vertex"
    """
    # Convert simple string to dict format
    if isinstance(fields, str):
        fields = {fields: {"type": "scalar", "components": [0]}}
    elif fields is None:
        fields = {"u": {"type": "scalar", "components": [0]}}
    
    if method == "P0":
        _export_vtk_p0(solver, u_dofs, filename, fields)
    elif method == "P1_vertex":
        _export_vtk_p1_vertex(solver, u_dofs, filename, fields)
    else:
        raise ValueError(f"Unknown export method: {method}")


def _evaluate_fields_at_point(solver, u_dofs, point, cell_id, fields):
    """
    Evaluate all fields at a given point.
    
    Returns:
    --------
    dict : {field_name: field_value}
        For scalars: field_value is a float
        For vectors: field_value is a numpy array
    """
    # Get solution values at this point
    sol_values = solver.evaluate_solution(u_dofs, point, cell_id)
    
    # Handle case where evaluate_solution returns a single value
    if not isinstance(sol_values, (tuple, list, np.ndarray)):
        sol_values = [sol_values]
    else:
        sol_values = np.atleast_1d(sol_values)
    
    # Extract field values based on component indices
    field_values = {}
    for field_name, field_spec in fields.items():
        components = field_spec["components"]
        field_type = field_spec["type"]
        
        if field_type == "scalar":
            # Single component
            field_values[field_name] = sol_values[components[0]]
        elif field_type == "vector":
            # Multiple components
            field_values[field_name] = np.array([sol_values[i] for i in components])
        else:
            raise ValueError(f"Unknown field type: {field_type}")
    
    return field_values


def _export_vtk_p0(solver, u_dofs, filename, fields):
    """Export with P0 projection (cell-centered values)."""
    mesh = solver.mesh
    
    # Initialize storage for all fields
    field_data = {}
    for field_name, field_spec in fields.items():
        if field_spec["type"] == "scalar":
            field_data[field_name] = np.zeros(mesh.n_cells)
        elif field_spec["type"] == "vector":
            n_components = len(field_spec["components"])
            field_data[field_name] = np.zeros((mesh.n_cells, n_components))
    
    # Evaluate at cell centroids
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        cell_fields = _evaluate_fields_at_point(solver, u_dofs, cent, cell_id, fields)
        
        for field_name, field_value in cell_fields.items():
            field_data[field_name][cell_id] = field_value
    
    _write_vtk_file(mesh, filename, fields, field_data, data_location="CELL")
    print(f"P0 projection exported to: {filename}")


def _export_vtk_p1_vertex(solver, u_dofs, filename, fields):
    """Export with P1 vertex interpolation (vertex-centered values)."""
    mesh = solver.mesh
    
    # Initialize storage for all fields
    field_data = {}
    for field_name, field_spec in fields.items():
        if field_spec["type"] == "scalar":
            field_data[field_name] = np.zeros(mesh.n_vertices)
        elif field_spec["type"] == "vector":
            n_components = len(field_spec["components"])
            field_data[field_name] = np.zeros((mesh.n_vertices, n_components))
    
    vertex_count = np.zeros(mesh.n_vertices)
    
    # Interpolate to vertices using averaging from adjacent cells
    for cell_id, cell in enumerate(mesh.cells):
        for vertex_id in cell:
            vertex_pos = mesh.vertices[vertex_id]
            vertex_fields = _evaluate_fields_at_point(solver, u_dofs, vertex_pos, cell_id, fields)
            
            for field_name, field_value in vertex_fields.items():
                field_data[field_name][vertex_id] += field_value
            
            vertex_count[vertex_id] += 1
    
    # Average values at vertices shared by multiple cells
    for field_name, field_spec in fields.items():
        if field_spec["type"] == "scalar":
            field_data[field_name] /= np.maximum(vertex_count, 1)
        elif field_spec["type"] == "vector":
            for i in range(mesh.n_vertices):
                if vertex_count[i] > 0:
                    field_data[field_name][i] /= vertex_count[i]
    
    _write_vtk_file(mesh, filename, fields, field_data, data_location="POINT")
    print(f"P1 vertex interpolation exported to: {filename}")


def _write_vtk_file(mesh, filename, fields, field_data, data_location="POINT"):
    """
    Write VTK file with mesh and field data.
    
    Parameters:
    -----------
    mesh : Mesh object
    filename : str
    fields : dict
        Field specifications
    field_data : dict
        {field_name: field_values_array}
    data_location : str
        "POINT" or "CELL"
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Solution with multiple fields\n")
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

        # Data section header
        if data_location == "POINT":
            f.write(f"\nPOINT_DATA {mesh.n_vertices}\n")
        else:  # CELL
            f.write(f"\nCELL_DATA {mesh.n_cells}\n")
        
        # Write all fields
        for field_name, field_spec in fields.items():
            field_type = field_spec["type"]
            data = field_data[field_name]
            
            if field_type == "scalar":
                f.write(f"SCALARS {field_name} double 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in data:
                    f.write(f"{val}\n")
                
            elif field_type == "vector":
                f.write(f"VECTORS {field_name} double\n")
                for i in range(len(data)):
                    vec = data[i]
                    # VTK vectors must be 3D
                    if len(vec) == 2:
                        f.write(f"{vec[0]} {vec[1]} 0.0\n")
                    elif len(vec) == 3:
                        f.write(f"{vec[0]} {vec[1]} {vec[2]}\n")
                    else:
                        raise ValueError(f"Unsupported vector dimension: {len(vec)}")


def project_and_export_to_triangular_mesh_vtk(solver, u_dofs, tria_mesh_file, 
                                              output_file="solution_tria.vtk", 
                                              fields=None):
    """
    Export solution to a triangular mesh where triangular vertices 
    correspond to polygonal mesh cell centroids.
    
    Parameters:
    -----------
    solver : Solver object
    u_dofs : array
        Solution DOF array from the polygonal mesh
    tria_mesh_file : str
        Path to the triangular mesh file (e.g., "mesh_tria.med")
    output_file : str
        Output VTK filename
    fields : dict or str
        Field specification (same format as export_to_vtk)
    """    
    # Convert simple string to dict format
    if isinstance(fields, str):
        fields = {fields: {"type": "scalar", "components": [0]}}
    elif fields is None:
        fields = {"u": {"type": "scalar", "components": [0]}}
    
    # Load triangular mesh
    print(f"Loading triangular mesh from {tria_mesh_file}...")
    tria_mesh = load_med_mesh_mc(tria_mesh_file)

    print(f"Triangular mesh has {tria_mesh.n_vertices} vertices")
    print(f"Polygonal mesh has {solver.mesh.n_cells} cells")
    
    # Build coordinate-based mapping: tria_vertex_id -> (poly_cell_id, evaluation_point)
    # The triangular mesh is the original mesh from which the polygonal mesh was derived
    # Each triangular vertex needs to be evaluated using the nearest polygon cell
    
    print("Building coordinate mapping between meshes...")
    vertex_mapping = []  # List of (poly_cell_id, point) for each tria vertex
    
    for tria_vtx_id in range(tria_mesh.n_vertices):
        tria_pos = tria_mesh.vertices[tria_vtx_id]
        
        # Find the nearest polygon cell centroid
        min_dist = float('inf')
        best_cell_id = -1
        
        for poly_cell_id in range(solver.mesh.n_cells):
            poly_cent = solver.mesh.cell_centroid(poly_cell_id)
            dist = np.linalg.norm(tria_pos - poly_cent)
            
            if dist < min_dist:
                min_dist = dist
                best_cell_id = poly_cell_id
        
        vertex_mapping.append((best_cell_id, tria_pos))
    
    print(f"Mapping complete.")

    # Initialize storage for all fields
    field_data = {}
    for field_name, field_spec in fields.items():
        if field_spec["type"] == "scalar":
            field_data[field_name] = np.zeros(tria_mesh.n_vertices)
        elif field_spec["type"] == "vector":
            n_components = len(field_spec["components"])
            field_data[field_name] = np.zeros((tria_mesh.n_vertices, n_components))
    
    # Evaluate solution at each triangular vertex position
    for tria_vtx_id in range(tria_mesh.n_vertices):
        poly_cell_id, eval_point = vertex_mapping[tria_vtx_id]
        
        # Evaluate solution at this point using the nearest cell context
        sol_values = solver.evaluate_solution(u_dofs, eval_point, poly_cell_id)
        
        # Convert to array format - handle tuple explicitly
        if isinstance(sol_values, tuple):
            sol_values = np.array(sol_values, dtype=float)
        elif not isinstance(sol_values, (list, np.ndarray)):
            sol_values = np.array([sol_values], dtype=float)
        else:
            sol_values = np.array(sol_values, dtype=float)
        
        # Extract each field based on component indices
        for field_name, field_spec in fields.items():
            components = field_spec["components"]
            field_type = field_spec["type"]
            
            if field_type == "scalar":
                # Extract single component
                field_data[field_name][tria_vtx_id] = sol_values[components[0]]
            elif field_type == "vector":
                # Extract multiple components
                for i, comp_idx in enumerate(components):
                    field_data[field_name][tria_vtx_id, i] = sol_values[comp_idx]
    
    _write_vtk_file(tria_mesh, output_file, fields, field_data, data_location="POINT")

    print(f"Solution exported to triangular mesh: {output_file}")
    print(f"  - Triangular mesh vertices: {tria_mesh.n_vertices}")
    print(f"  - Triangular mesh cells: {tria_mesh.n_cells}")