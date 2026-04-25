"""VTK file export utilities for scalar and vector fields."""

from pathlib import Path
import numpy as np
from .med_io import load_med_mesh_mc, _project_cell_data_to_nodes


def export_solution(solver, u_dofs, filename="solution.vtk", fields=None):
    """
    Export solution to VTK format with flexible field and projection specification.
    
    Parameters:
    -----------
    solver : Solver object
        The solver containing mesh and evaluate_solution method
    u_dofs : array
        Solution DOF array
    filename : str
        Output VTK filename
    fields : dict or str, optional
        Field specification. Can be:
        - str: single scalar field name (e.g., "u") → projects to "cell" by default
        - dict: {
            "field_name": {
                "type": "scalar"|"vector",
                "components": [indices],
                "projection": "cell"|"nodes",  # "cell" or "nodes" (default: "cell")
                "gradient": bool,              # compute gradient (default: False, scalars only)
                "gradient_magnitude": bool,    # compute gradient magnitude (default: False, scalars only)
                "zz_estimator": bool           # compute ZZ error estimator (default: False)
            }
        }
        
        Examples:
        - Simple scalar on cells: fields="u"
        - Explicit with projection: fields={
            "u": {
                "type": "scalar",
                "components": [0],
                "projection": "nodes"
            }
        }
        - With gradients: fields={
            "u": {
                "type": "scalar",
                "components": [0],
                "projection": "nodes",
                "gradient": True,
                "gradient_magnitude": True
            }
        }
        - With ZZ estimator: fields={
            "u": {
                "type": "scalar",
                "components": [0],
                "projection": "cell",
                "zz_estimator": True
            }
        }
    """
    if fields is None:
        raise ValueError("fields parameter is required")
    elif isinstance(fields, str):
        fields = {fields: {"type": "scalar", "components": [0], "projection": "cell"}}
    
    # Normalize field specifications (add defaults)
    for field_name, field_spec in fields.items():
        if "projection" not in field_spec:
            field_spec["projection"] = "cell"
        if "gradient" not in field_spec:
            field_spec["gradient"] = False
        if "gradient_magnitude" not in field_spec:
            field_spec["gradient_magnitude"] = False
        if "zz_estimator" not in field_spec:
            field_spec["zz_estimator"] = False
    
    # Export based on projections
    _export_vtk_multi(solver, u_dofs, filename, fields)


def _export_vtk_multi(solver, u_dofs, filename, fields):
    """
    Export multiple fields with different projections to a single VTK file.
    
    Parameters:
    -----------
    solver : Solver object
    u_dofs : array
        Solution DOF array
    filename : str
        Output VTK file name
    fields : dict
        Field specifications with normalized format
    """
    # Separate fields by projection type
    cell_fields = {}
    node_fields = {}
    
    for field_name, field_spec in fields.items():
        if field_spec["projection"] == "cell":
            cell_fields[field_name] = field_spec
        elif field_spec["projection"] == "nodes":
            node_fields[field_name] = field_spec
        else:
            raise ValueError(f"Unknown projection: {field_spec['projection']}")
    
    # If both cell and node projections, export to separate files
    if cell_fields and node_fields:
        # Export cell projection
        base_name = str(filename).replace('.vtk', '_cell.vtk')
        _export_vtk_p0(solver, u_dofs, base_name, cell_fields)
        
        # Export node projection
        base_name = str(filename).replace('.vtk', '_nodes.vtk')
        _export_vtk_p1_vertex(solver, u_dofs, base_name, node_fields)
    elif cell_fields:
        _export_vtk_p0(solver, u_dofs, filename, cell_fields)
    else:
        _export_vtk_p1_vertex(solver, u_dofs, filename, node_fields)


def _compute_gradients_numerical_vtk(solver, u_dofs, points, cell_ids, delta=1e-6):
    """
    Compute gradients numerically using finite differences.
    
    Parameters:
    -----------
    solver : Solver object
    u_dofs : array
        Solution DOF array
    points : array of shape (n, 2)
        Points at which to evaluate gradients
    cell_ids : array of shape (n,)
        Cell IDs for each point
    delta : float
        Step size for finite differences
    
    Returns:
    --------
    gradients : array of shape (n, 2)
        Gradients [du/dx, du/dy] at each point
    """
    n_points = len(points)
    gradients = np.zeros((n_points, 2))
    
    for i, (pt, cell_id) in enumerate(zip(points, cell_ids)):
        x, y = pt
        
        # Evaluate at displaced points
        u_x_plus = solver.evaluate_solution(u_dofs, np.array([x + delta, y]), cell_id)
        u_x_minus = solver.evaluate_solution(u_dofs, np.array([x - delta, y]), cell_id)
        u_y_plus = solver.evaluate_solution(u_dofs, np.array([x, y + delta]), cell_id)
        u_y_minus = solver.evaluate_solution(u_dofs, np.array([x, y - delta]), cell_id)
        
        # Handle vector results (take first component)
        if isinstance(u_x_plus, (list, np.ndarray)):
            u_x_plus = u_x_plus[0]
            u_x_minus = u_x_minus[0]
            u_y_plus = u_y_plus[0]
            u_y_minus = u_y_minus[0]
        
        du_dx = (u_x_plus - u_x_minus) / (2 * delta)
        du_dy = (u_y_plus - u_y_minus) / (2 * delta)
        
        gradients[i, 0] = du_dx
        gradients[i, 1] = du_dy
    
    return gradients


def _compute_zz_estimator(solver, u_dofs, component=0):
    """
    Compute the Zienkiewicz-Zhu (ZZ) error estimator on cells.
    
    The ZZ estimator is based on recovering a superconvergent gradient by
    L2-projection of element gradients onto nodes, then computing the error
    as the difference between the element gradient and the recovered gradient.
    This function computes the estimator as a cell-based quantity.
    
    Parameters:
    -----------
    solver : Solver object
        The solver with mesh and evaluate_solution method
    u_dofs : array
        Solution degrees of freedom
    component : int, optional
        Component index for scalar fields (default: 0)
    
    Returns:
    --------
    zz_error : array of shape (n_cells,)
        ZZ error estimator at each cell (||∇_h u - ∇* u||^2)
    """
    mesh = solver.mesh
    
    # Step 1: Compute element gradients at cell centroids
    cell_centroids = np.array([mesh.cell_centroid(cid) for cid in range(mesh.n_cells)])
    cell_ids = np.arange(mesh.n_cells)
    
    # Get gradients at cell centroids (element gradients)
    grad_element = _compute_gradients_numerical_vtk(solver, u_dofs, cell_centroids, cell_ids)
    # grad_element shape: (n_cells, 2)
    
    # Step 2: Project element gradients to vertices (L2 projection / averaging)
    # This gives us the "recovered" or "smoothed" gradient
    grad_recovered_vertices = np.zeros((mesh.n_vertices, 2))
    vertex_count = np.zeros(mesh.n_vertices)
    
    for cell_id, cell in enumerate(mesh.cells):
        for vertex_id in cell:
            grad_recovered_vertices[vertex_id] += grad_element[cell_id]
            vertex_count[vertex_id] += 1
    
    # Average gradients at vertices shared by multiple cells
    grad_recovered_vertices /= np.maximum(vertex_count[:, np.newaxis], 1)
    
    # Step 3: Compute error estimator as ||∇_h - ∇*||^2 per cell
    # Average recovered gradient over cell vertices, then compute L2 error
    zz_error = np.zeros(mesh.n_cells)
    
    for cell_id, cell in enumerate(mesh.cells):
        # Average recovered gradient over vertices of this cell
        grad_recovered_cell = np.mean(grad_recovered_vertices[cell], axis=0)
        # Error: ||∇_h - ∇*||^2
        grad_diff = grad_element[cell_id] - grad_recovered_cell
        zz_error[cell_id] = np.sum(grad_diff**2)
    
    return zz_error


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
    """Export with P0 projection (cell-centered values) with optional gradients."""
    mesh = solver.mesh
    
    # Initialize storage for all fields
    field_data = {}
    grad_data = {}
    grad_mag_data = {}
    zz_estimator_data = {}
    cell_centroids = np.zeros((mesh.n_cells, 2))
    
    for field_name, field_spec in fields.items():
        if field_spec["type"] == "scalar":
            field_data[field_name] = np.zeros(mesh.n_cells)
            if field_spec["gradient"]:
                grad_data[field_name] = np.zeros((mesh.n_cells, 2))
            if field_spec["gradient_magnitude"]:
                grad_mag_data[field_name] = np.zeros(mesh.n_cells)
            if field_spec["zz_estimator"]:
                zz_estimator_data[field_name] = np.zeros(mesh.n_cells)
        elif field_spec["type"] == "vector":
            n_components = len(field_spec["components"])
            field_data[field_name] = np.zeros((mesh.n_cells, n_components))
    
    # Compute gradients once if needed
    compute_any_gradient = any(f.get("gradient", False) or f.get("gradient_magnitude", False) or f.get("zz_estimator", False)
                               for f in fields.values() if f["type"] == "scalar")
    if compute_any_gradient:
        for cell_id in range(mesh.n_cells):
            cell_centroids[cell_id] = mesh.cell_centroid(cell_id)
        all_gradients = _compute_gradients_numerical_vtk(solver, u_dofs, cell_centroids, 
                                                         np.arange(mesh.n_cells))
    
    # Evaluate at cell centroids
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        cell_fields = _evaluate_fields_at_point(solver, u_dofs, cent, cell_id, fields)
        
        for field_name, field_value in cell_fields.items():
            field_data[field_name][cell_id] = field_value
        
        # Store gradient if needed
        if compute_any_gradient:
            for field_name, field_spec in fields.items():
                if field_spec["type"] == "scalar":
                    if field_spec.get("gradient", False):
                        grad_data[field_name][cell_id] = all_gradients[cell_id]
                    if field_spec.get("gradient_magnitude", False):
                        grad_mag_data[field_name][cell_id] = np.linalg.norm(all_gradients[cell_id])
    
    # Compute ZZ estimator if needed
    for field_name, field_spec in fields.items():
        if field_spec.get("zz_estimator", False) and field_name in zz_estimator_data:
            component_idx = field_spec["components"][0]  # Use first component for scalar
            zz_estimator_data[field_name] = _compute_zz_estimator(solver, u_dofs, component=component_idx)
    
    _write_vtk_file(mesh, filename, fields, field_data, grad_data, grad_mag_data,
                    zz_estimator_data, data_location="CELL")
    print(f"P0 projection exported to: {filename}")


def _export_vtk_p1_vertex(solver, u_dofs, filename, fields):
    """Export with P1 vertex interpolation (vertex-centered values) with optional gradients."""
    mesh = solver.mesh
    
    # Initialize storage for all fields
    field_data = {}
    grad_data = {}
    grad_mag_data = {}
    zz_estimator_data = {}
    vertex_count = np.zeros(mesh.n_vertices)
    
    for field_name, field_spec in fields.items():
        if field_spec["type"] == "scalar":
            field_data[field_name] = np.zeros(mesh.n_vertices)
            if field_spec.get("gradient", False):
                grad_data[field_name] = np.zeros((mesh.n_vertices, 2))
            if field_spec.get("gradient_magnitude", False):
                grad_mag_data[field_name] = np.zeros(mesh.n_vertices)
            if field_spec.get("zz_estimator", False):
                zz_estimator_data[field_name] = np.zeros(mesh.n_cells)
        elif field_spec["type"] == "vector":
            n_components = len(field_spec["components"])
            field_data[field_name] = np.zeros((mesh.n_vertices, n_components))
    
    # Compute cell gradients once if needed
    compute_any_gradient = any(f.get("gradient", False) or f.get("gradient_magnitude", False) or f.get("zz_estimator", False)
                               for f in fields.values() if f["type"] == "scalar")
    cell_gradients = None
    
    if compute_any_gradient:
        cell_centroids = np.array([mesh.cell_centroid(cid) for cid in range(mesh.n_cells)])
        cell_gradients = _compute_gradients_numerical_vtk(solver, u_dofs, cell_centroids, 
                                                          np.arange(mesh.n_cells))
    
    # Interpolate to vertices using averaging from adjacent cells
    for cell_id, cell in enumerate(mesh.cells):
        for vertex_id in cell:
            vertex_pos = mesh.vertices[vertex_id]
            vertex_fields = _evaluate_fields_at_point(solver, u_dofs, vertex_pos, cell_id, fields)
            
            for field_name, field_value in vertex_fields.items():
                field_data[field_name][vertex_id] += field_value
            
            # Accumulate gradients
            if compute_any_gradient:
                for field_name, field_spec in fields.items():
                    if field_spec["type"] == "scalar":
                        if field_spec.get("gradient", False):
                            grad_data[field_name][vertex_id] += cell_gradients[cell_id]
                        if field_spec.get("gradient_magnitude", False):
                            grad_mag_data[field_name][vertex_id] += np.linalg.norm(cell_gradients[cell_id])
            
            vertex_count[vertex_id] += 1
    
    # Compute ZZ estimator if needed
    for field_name, field_spec in fields.items():
        if field_spec.get("zz_estimator", False) and field_name in zz_estimator_data:
            component_idx = field_spec["components"][0]  # Use first component for scalar
            # Compute ZZ error estimator on cells, then project to nodes
            zz_cell_data = _compute_zz_estimator(solver, u_dofs, component=component_idx)
            zz_estimator_data[field_name] = _project_cell_data_to_nodes(zz_cell_data, mesh)
    
    # Average values at vertices shared by multiple cells
    for field_name, field_spec in fields.items():
        if field_spec["type"] == "scalar":
            field_data[field_name] /= np.maximum(vertex_count, 1)
            if field_spec.get("gradient", False):
                grad_data[field_name] /= np.maximum(vertex_count[:, np.newaxis], 1)
            if field_spec.get("gradient_magnitude", False):
                grad_mag_data[field_name] /= np.maximum(vertex_count, 1)
        elif field_spec["type"] == "vector":
            for i in range(mesh.n_vertices):
                if vertex_count[i] > 0:
                    field_data[field_name][i] /= vertex_count[i]
    
    _write_vtk_file(mesh, filename, fields, field_data, grad_data, grad_mag_data,
                    zz_estimator_data, data_location="POINT")
    print(f"P1 vertex interpolation exported to: {filename}")


def _write_vtk_file(mesh, filename, fields, field_data, grad_data=None, grad_mag_data=None,
                    zz_estimator_data=None, data_location="POINT"):
    """
    Write VTK file with mesh and field data, including optional gradients and ZZ estimator.
    
    Parameters:
    -----------
    mesh : Mesh object
    filename : str
    fields : dict
        Field specifications
    field_data : dict
        {field_name: field_values_array}
    grad_data : dict, optional
        {field_name: gradient_values_array} for scalar fields with gradients
    grad_mag_data : dict, optional
        {field_name: gradient_magnitude_values_array} for scalar fields with gradient_magnitude
    zz_estimator_data : dict, optional
        {field_name: zz_estimator_values_array} for scalar fields with ZZ estimator
    data_location : str
        "POINT" or "CELL"
    """
    if grad_data is None:
        grad_data = {}
    if grad_mag_data is None:
        grad_mag_data = {}
    if zz_estimator_data is None:
        zz_estimator_data = {}
        
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
        
        # Write gradient fields if present
        for field_name in grad_data:
            grad = grad_data[field_name]
            f.write(f"VECTORS {field_name}_gradient double\n")
            for i in range(len(grad)):
                g = grad[i]
                # 2D gradient, pad to 3D for VTK
                f.write(f"{g[0]} {g[1]} 0.0\n")
        
        # Write gradient magnitude fields if present
        for field_name in grad_mag_data:
            mag = grad_mag_data[field_name]
            f.write(f"SCALARS {field_name}_gradient_magnitude double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in mag:
                f.write(f"{val}\n")
        
        # Write ZZ estimator fields if present
        for field_name in zz_estimator_data:
            zz = zz_estimator_data[field_name]
            f.write(f"SCALARS zz_estimator_{field_name} double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in zz:
                f.write(f"{val}\n")


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
        Field specification (same format as export_solution)
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