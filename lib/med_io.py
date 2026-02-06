"""Mesh I/O utilities for various formats (MED, VTK)."""

try:
    import medcoupling as mc
except ImportError:
    print("MEDCoupling not found. Install it from SALOME or compile from source.")
    print("You can access it via: salome shell -- python")
    raise

import numpy as np
from .polygonal_mesh import PolygonalMesh


def load_med_mesh_mc(filename, mesh_name=None, mesh_level=0):
    """
    Load a 2D mesh from a MED file using MEDCoupling.
    
    Parameters:
    -----------
    filename : str
        Path to the MED file
    mesh_name : str, optional
        Name of the mesh to read. If None, reads the first mesh.
    mesh_level : int, default=0
        Mesh level (0 for highest dimension cells)
    
    Returns:
    --------
    PolygonalMesh
        A PolygonalMesh object containing vertices, cells, and boundary edges
    """
    # Read the MED file
    med_mesh = mc.MEDFileMesh.New(filename)
    
    # Get mesh name if not provided
    if mesh_name is None:
        mesh_name = med_mesh.getName()
        print(f"Reading mesh: {mesh_name}")
    
    # Get the mesh at specified level
    umesh = med_mesh.getMeshAtLevel(mesh_level)
    
    # Merge duplicate nodes (important for proper connectivity)
    print(f"Nodes before merge: {umesh.getNumberOfNodes()}")
    umesh.mergeNodes(1e-10)
    print(f"Nodes after merge: {umesh.getNumberOfNodes()}")
    
    # Extract coordinates (only 2D)
    coords = umesh.getCoords()
    vertices = coords.toNumPyArray()[:, :2]
    
    # Extract cells
    cells = []
    n_cells = umesh.getNumberOfCells()
    
    for i in range(n_cells):
        # Get connectivity for this cell
        cell_conn = umesh.getNodeIdsOfCell(i)
        cells.append(list(cell_conn))
    
    print(f"Loaded {len(vertices)} vertices and {len(cells)} cells")
    
    # Extract boundary edges if available
    boundary_edge_tuples = set()
    
    try:
        # Try to get boundary mesh (mesh at level -1)
        boundary_mesh = med_mesh.getMeshAtLevel(-1)
        n_boundary_cells = boundary_mesh.getNumberOfCells()
        
        print(f"Found {n_boundary_cells} boundary edges")
        
        for i in range(n_boundary_cells):
            edge_conn = boundary_mesh.getNodeIdsOfCell(i)
            if len(edge_conn) >= 2:
                v1, v2 = edge_conn[0], edge_conn[1]
                boundary_edge_tuples.add(tuple(sorted((v1, v2))))
    except:
        print("No explicit boundary edges found. Will identify from connectivity.")
    
    # Create PolygonalMesh
    poly_mesh = PolygonalMesh(vertices, cells)
    
    # Map boundary edges to mesh edge indices
    if boundary_edge_tuples:
        poly_boundary_edges = [
            i for i, e in enumerate(poly_mesh.edges)
            if tuple(sorted(e)) in boundary_edge_tuples
        ]
    else:
        # Use default boundary detection (edges with only one adjacent cell)
        poly_boundary_edges = poly_mesh.boundary_edges
    
    poly_mesh.boundary_edges = poly_boundary_edges
    print(f"Identified {len(poly_boundary_edges)} boundary edges")
    
    return poly_mesh

def load_med_mesh_with_groups(filename, mesh_name=None, mesh_level=0):
    """
    Load a mesh with group information from MED file.
    Returns mesh and dictionary of groups.
    """
    med_mesh = mc.MEDFileMesh.New(filename)
    
    if mesh_name is None:
        mesh_name = med_mesh.getName()
    
    umesh = med_mesh.getMeshAtLevel(mesh_level)
    umesh.mergeNodes(1e-10)
    
    # Extract groups
    groups = {}
    try:
        group_names = med_mesh.getGroupsNames()
        print(f"Found groups: {group_names}")
        
        for group_name in group_names:
            group_arr = med_mesh.getGroupArr(mesh_level, group_name)
            groups[group_name] = group_arr.toNumPyArray()
    except:
        print("No groups found in mesh")
    
    # Convert to PolygonalMesh
    coords = umesh.getCoords()
    vertices = coords.toNumPyArray()[:, :2]
    
    cells = []
    for i in range(umesh.getNumberOfCells()):
        cell_conn = umesh.getNodeIdsOfCell(i)
        cells.append(list(cell_conn))
    
    poly_mesh = PolygonalMesh(vertices, cells)
    
    # Get boundary edges
    try:
        boundary_mesh = med_mesh.getMeshAtLevel(-1)
        boundary_edge_tuples = set()
        
        for i in range(boundary_mesh.getNumberOfCells()):
            edge_conn = boundary_mesh.getNodeIdsOfCell(i)
            if len(edge_conn) >= 2:
                v1, v2 = edge_conn[0], edge_conn[1]
                boundary_edge_tuples.add(tuple(sorted((v1, v2))))
        
        poly_boundary_edges = [
            i for i, e in enumerate(poly_mesh.edges)
            if tuple(sorted(e)) in boundary_edge_tuples
        ]
        poly_mesh.boundary_edges = poly_boundary_edges
    except:
        pass
    
    return poly_mesh, groups


def extract_edge_groups_from_med(filename, mesh_name=None):
    """
    Extract edge groups from MED file for boundary condition specification.
    
    Parameters:
    -----------
    filename : str
        Path to MED file
    mesh_name : str, optional
        Name of mesh to read
    
    Returns:
    --------
    dict
        Dictionary mapping group names to arrays of global edge indices
        Format: {'group_name': [edge_idx1, edge_idx2, ...]}
    """
    med_mesh = mc.MEDFileMesh.New(filename)
    
    if mesh_name is None:
        mesh_name = med_mesh.getName()
    
    # Get the boundary mesh (level -1)
    try:
        boundary_mesh = med_mesh.getMeshAtLevel(-1)
    except:
        print("No boundary mesh found at level -1")
        return {}
    
    # Get volume mesh to build edge mapping
    umesh = med_mesh.getMeshAtLevel(0)
    umesh.mergeNodes(1e-10)
    
    # Build edge to index mapping from volume mesh
    edge_to_idx = {}
    edge_list = []
    
    # Extract all edges from volume mesh cells
    for cell_id in range(umesh.getNumberOfCells()):
        cell_conn = umesh.getNodeIdsOfCell(cell_id)
        n_verts = len(cell_conn)
        for i in range(n_verts):
            v1, v2 = cell_conn[i], cell_conn[(i+1) % n_verts]
            edge = tuple(sorted([v1, v2]))
            if edge not in edge_to_idx:
                edge_to_idx[edge] = len(edge_list)
                edge_list.append(edge)
    
    # Extract groups from boundary mesh
    edge_groups = {}
    
    try:
        group_names = med_mesh.getGroupsNames()
        print(f"Found boundary groups: {group_names}")
        
        for group_name in group_names:
            try:
                # Get cell IDs in this group at boundary level (-1)
                group_arr = med_mesh.getGroupArr(-1, group_name)
                boundary_cell_ids = group_arr.toNumPyArray()
                
                # Map boundary cells to global edge indices
                group_edge_indices = []
                for bcell_id in boundary_cell_ids:
                    edge_conn = boundary_mesh.getNodeIdsOfCell(int(bcell_id))
                    if len(edge_conn) >= 2:
                        v1, v2 = edge_conn[0], edge_conn[1]
                        edge = tuple(sorted([v1, v2]))
                        if edge in edge_to_idx:
                            group_edge_indices.append(edge_to_idx[edge])
                
                edge_groups[group_name] = np.array(group_edge_indices)
                print(f"  Group '{group_name}': {len(group_edge_indices)} edges")
                
            except Exception as e:
                print(f"  Could not load group '{group_name}': {e}")
                continue
                
    except Exception as e:
        print(f"Error reading groups: {e}")
    
    return edge_groups


def _evaluate_fields_at_point_med(solver, u_dofs, point, cell_id, fields):
    """
    Evaluate all fields at a given point.
    
    Parameters:
    -----------
    solver : Solver object
    u_dofs : array
        Solution DOF array
    point : array-like
        Point coordinates [x, y]
    cell_id : int
        Cell ID for evaluation context
    fields : dict
        Field specifications: {name: {"type": "scalar"|"vector", "components": [indices]}}
    
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
            field_values[field_name] = float(sol_values[components[0]])
        elif field_type == "vector":
            # Multiple components
            field_values[field_name] = np.array([sol_values[i] for i in components], dtype=float)
        else:
            raise ValueError(f"Unknown field type: {field_type}")
    
    return field_values


def export_to_med(solver, u_dofs, filename="solution.med", field_name="u",
                  method="P0", fields=None):
    """
    Export solution to MED format with flexible field specification.
    
    Parameters:
    -----------
    solver : Solver object
        The solver containing mesh and evaluate_solution method
    u_dofs : array
        Solution DOF array
    filename : str
        Output MED filename
    field_name : str
        Legacy parameter (used if fields is None). Default field name.
    method : str
        Export method: "P0", "P1_vertex", "P0_P1", "P0_P1_gradients", "P0_P1_gradient_mag"
    fields : dict or str, optional
        Field specification. Can be:
        - str: single scalar field name (e.g., "u" for Poisson)
        - dict: {"field_name": {"type": "scalar"|"vector", "components": [indices]}}
        
        Examples:
        - Poisson: fields="u" or fields={"u": {"type": "scalar", "components": [0]}}
        - Stokes: fields={
                      "velocity": {"type": "vector", "components": [0, 1]},
                      "pressure": {"type": "scalar", "components": [2]}
                  }
    
    If fields is None, uses the legacy field_name parameter for backward compatibility.
    """
    if mc is None:
        raise ImportError("MEDCoupling (mc) is required to export MED files")
    
    # Convert simple string to dict format
    if fields is None:
        fields = {field_name: {"type": "scalar", "components": [0]}}
    elif isinstance(fields, str):
        fields = {fields: {"type": "scalar", "components": [0]}}
    
    # For legacy methods that don't support multiple fields, handle specially
    if method in ["P0_P1", "P0_P1_gradients", "P0_P1_gradient_mag"]:
        # These methods create their own field naming, use field_name as base
        if method == "P0_P1":
            _export_med_p0_p1(solver, u_dofs, filename, field_name + "_P0", field_name + "_P1")
        elif method == "P0_P1_gradients":
            _export_med_p0_p1_gradients(solver, u_dofs, filename, field_name)
        elif method == "P0_P1_gradient_mag":
            _export_med_p0_p1_gradient_mag(solver, u_dofs, filename, field_name)
    elif method == "P0":
        _export_med_p0_multi(solver, u_dofs, filename, fields)
    elif method == "P1_vertex":
        _export_med_p1_vertex_multi(solver, u_dofs, filename, fields)
    else:
        raise ValueError(f"Unknown export method: {method}")


def _export_med_p0_multi(solver, u_dofs, filename, fields):
    """
    Export multiple fields at P0 (cell-centered) to MED format.
    
    Parameters:
    -----------
    solver : Solver object
    u_dofs : array
        Solution DOF array
    filename : str
        Output MED file name
    fields : dict
        Field specifications: {name: {"type": "scalar"|"vector", "components": [indices]}}
    """
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
        cell_fields = _evaluate_fields_at_point_med(solver, u_dofs, cent, cell_id, fields)
        
        for field_name, field_value in cell_fields.items():
            field_data[field_name][cell_id] = field_value

    # Create MEDCoupling mesh
    coords_array = mesh.vertices
    if coords_array.shape[1] == 2:
        # Add z=0 coordinate
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    # Create coordinate array
    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    # Create unstructured mesh
    umesh = mc.MEDCouplingUMesh("solution_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group cells by type for MEDCoupling contiguity requirement
    cells_by_type = {}
    for cell_id, cell in enumerate(mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
        cells_by_type[cell_type].append((cell_id, cell))

    # Build mapping from new cell order to original cell_id
    cell_mapping = []
    umesh.allocateCells(mesh.n_cells)

    # Insert cells grouped by type
    for cell_type in sorted(cells_by_type.keys()):
        for cell_id, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)
            cell_mapping.append(cell_id)

    umesh.finishInsertingCells()

    # Write MED file with mesh
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    # Write all fields to MED file
    for field_name, field_spec in fields.items():
        # Reorder field values according to cell_mapping
        field_reordered = field_data[field_name][cell_mapping]
        
        # Create MEDCoupling field
        field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
        field.setName(field_name)
        field.setMesh(umesh)
        field.setTime(0.0, 0, 0)  # time, iteration, order

        # Set field values
        if field_spec["type"] == "scalar":
            field_array = mc.DataArrayDouble(field_reordered)
            field_array.setInfoOnComponent(0, field_name)
        elif field_spec["type"] == "vector":
            n_components = len(field_spec["components"])
            # Use MEDCoupling's constructor with (data, num_tuples, num_components)
            field_array = mc.DataArrayDouble(field_reordered.ravel().tolist(), len(field_reordered), n_components)
            for i, comp_idx in enumerate(field_spec["components"]):
                field_array.setInfoOnComponent(i, f"{field_name}_{i}")
        
        field.setArray(field_array)
        field.checkConsistencyLight()

        # Write field to MED file
        med_writer = mc.MEDFileField1TS()
        med_writer.setFieldNoProfileSBT(field)
        med_writer.write(filename, 0)  # 0 = append mode

    print(f"P0 projection exported to MED: {filename}")
    print(f"  Fields: {', '.join(fields.keys())}")


def _export_med_p1_vertex_multi(solver, u_dofs, filename, fields):
    """
    Export multiple fields at P1 vertex (vertex-centered with averaging) to MED format.
    
    Parameters:
    -----------
    solver : Solver object
    u_dofs : array
        Solution DOF array
    filename : str
        Output MED file name
    fields : dict
        Field specifications: {name: {"type": "scalar"|"vector", "components": [indices]}}
    """
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
            vertex_fields = _evaluate_fields_at_point_med(solver, u_dofs, vertex_pos, cell_id, fields)
            
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

    # Create MEDCoupling mesh
    coords_array = mesh.vertices
    if coords_array.shape[1] == 2:
        # Add z=0 coordinate
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    # Create coordinate array
    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    # Create unstructured mesh
    umesh = mc.MEDCouplingUMesh("solution_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group cells by type for MEDCoupling contiguity requirement
    cells_by_type = {}
    for cell_id, cell in enumerate(mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
        cells_by_type[cell_type].append((cell_id, cell))

    umesh.allocateCells(mesh.n_cells)

    # Insert cells grouped by type
    for cell_type in sorted(cells_by_type.keys()):
        for cell_id, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)

    umesh.finishInsertingCells()

    # Write MED file with mesh
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    # Write all fields to MED file
    for field_name, field_spec in fields.items():
        # Create MEDCoupling field on nodes (no reordering needed for node fields)
        field = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
        field.setName(field_name)
        field.setMesh(umesh)
        field.setTime(0.0, 0, 0)

        # Set field values
        if field_spec["type"] == "scalar":
            field_array = mc.DataArrayDouble(field_data[field_name])
            field_array.setInfoOnComponent(0, field_name)
        elif field_spec["type"] == "vector":
            n_components = len(field_spec["components"])
            # Use MEDCoupling's constructor with (data, num_tuples, num_components)
            field_array = mc.DataArrayDouble(field_data[field_name].ravel().tolist(), len(field_data[field_name]), n_components)
            for i, comp_idx in enumerate(field_spec["components"]):
                field_array.setInfoOnComponent(i, f"{field_name}_{i}")
        
        field.setArray(field_array)
        field.checkConsistencyLight()

        # Write field to MED file
        med_writer = mc.MEDFileField1TS()
        med_writer.setFieldNoProfileSBT(field)
        med_writer.write(filename, 0)  # 0 = append mode

    print(f"P1 vertex interpolation exported to MED: {filename}")
    print(f"  Fields: {', '.join(fields.keys())}")


def _export_med_p1_vertex(solver, u_dofs, filename, field_name):
    """Legacy function for backward compatibility. Use _export_med_p1_vertex_multi with fields dict."""
    fields = {field_name: {"type": "scalar", "components": [0]}}
    _export_med_p1_vertex_multi(solver, u_dofs, filename, fields)



def _export_med_p0_p1(solver, u_dofs, filename, field_name_p0, field_name_p1):
    """
    Export both P0 (cell-centered) and P1 (vertex-based) projections to the same MED file.
    
    Parameters:
    -----------
    solver : Solver object
        The solver with the mesh
    u_dofs : array
        Solution degrees of freedom
    filename : str
        Output MED file name
    field_name_p0 : str
        Name for P0 field
    field_name_p1 : str
        Name for P1 field
    """
    mesh = solver.mesh
    
    # Compute P0 values at cell centroids
    u_cells = np.zeros(mesh.n_cells)
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        u_cells[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)
    
    # Compute P1 values at vertices using averaging from adjacent cells
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

    # Create MEDCoupling mesh
    coords_array = mesh.vertices
    if coords_array.shape[1] == 2:
        # Add z=0 coordinate
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    # Create coordinate array
    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    # Create unstructured mesh
    umesh = mc.MEDCouplingUMesh("solution_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group cells by type for MEDCoupling contiguity requirement
    cells_by_type = {}
    for cell_id, cell in enumerate(mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
        cells_by_type[cell_type].append((cell_id, cell))

    # Build mapping from new cell order to original cell_id
    cell_mapping = []
    umesh.allocateCells(mesh.n_cells)

    # Insert cells grouped by type
    for cell_type in sorted(cells_by_type.keys()):
        for cell_id, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)
            cell_mapping.append(cell_id)

    umesh.finishInsertingCells()

    # Reorder P0 field values according to cell_mapping
    u_cells_reordered = u_cells[cell_mapping]

    # Create P0 field on cells
    field_p0 = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field_p0.setName(field_name_p0)
    field_p0.setMesh(umesh)
    field_p0.setTime(0.0, 0, 0)

    # Set P0 field values
    field_array_p0 = mc.DataArrayDouble(u_cells_reordered)
    field_array_p0.setInfoOnComponent(0, field_name_p0)
    field_p0.setArray(field_array_p0)
    field_p0.checkConsistencyLight()

    # Create P1 field on nodes
    field_p1 = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field_p1.setName(field_name_p1)
    field_p1.setMesh(umesh)
    field_p1.setTime(0.0, 0, 0)

    # Set P1 field values
    field_array_p1 = mc.DataArrayDouble(u_vertices)
    field_array_p1.setInfoOnComponent(0, field_name_p1)
    field_p1.setArray(field_array_p1)
    field_p1.checkConsistencyLight()

    # Write MED file with mesh
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    # Write P0 field
    med_writer_p0 = mc.MEDFileField1TS()
    med_writer_p0.setFieldNoProfileSBT(field_p0)
    med_writer_p0.write(filename, 0)  # 0 = append mode

    # Write P1 field
    med_writer_p1 = mc.MEDFileField1TS()
    med_writer_p1.setFieldNoProfileSBT(field_p1)
    med_writer_p1.write(filename, 0)  # 0 = append mode

    print(f"P0 and P1 projections exported to MED: {filename}")


def _export_med_p0(solver, u_dofs, filename, field_name):
    """Legacy function for backward compatibility. Use _export_med_p0_multi with fields dict."""
    fields = {field_name: {"type": "scalar", "components": [0]}}
    _export_med_p0_multi(solver, u_dofs, filename, fields)

def _compute_gradients_numerical(solver, u_dofs, points, cell_ids, delta=1e-6):
    """
    Compute gradients numerically using finite differences.
    
    Parameters:
    -----------
    solver : Solver object
    u_dofs : array
        Solution degrees of freedom
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
        # Compute gradient using central differences
        x, y = pt
        
        # Evaluate at displaced points
        u_x_plus = solver.evaluate_solution(u_dofs, np.array([x + delta, y]), cell_id)
        u_x_minus = solver.evaluate_solution(u_dofs, np.array([x - delta, y]), cell_id)
        u_y_plus = solver.evaluate_solution(u_dofs, np.array([x, y + delta]), cell_id)
        u_y_minus = solver.evaluate_solution(u_dofs, np.array([x, y - delta]), cell_id)
        
        # Compute derivatives
        du_dx = (u_x_plus - u_x_minus) / (2 * delta)
        du_dy = (u_y_plus - u_y_minus) / (2 * delta)
        
        gradients[i, 0] = du_dx
        gradients[i, 1] = du_dy
    
    return gradients

def _export_med_p0_p1_gradients(solver, u_dofs, filename, field_name):
    """
    Export P0 and P1 fields along with their gradients to the same MED file.
    
    Creates four fields:
    - {field_name}_P0: P0 projection (cell-centered)
    - {field_name}_P1: P1 projection (vertex-based)
    - gradient_{field_name}_P0: P0 gradients (cell-centered, 2 components)
    - gradient_{field_name}_P1: P1 gradients (vertex-based, 2 components)
    
    Parameters:
    -----------
    solver : Solver object
        The solver with the mesh
    u_dofs : array
        Solution degrees of freedom
    filename : str
        Output MED file name
    field_name : str
        Base name for fields
    """
    mesh = solver.mesh
    
    # Compute P0 values and gradients at cell centroids
    u_cells = np.zeros(mesh.n_cells)
    grad_cells = np.zeros((mesh.n_cells, 2))
    cell_centroids = np.zeros((mesh.n_cells, 2))
    
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        cell_centroids[cell_id] = cent
        u_cells[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)
    
    grad_cells = _compute_gradients_numerical(solver, u_dofs, cell_centroids, 
                                               np.arange(mesh.n_cells))
    
    # Compute P1 values and gradients at vertices using averaging from adjacent cells
    u_vertices = np.zeros(mesh.n_vertices)
    grad_vertices = np.zeros((mesh.n_vertices, 2))
    vertex_count = np.zeros(mesh.n_vertices)

    for cell_id, cell in enumerate(mesh.cells):
        for vertex_id in cell:
            vertex_pos = mesh.vertices[vertex_id]
            u_val = solver.evaluate_solution(u_dofs, vertex_pos, cell_id)
            u_vertices[vertex_id] += u_val
            grad_vertices[vertex_id] += grad_cells[cell_id]
            vertex_count[vertex_id] += 1

    # Average values and gradients at vertices shared by multiple cells
    u_vertices /= np.maximum(vertex_count, 1)
    grad_vertices /= np.maximum(vertex_count[:, np.newaxis], 1)

    # Create MEDCoupling mesh
    coords_array = mesh.vertices
    if coords_array.shape[1] == 2:
        # Add z=0 coordinate
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    # Create coordinate array
    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    # Create unstructured mesh
    umesh = mc.MEDCouplingUMesh("solution_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group cells by type for MEDCoupling contiguity requirement
    cells_by_type = {}
    for cell_id, cell in enumerate(mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
        cells_by_type[cell_type].append((cell_id, cell))

    # Build mapping from new cell order to original cell_id
    cell_mapping = []
    umesh.allocateCells(mesh.n_cells)

    # Insert cells grouped by type
    for cell_type in sorted(cells_by_type.keys()):
        for cell_id, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)
            cell_mapping.append(cell_id)

    umesh.finishInsertingCells()

    # Reorder field values according to cell_mapping
    u_cells_reordered = u_cells[cell_mapping]
    grad_cells_reordered = grad_cells[cell_mapping]

    # Create P0 field on cells
    field_p0 = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field_p0.setName(field_name + "_P0")
    field_p0.setMesh(umesh)
    field_p0.setTime(0.0, 0, 0)

    field_array_p0 = mc.DataArrayDouble(u_cells_reordered)
    field_array_p0.setInfoOnComponent(0, field_name + "_P0")
    field_p0.setArray(field_array_p0)
    field_p0.checkConsistencyLight()

    # Create P0 gradient field on cells (2 components: dx, dy)
    field_grad_p0 = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field_grad_p0.setName("gradient_" + field_name + "_P0")
    field_grad_p0.setMesh(umesh)
    field_grad_p0.setTime(0.0, 0, 0)

    field_array_grad_p0 = mc.DataArrayDouble(grad_cells_reordered)
    field_array_grad_p0.setInfoOnComponent(0, "d" + field_name + "/dx")
    field_array_grad_p0.setInfoOnComponent(1, "d" + field_name + "/dy")
    field_grad_p0.setArray(field_array_grad_p0)
    field_grad_p0.checkConsistencyLight()

    # Create P1 field on nodes
    field_p1 = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field_p1.setName(field_name + "_P1")
    field_p1.setMesh(umesh)
    field_p1.setTime(0.0, 0, 0)

    field_array_p1 = mc.DataArrayDouble(u_vertices)
    field_array_p1.setInfoOnComponent(0, field_name + "_P1")
    field_p1.setArray(field_array_p1)
    field_p1.checkConsistencyLight()

    # Create P1 gradient field on nodes (2 components: dx, dy)
    field_grad_p1 = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field_grad_p1.setName("gradient_" + field_name + "_P1")
    field_grad_p1.setMesh(umesh)
    field_grad_p1.setTime(0.0, 0, 0)

    field_array_grad_p1 = mc.DataArrayDouble(grad_vertices)
    field_array_grad_p1.setInfoOnComponent(0, "d" + field_name + "/dx")
    field_array_grad_p1.setInfoOnComponent(1, "d" + field_name + "/dy")
    field_grad_p1.setArray(field_array_grad_p1)
    field_grad_p1.checkConsistencyLight()

    # Write MED file with mesh
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    # Write all fields
    med_writer_p0 = mc.MEDFileField1TS()
    med_writer_p0.setFieldNoProfileSBT(field_p0)
    med_writer_p0.write(filename, 0)

    med_writer_grad_p0 = mc.MEDFileField1TS()
    med_writer_grad_p0.setFieldNoProfileSBT(field_grad_p0)
    med_writer_grad_p0.write(filename, 0)

    med_writer_p1 = mc.MEDFileField1TS()
    med_writer_p1.setFieldNoProfileSBT(field_p1)
    med_writer_p1.write(filename, 0)

    med_writer_grad_p1 = mc.MEDFileField1TS()
    med_writer_grad_p1.setFieldNoProfileSBT(field_grad_p1)
    med_writer_grad_p1.write(filename, 0)

    print(f"P0 and P1 fields with gradients exported to MED: {filename}")
    print(f"  Fields: {field_name}_P0, {field_name}_P1, gradient_{field_name}_P0, gradient_{field_name}_P1")

def _export_med_p0_p1_gradient_mag(solver, u_dofs, filename, field_name):
    """
    Export P0 and P1 fields along with their gradient magnitudes to the same MED file.
    
    Creates four fields:
    - {field_name}_P0: P0 projection (cell-centered)
    - {field_name}_P1: P1 projection (vertex-based)
    - gradient_mag_{field_name}_P0: P0 gradient magnitude (cell-centered scalar)
    - gradient_mag_{field_name}_P1: P1 gradient magnitude (vertex-based scalar)
    
    Parameters:
    -----------
    solver : Solver object
        The solver with the mesh
    u_dofs : array
        Solution degrees of freedom
    filename : str
        Output MED file name
    field_name : str
        Base name for fields
    """
    mesh = solver.mesh
    
    # Compute P0 values and gradients at cell centroids
    u_cells = np.zeros(mesh.n_cells)
    grad_mag_cells = np.zeros(mesh.n_cells)
    cell_centroids = np.zeros((mesh.n_cells, 2))
    
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        cell_centroids[cell_id] = cent
        u_cells[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)
    
    grad_cells = _compute_gradients_numerical(solver, u_dofs, cell_centroids, 
                                               np.arange(mesh.n_cells))
    grad_mag_cells = np.sqrt(grad_cells[:, 0]**2 + grad_cells[:, 1]**2)
    
    # Compute P1 values and gradients at vertices using averaging from adjacent cells
    u_vertices = np.zeros(mesh.n_vertices)
    grad_mag_vertices = np.zeros(mesh.n_vertices)
    vertex_count = np.zeros(mesh.n_vertices)

    for cell_id, cell in enumerate(mesh.cells):
        for vertex_id in cell:
            vertex_pos = mesh.vertices[vertex_id]
            u_val = solver.evaluate_solution(u_dofs, vertex_pos, cell_id)
            u_vertices[vertex_id] += u_val
            grad_mag_vertices[vertex_id] += grad_mag_cells[cell_id]
            vertex_count[vertex_id] += 1

    # Average values and gradient magnitudes at vertices shared by multiple cells
    u_vertices /= np.maximum(vertex_count, 1)
    grad_mag_vertices /= np.maximum(vertex_count, 1)

    # Create MEDCoupling mesh
    coords_array = mesh.vertices
    if coords_array.shape[1] == 2:
        # Add z=0 coordinate
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    # Create coordinate array
    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    # Create unstructured mesh
    umesh = mc.MEDCouplingUMesh("solution_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group cells by type for MEDCoupling contiguity requirement
    cells_by_type = {}
    for cell_id, cell in enumerate(mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
        cells_by_type[cell_type].append((cell_id, cell))

    # Build mapping from new cell order to original cell_id
    cell_mapping = []
    umesh.allocateCells(mesh.n_cells)

    # Insert cells grouped by type
    for cell_type in sorted(cells_by_type.keys()):
        for cell_id, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)
            cell_mapping.append(cell_id)

    umesh.finishInsertingCells()

    # Reorder field values according to cell_mapping
    u_cells_reordered = u_cells[cell_mapping]
    grad_mag_cells_reordered = grad_mag_cells[cell_mapping]

    # Create P0 field on cells
    field_p0 = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field_p0.setName(field_name + "_P0")
    field_p0.setMesh(umesh)
    field_p0.setTime(0.0, 0, 0)

    field_array_p0 = mc.DataArrayDouble(u_cells_reordered)
    field_array_p0.setInfoOnComponent(0, field_name + "_P0")
    field_p0.setArray(field_array_p0)
    field_p0.checkConsistencyLight()

    # Create P0 gradient magnitude field on cells
    field_grad_mag_p0 = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field_grad_mag_p0.setName("gradient_mag_" + field_name + "_P0")
    field_grad_mag_p0.setMesh(umesh)
    field_grad_mag_p0.setTime(0.0, 0, 0)

    field_array_grad_mag_p0 = mc.DataArrayDouble(grad_mag_cells_reordered)
    field_array_grad_mag_p0.setInfoOnComponent(0, "|grad " + field_name + "|")
    field_grad_mag_p0.setArray(field_array_grad_mag_p0)
    field_grad_mag_p0.checkConsistencyLight()

    # Create P1 field on nodes
    field_p1 = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field_p1.setName(field_name + "_P1")
    field_p1.setMesh(umesh)
    field_p1.setTime(0.0, 0, 0)

    field_array_p1 = mc.DataArrayDouble(u_vertices)
    field_array_p1.setInfoOnComponent(0, field_name + "_P1")
    field_p1.setArray(field_array_p1)
    field_p1.checkConsistencyLight()

    # Create P1 gradient magnitude field on nodes
    field_grad_mag_p1 = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field_grad_mag_p1.setName("gradient_mag_" + field_name + "_P1")
    field_grad_mag_p1.setMesh(umesh)
    field_grad_mag_p1.setTime(0.0, 0, 0)

    field_array_grad_mag_p1 = mc.DataArrayDouble(grad_mag_vertices)
    field_array_grad_mag_p1.setInfoOnComponent(0, "|grad " + field_name + "|")
    field_grad_mag_p1.setArray(field_array_grad_mag_p1)
    field_grad_mag_p1.checkConsistencyLight()

    # Write MED file with mesh
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    # Write all fields
    med_writer_p0 = mc.MEDFileField1TS()
    med_writer_p0.setFieldNoProfileSBT(field_p0)
    med_writer_p0.write(filename, 0)

    med_writer_grad_mag_p0 = mc.MEDFileField1TS()
    med_writer_grad_mag_p0.setFieldNoProfileSBT(field_grad_mag_p0)
    med_writer_grad_mag_p0.write(filename, 0)

    med_writer_p1 = mc.MEDFileField1TS()
    med_writer_p1.setFieldNoProfileSBT(field_p1)
    med_writer_p1.write(filename, 0)

    med_writer_grad_mag_p1 = mc.MEDFileField1TS()
    med_writer_grad_mag_p1.setFieldNoProfileSBT(field_grad_mag_p1)
    med_writer_grad_mag_p1.write(filename, 0)

    print(f"P0 and P1 fields with gradient magnitudes exported to MED: {filename}")
    print(f"  Fields: {field_name}_P0, {field_name}_P1, gradient_mag_{field_name}_P0, gradient_mag_{field_name}_P1")

def project_and_export_to_triangular_mesh_med(solver, u_dofs, tria_mesh_file,
                                         output_file="solution_tria.med",
                                         field_name="u"):
    """
    Project P1 DG solution (polymesh cell-centroid values) onto a triangular
    MED mesh whose vertices correspond to the polymesh cell centroids, and
    write the result into a MED file (node-based field).
    """
    if mc is None:
        raise ImportError("MEDCoupling (mc) is required to export MED files")

    print(f"Loading triangular mesh from {tria_mesh_file}...")
    tria_mesh = load_med_mesh_mc(tria_mesh_file)

    if tria_mesh.n_vertices != solver.mesh.n_cells:
        print(f"WARNING: Triangular mesh has {tria_mesh.n_vertices} vertices "
              f"but polymesh has {solver.mesh.n_cells} cells!")
        print("Proceeding anyway, but results may be incorrect.")

    # Evaluate DG solution at each polymesh cell centroid -> values at tria nodes
    u_tria_vertices = np.zeros(tria_mesh.n_vertices)
    for cell_id in range(min(solver.mesh.n_cells, tria_mesh.n_vertices)):
        cent = solver.mesh.cell_centroid(cell_id)
        u_tria_vertices[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)

    # Build MEDCoupling mesh from the triangular mesh data
    coords_array = tria_mesh.vertices
    if coords_array.shape[1] == 2:
        coords_3d = np.column_stack([coords_array, np.zeros(len(coords_array))])
    else:
        coords_3d = coords_array

    coords_mc = mc.DataArrayDouble(coords_3d)
    coords_mc.setInfoOnComponents(["X", "Y", "Z"])

    umesh = mc.MEDCouplingUMesh("tria_mesh", 2)
    umesh.setCoords(coords_mc)

    # Group and insert cells (keep same approach used elsewhere)
    cells_by_type = {}
    for cell_id, cell in enumerate(tria_mesh.cells):
        n_nodes = len(cell)
        if n_nodes == 3:
            cell_type = mc.NORM_TRI3
        elif n_nodes == 4:
            cell_type = mc.NORM_QUAD4
        else:
            cell_type = mc.NORM_POLYGON

        cells_by_type.setdefault(cell_type, []).append((cell_id, cell))

    umesh.allocateCells(tria_mesh.n_cells)
    for cell_type in sorted(cells_by_type.keys()):
        for _, cell in cells_by_type[cell_type]:
            umesh.insertNextCell(cell_type, cell)
    umesh.finishInsertingCells()

    # Create node-based field and write MED file
    field = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field.setName(field_name)
    field.setMesh(umesh)
    field.setTime(0.0, 0, 0)

    field_array = mc.DataArrayDouble(u_tria_vertices)
    field_array.setInfoOnComponent(0, field_name)
    field.setArray(field_array)

    field.checkConsistencyLight()

    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("tria_mesh")
    med_mesh.write(output_file, 2)  # overwrite

    med_writer = mc.MEDFileField1TS()
    med_writer.setFieldNoProfileSBT(field)
    med_writer.write(output_file, 0)  # append

    print(f"Solution exported to triangular MED: {output_file}")
    print(f"  - Triangular mesh vertices: {tria_mesh.n_vertices}")
