import numpy as np

try:
    import medcoupling as mc
except ImportError:
    mc = None  # allow imports on systems without MEDCoupling for testing

from med_reader import load_med_mesh_mc

def export_to_med(solver, u_dofs, filename="solution.med", field_name="u",
                  u_exact_func=None, method="P0"):
    if mc is None:
        raise ImportError("MEDCoupling (mc) is required to export MED files")
    if method == "P0":
        _export_med_p0(solver, u_dofs, filename, field_name, u_exact_func)
    elif method == "P1_vertex":
        _export_med_p1_vertex(solver, u_dofs, filename, field_name, u_exact_func)
    else:
        raise ValueError(f"Unknown export method: {method}")


def _export_med_p0(solver, u_dofs, filename, field_name, u_exact_func):
    mesh = solver.mesh
    # Evaluate at cell centroids
    u_cells = np.zeros(mesh.n_cells)
    for cell_id in range(mesh.n_cells):
        cent = mesh.cell_centroid(cell_id)
        u_cells[cell_id] = solver.evaluate_solution(u_dofs, cent, cell_id)

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

    # Create field on cells
    field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field.setName(field_name)
    field.setMesh(umesh)
    field.setTime(0.0, 0, 0)  # time, iteration, order

    # Set field values
    field_array = mc.DataArrayDouble(u_cells_reordered)
    field_array.setInfoOnComponent(0, field_name)
    field.setArray(field_array)

    # Check consistency
    field.checkConsistencyLight()

    # Write MED file
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    med_writer = mc.MEDFileField1TS()
    med_writer.setFieldNoProfileSBT(field)
    med_writer.write(filename, 0)  # 0 = append mode

    # Add error field if available
    if u_exact_func is not None:
        errors = np.zeros(mesh.n_cells)
        for cell_id in range(mesh.n_cells):
            cent = mesh.cell_centroid(cell_id)
            errors[cell_id] = abs(u_cells[cell_id] - u_exact_func(cent[0], cent[1]))

        # Reorder errors
        errors_reordered = errors[cell_mapping]

        error_field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
        error_field.setName("error")
        error_field.setMesh(umesh)
        error_field.setTime(0.0, 0, 0)
        error_array = mc.DataArrayDouble(errors_reordered)
        error_array.setInfoOnComponent(0, "error")
        error_field.setArray(error_array)

        error_field.checkConsistencyLight()

        error_writer = mc.MEDFileField1TS()
        error_writer.setFieldNoProfileSBT(error_field)
        error_writer.write(filename, 0)  # append

    print(f"P0 projection exported to MED: {filename}")


def _export_med_p1_vertex(solver, u_dofs, filename, field_name, u_exact_func):
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

    # Create field on nodes (no reordering needed for node fields)
    field = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
    field.setName(field_name)
    field.setMesh(umesh)
    field.setTime(0.0, 0, 0)

    # Set field values
    field_array = mc.DataArrayDouble(u_vertices)
    field_array.setInfoOnComponent(0, field_name)
    field.setArray(field_array)

    # Check consistency
    field.checkConsistencyLight()

    # Write MED file
    med_mesh = mc.MEDFileUMesh()
    med_mesh.setMeshAtLevel(0, umesh)
    med_mesh.setName("solution_mesh")
    med_mesh.write(filename, 2)  # 2 = write mode (overwrite)

    med_writer = mc.MEDFileField1TS()
    med_writer.setFieldNoProfileSBT(field)
    med_writer.write(filename, 0)  # 0 = append mode

    # Add error field if available
    if u_exact_func is not None:
        errors = np.zeros(mesh.n_vertices)
        for vertex_id in range(mesh.n_vertices):
            vertex_pos = mesh.vertices[vertex_id]
            errors[vertex_id] = abs(u_vertices[vertex_id] - u_exact_func(vertex_pos[0], vertex_pos[1]))

        error_field = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
        error_field.setName("error")
        error_field.setMesh(umesh)
        error_field.setTime(0.0, 0, 0)
        error_array = mc.DataArrayDouble(errors)
        error_array.setInfoOnComponent(0, "error")
        error_field.setArray(error_array)

        error_field.checkConsistencyLight()

        error_writer = mc.MEDFileField1TS()
        error_writer.setFieldNoProfileSBT(error_field)
        error_writer.write(filename, 0)  # append

    print(f"P1 vertex interpolation exported to MED: {filename}")

def project_and_export_to_triangular_mesh_med(solver, u_dofs, tria_mesh_file,
                                         output_file="solution_tria.med",
                                         field_name="u",
                                         u_exact_func=None):
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

    # Optional error field (node-based)
    if u_exact_func is not None:
        errors = np.zeros(tria_mesh.n_vertices)
        for vid in range(tria_mesh.n_vertices):
            vpos = tria_mesh.vertices[vid]
            errors[vid] = abs(u_tria_vertices[vid] - u_exact_func(vpos[0], vpos[1]))

        error_field = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
        error_field.setName("error")
        error_field.setMesh(umesh)
        error_field.setTime(0.0, 0, 0)

        error_array = mc.DataArrayDouble(errors)
        error_array.setInfoOnComponent(0, "error")
        error_field.setArray(error_array)

        error_field.checkConsistencyLight()

        error_writer = mc.MEDFileField1TS()
        error_writer.setFieldNoProfileSBT(error_field)
        error_writer.write(output_file, 0)  # append

    print(f"Solution exported to triangular MED: {output_file}")
    print(f"  - Triangular mesh vertices: {tria_mesh.n_vertices}")
    print(f"  - Triangular mesh cells: {tria_mesh.n_cells}")

    if u_exact_func is not None:
        print(f"  - Max error: {np.max(errors):.6e}")
        print(f"  - Mean error: {np.mean(errors):.6e}")
        print(f"  - L2 error: {np.sqrt(np.mean(errors**2)):.6e}")
