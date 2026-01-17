try:
    import medcoupling as mc
except ImportError:
    print("MEDCoupling not found. Install it from SALOME or compile from source.")
    print("You can access it via: salome shell -- python")
    raise

import numpy as np
import poly_test

class PolygonalMesh:
    """Simple polygonal mesh structure"""
    def __init__(self, vertices, cells, boundary_edges=None):
        self.vertices = np.array(vertices)
        self.cells = cells
        self.n_cells = len(cells)
        self.n_vertices = len(vertices)
        
        # Build edge connectivity
        self.edges = []
        self.edge_to_cells = {}
        self.cell_to_edges = [[] for _ in range(self.n_cells)]
        
        for cell_id, cell in enumerate(cells):
            n_edges = len(cell)
            for i in range(n_edges):
                v1, v2 = cell[i], cell[(i+1) % n_edges]
                edge = tuple(sorted([v1, v2]))
                
                if edge not in self.edge_to_cells:
                    edge_id = len(self.edges)
                    self.edges.append(edge)
                    self.edge_to_cells[edge] = []
                else:
                    edge_id = self.edges.index(edge)
                
                self.edge_to_cells[edge].append(cell_id)
                self.cell_to_edges[cell_id].append(edge_id)
        
        if boundary_edges is None:
            self.boundary_edges = [i for i, edge in enumerate(self.edges) 
                                   if len(self.edge_to_cells[edge]) == 1]
        else:
            self.boundary_edges = boundary_edges
    
    def cell_centroid(self, cell_id):
        verts = self.vertices[self.cells[cell_id]]
        return np.mean(verts, axis=0)
    
    def cell_area(self, cell_id):
        verts = self.vertices[self.cells[cell_id]]
        x, y = verts[:, 0], verts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def cell_diameter(self, cell_id):
        verts = self.vertices[self.cells[cell_id]]
        max_dist = 0
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                dist = np.linalg.norm(verts[i] - verts[j])
                max_dist = max(max_dist, dist)
        return max_dist
    
    def edge_length(self, edge_id):
        v1, v2 = self.edges[edge_id]
        return np.linalg.norm(self.vertices[v2] - self.vertices[v1])
    
    def edge_midpoint(self, edge_id):
        v1, v2 = self.edges[edge_id]
        return 0.5 * (self.vertices[v1] + self.vertices[v2])
    
    def edge_normal(self, edge_id, cell_id):
        v1, v2 = self.edges[edge_id]
        edge_vec = self.vertices[v2] - self.vertices[v1]
        normal = np.array([edge_vec[1], -edge_vec[0]])
        normal = normal / np.linalg.norm(normal)
        
        edge_mid = self.edge_midpoint(edge_id)
        cell_cent = self.cell_centroid(cell_id)
        if np.dot(normal, edge_mid - cell_cent) < 0:
            normal = -normal
        
        return normal

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

def create_square_mesh(n=4):
    """Create a simple square mesh divided into quadrilaterals"""
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    
    vertices = []
    for j in range(n+1):
        for i in range(n+1):
            vertices.append([x[i], y[j]])
    vertices = np.array(vertices)
    
    cells = []
    for j in range(n):
        for i in range(n):
            idx = j * (n+1) + i
            cell = [idx, idx+1, idx+n+2, idx+n+1]
            cells.append(cell)
    
    return PolygonalMesh(vertices, cells)
