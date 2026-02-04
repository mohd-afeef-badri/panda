"""Polygonal mesh data structure."""

import numpy as np


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


def create_square_mesh(n=4):
    """Create a simple square mesh divided into quadrilaterals
    
    Parameters:
    -----------
    n : int, default=4
        Number of divisions in each direction
    
    Returns:
    --------
    PolygonalMesh
        A unit square [0,1]x[0,1] divided into n×n quadrilateral cells
    """
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


def create_rectangle_mesh(length=1.0, height=1.0, nx=4, ny=4):
    """Create a rectangle mesh divided into quadrilaterals.

    Parameters:
    -----------
    length : float
        Extent in the x-direction
    height : float
        Extent in the y-direction
    nx : int
        Number of divisions in x
    ny : int
        Number of divisions in y

    Returns:
    --------
    PolygonalMesh
        A rectangle [0,length]x[0,height] divided into nx×ny quad cells
    """
    x = np.linspace(0, length, nx+1)
    y = np.linspace(0, height, ny+1)

    vertices = []
    for j in range(ny+1):
        for i in range(nx+1):
            vertices.append([x[i], y[j]])
    vertices = np.array(vertices)

    cells = []
    for j in range(ny):
        for i in range(nx):
            idx = j * (nx+1) + i
            cell = [idx, idx+1, idx+nx+2, idx+nx+1]
            cells.append(cell)

    return PolygonalMesh(vertices, cells)


def create_circle_mesh(radius=1.0, n_radial=3, n_circ=32):
    """Create a disk (circular) mesh using concentric rings.

    The mesh is formed by a center vertex plus `n_radial` rings, each with
    `n_circ` points. Cells are triangles adjacent to the center for the
    innermost ring and quads between consecutive rings elsewhere.

    Parameters:
    -----------
    radius : float
        Radius of the disk
    n_radial : int
        Number of radial subdivisions (rings)
    n_circ : int
        Number of angular subdivisions (per ring)

    Returns:
    --------
    PolygonalMesh
        A polygonal mesh approximating the disk
    """
    if n_radial < 1:
        raise ValueError("n_radial must be >= 1")
    if n_circ < 3:
        raise ValueError("n_circ must be >= 3")

    vertices = []
    # center
    vertices.append([0.0, 0.0])

    # rings
    radii = np.linspace(0.0, radius, n_radial+1)[1:]
    for r in radii:
        for j in range(n_circ):
            theta = 2.0 * np.pi * j / n_circ
            vertices.append([r * np.cos(theta), r * np.sin(theta)])

    vertices = np.array(vertices)

    def idx(ring, ang):
        # ring: 1..n_radial, ang: 0..n_circ-1
        return 1 + (ring-1) * n_circ + (ang % n_circ)

    cells = []
    # innermost cells (triangles connecting center and first ring)
    for j in range(n_circ):
        cells.append([0, idx(1, j), idx(1, j+1)])

    # cells between rings (quads)
    for ring in range(2, n_radial+1):
        for j in range(n_circ):
            a = idx(ring-1, j)
            b = idx(ring-1, j+1)
            c = idx(ring, j+1)
            d = idx(ring, j)
            cells.append([a, b, c, d])

    return PolygonalMesh(vertices, cells)
