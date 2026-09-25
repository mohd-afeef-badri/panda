"""Polygonal mesh data structure."""

import numpy as np


class PolygonalMesh:
    """Simple polygonal mesh structure"""
    def __init__(self, vertices, cells, boundary_edges=None):
        self.vertices = np.array(vertices)
        self.cells = cells
        self.dimension = 2
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


class PolyhedralMesh:
    """A three-dimensional mesh made from arbitrary planar-faced polyhedra.

    Parameters
    ----------
    vertices : array-like, shape (n_vertices, 3)
        Vertex coordinates.
    cells : sequence
        Each cell is a sequence of faces and each face is a sequence of vertex
        indices.  Face orientation is not significant; outward normals are
        computed for the cell requesting them.
    boundary_faces : sequence of int, optional
        Explicit boundary-face indices.  By default, faces adjacent to exactly
        one cell are boundary faces.

    Notes
    -----
    Faces shared by two cells are identified by their set of vertex indices.
    The mesh therefore supports tetrahedra, hexahedra, prisms, and convex
    conforming polyhedra without requiring a particular local face ordering.
    """

    dimension = 3

    def __init__(self, vertices, cells, boundary_faces=None):
        self.vertices = np.asarray(vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("PolyhedralMesh vertices must have shape (n, 3)")

        self.n_vertices = len(self.vertices)
        self.n_cells = len(cells)
        self.faces = []
        self.face_to_cells = {}
        self.cell_to_faces = [[] for _ in range(self.n_cells)]
        self.cells = []
        face_indices = {}

        for cell_id, cell_faces in enumerate(cells):
            if len(cell_faces) < 4:
                raise ValueError("A polyhedral cell must contain at least four faces")
            cell_vertices = []
            for face in cell_faces:
                face = tuple(int(vertex) for vertex in face)
                if len(face) < 3:
                    raise ValueError("A polyhedral face must contain at least three vertices")
                if min(face) < 0 or max(face) >= self.n_vertices:
                    raise ValueError("Face contains an invalid vertex index")

                key = tuple(sorted(face))
                if key not in face_indices:
                    face_indices[key] = len(self.faces)
                    self.faces.append(face)
                    self.face_to_cells[key] = []
                face_id = face_indices[key]
                self.face_to_cells[key].append(cell_id)
                self.cell_to_faces[cell_id].append(face_id)
                cell_vertices.extend(face)
            self.cells.append(list(dict.fromkeys(cell_vertices)))

        non_manifold = [
            face for face in self.faces
            if len(self.face_to_cells[tuple(sorted(face))]) > 2
        ]
        if non_manifold:
            raise ValueError("A mesh face cannot be shared by more than two cells")

        if boundary_faces is None:
            self.boundary_faces = [
                face_id for face_id, face in enumerate(self.faces)
                if len(self.face_to_cells[tuple(sorted(face))]) == 1
            ]
        else:
            self.boundary_faces = list(boundary_faces)

    def cell_centroid(self, cell_id):
        """Return the vertex-average cell center used by the local P1 basis."""
        return np.mean(self.vertices[self.cells[cell_id]], axis=0)

    def cell_volume(self, cell_id):
        """Compute cell volume by decomposing its faces into tetrahedra."""
        center = self.cell_centroid(cell_id)
        volume = 0.0
        for face_id in self.cell_to_faces[cell_id]:
            face_vertices = self.vertices[list(self.faces[face_id])]
            anchor = face_vertices[0]
            for i in range(1, len(face_vertices) - 1):
                volume += abs(np.dot(
                    anchor - center,
                    np.cross(face_vertices[i] - center, face_vertices[i + 1] - center),
                )) / 6.0
        return volume

    def cell_diameter(self, cell_id):
        verts = self.vertices[self.cells[cell_id]]
        return max(
            np.linalg.norm(verts[i] - verts[j])
            for i in range(len(verts)) for j in range(i + 1, len(verts))
        )

    def face_centroid(self, face_id):
        return np.mean(self.vertices[list(self.faces[face_id])], axis=0)

    # ``face_midpoint`` is useful to dimension-independent clients such as the
    # boundary-condition manager, even though a polygon has a centroid rather
    # than a unique midpoint.
    face_midpoint = face_centroid

    def face_area(self, face_id):
        verts = self.vertices[list(self.faces[face_id])]
        anchor = verts[0]
        return sum(
            np.linalg.norm(np.cross(verts[i] - anchor, verts[i + 1] - anchor)) / 2.0
            for i in range(1, len(verts) - 1)
        )

    def face_normal(self, face_id, cell_id):
        """Return the unit normal pointing out of ``cell_id``."""
        verts = self.vertices[list(self.faces[face_id])]
        normal = np.zeros(3)
        # Newell's formula is stable for convex planar polygon faces.
        for i, point in enumerate(verts):
            nxt = verts[(i + 1) % len(verts)]
            normal += np.array([
                (point[1] - nxt[1]) * (point[2] + nxt[2]),
                (point[2] - nxt[2]) * (point[0] + nxt[0]),
                (point[0] - nxt[0]) * (point[1] + nxt[1]),
            ])
        norm = np.linalg.norm(normal)
        if norm <= np.finfo(float).eps:
            raise ValueError(f"Degenerate face {face_id} has zero area")
        normal /= norm
        if np.dot(normal, self.face_centroid(face_id) - self.cell_centroid(cell_id)) < 0:
            normal = -normal
        return normal

    def oriented_cell_faces(self, cell_id):
        """Return cell faces wound counter-clockwise when viewed from outside."""
        cell_center = self.cell_centroid(cell_id)
        oriented_faces = []
        for face_id in self.cell_to_faces[cell_id]:
            face = list(self.faces[face_id])
            verts = self.vertices[face]
            normal = np.zeros(3)
            for i, point in enumerate(verts):
                nxt = verts[(i + 1) % len(verts)]
                normal += np.array([
                    (point[1] - nxt[1]) * (point[2] + nxt[2]),
                    (point[2] - nxt[2]) * (point[0] + nxt[0]),
                    (point[0] - nxt[0]) * (point[1] + nxt[1]),
                ])
            if np.dot(normal, np.mean(verts, axis=0) - cell_center) < 0.0:
                face.reverse()
            oriented_faces.append(face)
        return oriented_faces

    def face_quadrature(self, face_id):
        """Degree-two quadrature points and weights for a polygonal face."""
        verts = self.vertices[list(self.faces[face_id])]
        center = self.face_centroid(face_id)
        points = []
        weights = []
        barycentric_points = (
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        )
        for i in range(len(verts)):
            a, b, c = center, verts[i], verts[(i + 1) % len(verts)]
            triangle_area = np.linalg.norm(np.cross(b - a, c - a)) / 2.0
            if triangle_area <= np.finfo(float).eps:
                continue
            for la, lb, lc in barycentric_points:
                points.append(la * a + lb * b + lc * c)
                weights.append(triangle_area / 3.0)
        return points, weights

    def cell_quadrature(self, cell_id):
        """Degree-three quadrature points and weights for a convex cell.

        The polyhedron is split into tetrahedra from the cell center to a
        triangulation of each face.  The five-point tetrahedron rule then
        integrates cubic polynomials exactly.
        """
        center = self.cell_centroid(cell_id)
        points = []
        weights = []
        for face_id in self.cell_to_faces[cell_id]:
            verts = self.vertices[list(self.faces[face_id])]
            anchor = verts[0]
            for i in range(1, len(verts) - 1):
                tetra = np.array([center, anchor, verts[i], verts[i + 1]])
                volume = abs(np.dot(
                    tetra[1] - tetra[0],
                    np.cross(tetra[2] - tetra[0], tetra[3] - tetra[0]),
                )) / 6.0
                if volume <= np.finfo(float).eps:
                    continue
                tetra_center = np.mean(tetra, axis=0)
                points.append(tetra_center)
                weights.append(-4.0 * volume / 5.0)
                for vertex_id in range(4):
                    barycentric = np.full(4, 1.0 / 6.0)
                    barycentric[vertex_id] = 1.0 / 2.0
                    points.append(barycentric @ tetra)
                    weights.append(9.0 * volume / 20.0)
        return points, weights


def create_box_mesh(length=1.0, width=1.0, height=1.0, nx=1, ny=1, nz=1):
    """Create a Cartesian hexahedral mesh of a rectangular box."""
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, and nz must all be >= 1")
    if min(length, width, height) <= 0:
        raise ValueError("Box dimensions must be positive")

    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, width, ny + 1)
    zs = np.linspace(0.0, height, nz + 1)
    vertices = np.array([
        [xs[i], ys[j], zs[k]]
        for k in range(nz + 1)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ])

    def vertex(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000 = vertex(i, j, k)
                v100 = vertex(i + 1, j, k)
                v110 = vertex(i + 1, j + 1, k)
                v010 = vertex(i, j + 1, k)
                v001 = vertex(i, j, k + 1)
                v101 = vertex(i + 1, j, k + 1)
                v111 = vertex(i + 1, j + 1, k + 1)
                v011 = vertex(i, j + 1, k + 1)
                cells.append([
                    [v000, v010, v110, v100],  # z = lower
                    [v001, v101, v111, v011],  # z = upper
                    [v000, v100, v101, v001],  # y = lower
                    [v010, v011, v111, v110],  # y = upper
                    [v000, v001, v011, v010],  # x = lower
                    [v100, v110, v111, v101],  # x = upper
                ])
    return PolyhedralMesh(vertices, cells)


def create_cube_mesh(n=1):
    """Create an ``n x n x n`` hexahedral mesh of the unit cube."""
    return create_box_mesh(nx=n, ny=n, nz=n)


def create_tetrahedral_cube_mesh(n=1):
    """Create a conforming six-tetrahedra-per-voxel unit-cube mesh."""
    if n < 1:
        raise ValueError("n must be >= 1")
    coordinates = np.linspace(0.0, 1.0, n + 1)
    vertices = np.array([
        [coordinates[i], coordinates[j], coordinates[k]]
        for k in range(n + 1)
        for j in range(n + 1)
        for i in range(n + 1)
    ])

    def vertex(i, j, k):
        return k * (n + 1)**2 + j * (n + 1) + i

    def tetrahedron(a, b, c, d):
        return [[a, b, c], [a, d, b], [b, d, c], [c, d, a]]

    cells = []
    # Freudenthal/Kuhn triangulation: all tetrahedra share the lower-to-upper
    # body diagonal and the induced face diagonals agree across voxels.
    import itertools
    for k in range(n):
        for j in range(n):
            for i in range(n):
                origin = np.array([i, j, k])
                for permutation in itertools.permutations(range(3)):
                    lattice_points = [origin.copy()]
                    point = origin.copy()
                    for axis in permutation:
                        point = point.copy()
                        point[axis] += 1
                        lattice_points.append(point)
                    ids = [vertex(*point) for point in lattice_points]
                    cells.append(tetrahedron(*ids))
    return PolyhedralMesh(vertices, cells)


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
