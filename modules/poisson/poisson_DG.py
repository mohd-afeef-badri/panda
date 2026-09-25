import sys
from pathlib import Path

# Add grandparent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from panda.lib import boundary_conditions
from panda.lib.linear_solvers import solve_linear_system

class P1DGPoissonSolver:
    """
    P1 DG solver for ``-Delta u = f`` using the SIPG method in 2D or 3D.

    The local basis is centered at each cell centroid.  It contains one
    constant and one linear function per coordinate, so there are three DOFs
    per polygon in 2D and four DOFs per polyhedron in 3D.
    """
    def __init__(self, mesh, bc_manager, penalty_param=10.0,
                 linear_solver="direct", solver_options=None):
        self.mesh = mesh
        self.bc_manager = bc_manager
        self.penalty = penalty_param
        self.dimension = getattr(mesh, "dimension", np.asarray(mesh.vertices).shape[1])
        if self.dimension not in (2, 3):
            raise ValueError("P1DGPoissonSolver supports only 2D and 3D meshes")
        self.n_dofs_per_cell = self.dimension + 1
        self.n_dofs = mesh.n_cells * self.n_dofs_per_cell
        self.linear_solver = linear_solver
        self.solver_options = dict(solver_options or {})
        self.last_solve_info = None

    def local_to_global(self, cell_id, local_dof):
        """Map local DOF to global DOF index"""
        return cell_id * self.n_dofs_per_cell + local_dof
    
    def evaluate_basis(self, cell_id, point, derivatives=False):
        """
        Evaluate the centroid-based P1 basis at a point.

        With ``derivatives=True``, the return value is ``(values, *grads)``:
        three arrays in 2D and four arrays in 3D.  This retains the original
        2D API while naturally adding the z derivative.
        """
        cent = np.asarray(self.mesh.cell_centroid(cell_id), dtype=float)
        point = np.asarray(point, dtype=float)
        vals = np.concatenate(([1.0], point[:self.dimension] - cent[:self.dimension]))

        if not derivatives:
            return vals

        gradients = np.zeros((self.n_dofs_per_cell, self.dimension))
        gradients[1:, :] = np.eye(self.dimension)
        return (vals, *(gradients[:, direction] for direction in range(self.dimension)))

    def _basis_and_gradients(self, cell_id, point):
        evaluated = self.evaluate_basis(cell_id, point, derivatives=True)
        return evaluated[0], np.column_stack(evaluated[1:])

    def _cell_measure(self, cell_id):
        if self.dimension == 2:
            return self.mesh.cell_area(cell_id)
        return self.mesh.cell_volume(cell_id)

    def _face_data(self, face_id):
        """Return adjacent cells, measure, and representative point."""
        if self.dimension == 2:
            edge = self.mesh.edges[face_id]
            return (
                self.mesh.edge_to_cells[edge],
                self.mesh.edge_length(face_id),
                self.mesh.edge_midpoint(face_id),
            )
        face = self.mesh.faces[face_id]
        return (
            self.mesh.face_to_cells[tuple(sorted(face))],
            self.mesh.face_area(face_id),
            self.mesh.face_centroid(face_id),
        )

    def _normal(self, face_id, cell_id):
        if self.dimension == 2:
            return self.mesh.edge_normal(face_id, cell_id)
        return self.mesh.face_normal(face_id, cell_id)

    def _face_quadrature(self, face_id, measure, midpoint):
        # Preserve the original edge-midpoint assembly exactly in 2D.  In 3D,
        # degree-two polygon quadrature exactly integrates all P1 face terms.
        if self.dimension == 2:
            return [(np.asarray(midpoint), measure)]
        if hasattr(self.mesh, "face_quadrature"):
            points, weights = self.mesh.face_quadrature(face_id)
            return list(zip(points, weights))
        return [(np.asarray(midpoint), measure)]

    def _cell_quadrature(self, cell_id, measure, centroid):
        if self.dimension == 3 and hasattr(self.mesh, "cell_quadrature"):
            points, weights = self.mesh.cell_quadrature(cell_id)
            return list(zip(points, weights))
        return [(np.asarray(centroid), measure)]
    
    def assemble_system(self, f_func):
        """
        Assemble SIPG system according to deal.II step-74 formulation:

        ∑_K (∇v_h, ∇u_h)_K
        - ∑_{F∈F_i} { <[[v_h]], {∇u_h}·n>_F + <{∇v_h}·n, [[u_h]]>_F - <[[v_h]], σ[[u_h]]>_F }
        - ∑_{F∈F_b} { <v_h, ∇u_h·n>_F + <∇v_h·n, u_h>_F - <v_h, σu_h>_F }
        = (v_h, f)_Ω - ∑_{F∈F_b} { <∇v_h·n, g_D>_F - <v_h, σg_D>_F }
        
        where σ = γ/h_f and [[·]] denotes jump, {·} denotes average
        Parameters:
        -----------
        f_func : callable
            Source term ``f(x, y)`` in 2D or ``f(x, y, z)`` in 3D.
        """
        A = lil_matrix((self.n_dofs, self.n_dofs))
        b = np.zeros(self.n_dofs)
        
        # Assemble volume terms: (∇v, ∇u)_K
        for cell_id in range(self.mesh.n_cells):
            measure = self._cell_measure(cell_id)
            cent = self.mesh.cell_centroid(cell_id)
            
            # Load vector
            for point, weight in self._cell_quadrature(cell_id, measure, cent):
                phi = self.evaluate_basis(cell_id, point)
                f_val = f_func(*point)
                for i in range(self.n_dofs_per_cell):
                    i_global = self.local_to_global(cell_id, i)
                    b[i_global] += f_val * phi[i] * weight
            
            # Stiffness matrix: ∫ ∇φ_i · ∇φ_j dx
            _, gradients = self._basis_and_gradients(cell_id, cent)
            
            for i in range(self.n_dofs_per_cell):
                for j in range(self.n_dofs_per_cell):
                    i_global = self.local_to_global(cell_id, i)
                    j_global = self.local_to_global(cell_id, j)
                    stiff = np.dot(gradients[i], gradients[j]) * measure
                    A[i_global, j_global] += stiff
        
        # Assemble face terms (SIPG)
        n_faces = len(self.mesh.edges) if self.dimension == 2 else len(self.mesh.faces)
        for edge_id in range(n_faces):
            cells, h_e, edge_mid = self._face_data(edge_id)
            
            if len(cells) == 2:  # Interior edge
                self._assemble_interior_face(A, edge_id, cells, h_e, edge_mid)
            else:  # Boundary edge
                bc = self.bc_manager.get_bc(edge_id)
                if bc.bc_type == 'dirichlet':
                    self._assemble_dirichlet_face(A, b, edge_id, cells[0], h_e, edge_mid, bc)
                elif bc.bc_type == 'neumann':
                    self._assemble_neumann_face(A, b, edge_id, cells[0], h_e, edge_mid, bc)
        
        return csr_matrix(A), b
    
    def _assemble_interior_face(self, A, edge_id, cells, h_e, edge_mid):
        """Assemble interior face terms (SIPG)"""
        cell_i, cell_j = cells
        n = self._normal(edge_id, cell_i)

        # Penalty parameter σ = γ/h
        h = min(self.mesh.cell_diameter(cell_i), self.mesh.cell_diameter(cell_j))
        sigma = self.penalty / h

        for point, weight in self._face_quadrature(edge_id, h_e, edge_mid):
            phi_i, gradients_i = self._basis_and_gradients(cell_i, point)
            phi_j, gradients_j = self._basis_and_gradients(cell_j, point)
            grad_n_i = gradients_i @ n
            grad_n_j = gradients_j @ n

            for i in range(self.n_dofs_per_cell):
                for j in range(self.n_dofs_per_cell):
                    i_i = self.local_to_global(cell_i, i)
                    j_i = self.local_to_global(cell_i, j)
                    i_j = self.local_to_global(cell_j, i)
                    j_j = self.local_to_global(cell_j, j)

                    A[i_i, j_i] += weight * (
                        -0.5 * phi_i[i] * grad_n_i[j]
                        -0.5 * grad_n_i[i] * phi_i[j]
                        +sigma * phi_i[i] * phi_i[j]
                    )
                    A[i_i, j_j] += weight * (
                        -0.5 * phi_i[i] * grad_n_j[j]
                        +0.5 * grad_n_i[i] * phi_j[j]
                        -sigma * phi_i[i] * phi_j[j]
                    )
                    A[i_j, j_i] += weight * (
                        +0.5 * phi_j[i] * grad_n_i[j]
                        -0.5 * grad_n_j[i] * phi_i[j]
                        -sigma * phi_j[i] * phi_i[j]
                    )
                    A[i_j, j_j] += weight * (
                        +0.5 * phi_j[i] * grad_n_j[j]
                        +0.5 * grad_n_j[i] * phi_j[j]
                        +sigma * phi_j[i] * phi_j[j]
                    )
    
    def _assemble_dirichlet_face(self, A, b, edge_id, cell_i, h_e, edge_mid, bc):
        """Assemble Dirichlet boundary face terms"""
        n = self._normal(edge_id, cell_i)
        h = self.mesh.cell_diameter(cell_i)
        sigma = self.penalty / h

        for point, weight in self._face_quadrature(edge_id, h_e, edge_mid):
            phi_i, gradients_i = self._basis_and_gradients(cell_i, point)
            grad_n_i = gradients_i @ n
            g_val = bc.evaluate(*point)

            for i in range(self.n_dofs_per_cell):
                i_i = self.local_to_global(cell_i, i)
                for j in range(self.n_dofs_per_cell):
                    j_i = self.local_to_global(cell_i, j)
                    A[i_i, j_i] += weight * (
                        -phi_i[i] * grad_n_i[j]
                        -grad_n_i[i] * phi_i[j]
                        +sigma * phi_i[i] * phi_i[j]
                    )
                b[i_i] += weight * (-grad_n_i[i] * g_val + sigma * phi_i[i] * g_val)
    
    def _assemble_neumann_face(self, A, b, edge_id, cell_i, h_e, edge_mid, bc):
        """Assemble Neumann boundary face terms"""
        for point, weight in self._face_quadrature(edge_id, h_e, edge_mid):
            phi_i = self.evaluate_basis(cell_i, point, derivatives=False)
            g_val = bc.evaluate(*point)
            for i in range(self.n_dofs_per_cell):
                i_i = self.local_to_global(cell_i, i)
                b[i_i] += phi_i[i] * g_val * weight
    
    def solve(self, f_func):
        """Solve the Poisson problem with BC from bc_manager"""
        A, b = self.assemble_system(f_func)
        self.last_solve_info = None
        u_dofs, self.last_solve_info = solve_linear_system(
            A, b, method=self.linear_solver, **self.solver_options
        )
        return u_dofs
    
    def evaluate_solution(self, u_dofs, point, cell_id):
        """Evaluate solution at a point in a given cell"""
        phi = self.evaluate_basis(cell_id, point)
        u_val = 0.0
        for i in range(self.n_dofs_per_cell):
            i_global = self.local_to_global(cell_id, i)
            u_val += u_dofs[i_global] * phi[i]
        return u_val
