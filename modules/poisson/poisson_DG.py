import sys
from pathlib import Path

# Add grandparent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from panda.lib import boundary_conditions

class P1DGPoissonSolver:
    """
    P1 DG solver for -Δu = f with Dirichlet BC using SIPG method
    Uses piecewise linear approximation on each cell
    For each cells: 3 DOFs per cell (constant + x + y gradient)
    """
    def __init__(self, mesh, bc_manager, penalty_param=10.0):
        self.mesh = mesh
        self.bc_manager = bc_manager
        self.penalty = penalty_param
        self.n_dofs_per_cell = 3  # P1 in 2D: [1, x, y]
        self.n_dofs = mesh.n_cells * self.n_dofs_per_cell

    def local_to_global(self, cell_id, local_dof):
        """Map local DOF to global DOF index"""
        return cell_id * self.n_dofs_per_cell + local_dof
    
    def evaluate_basis(self, cell_id, point, derivatives=False):
        """
        Evaluate P1 basis functions at a point
        Basis: phi_0 = 1, phi_1 = x - x_c, phi_2 = y - y_c
        where (x_c, y_c) is cell centroid
        """
        cent = self.mesh.cell_centroid(cell_id)
        x_rel = point[0] - cent[0]
        y_rel = point[1] - cent[1]
        
        if not derivatives:
            return np.array([1.0, x_rel, y_rel])
        else:
            # Return [values, grad_x, grad_y]
            vals = np.array([1.0, x_rel, y_rel])
            grad_x = np.array([0.0, 1.0, 0.0])
            grad_y = np.array([0.0, 0.0, 1.0])
            return vals, grad_x, grad_y
    
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
            Source term f(x, y)
        """
        A = lil_matrix((self.n_dofs, self.n_dofs))
        b = np.zeros(self.n_dofs)
        
        # Assemble volume terms: (∇v, ∇u)_K
        for cell_id in range(self.mesh.n_cells):
            area = self.mesh.cell_area(cell_id)
            cent = self.mesh.cell_centroid(cell_id)
            
            # Load vector
            phi = self.evaluate_basis(cell_id, cent)
            f_val = f_func(cent[0], cent[1])
            
            for i in range(self.n_dofs_per_cell):
                i_global = self.local_to_global(cell_id, i)
                b[i_global] += f_val * phi[i] * area
            
            # Stiffness matrix: ∫ ∇φ_i · ∇φ_j dx
            _, grad_x, grad_y = self.evaluate_basis(cell_id, cent, derivatives=True)
            
            for i in range(self.n_dofs_per_cell):
                for j in range(self.n_dofs_per_cell):
                    i_global = self.local_to_global(cell_id, i)
                    j_global = self.local_to_global(cell_id, j)
                    stiff = (grad_x[i]*grad_x[j] + grad_y[i]*grad_y[j]) * area
                    A[i_global, j_global] += stiff
        
        # Assemble face terms (SIPG)
        for edge_id in range(len(self.mesh.edges)):
            cells = self.mesh.edge_to_cells[self.mesh.edges[edge_id]]
            h_e = self.mesh.edge_length(edge_id)
            edge_mid = self.mesh.edge_midpoint(edge_id)
            
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
        n = self.mesh.edge_normal(edge_id, cell_i)
        
        # Evaluate basis functions
        phi_i, grad_x_i, grad_y_i = self.evaluate_basis(cell_i, edge_mid, derivatives=True)
        phi_j, grad_x_j, grad_y_j = self.evaluate_basis(cell_j, edge_mid, derivatives=True)
        
        # Normal gradients
        grad_n_i = grad_x_i * n[0] + grad_y_i * n[1]
        grad_n_j = grad_x_j * n[0] + grad_y_j * n[1]
        
        # Penalty parameter σ = γ/h
        h = min(self.mesh.cell_diameter(cell_i), self.mesh.cell_diameter(cell_j))
        sigma = self.penalty / h
        
        # SIPG bilinear form on interior faces
        # Note: [[u]] = u_i - u_j (jump), {∇u}·n = 0.5*(∇u_i + ∇u_j)·n (average)
        for i in range(self.n_dofs_per_cell):
            for j in range(self.n_dofs_per_cell):
                i_i = self.local_to_global(cell_i, i)
                j_i = self.local_to_global(cell_i, j)
                i_j = self.local_to_global(cell_j, i)
                j_j = self.local_to_global(cell_j, j)
                
                # Jump [[v]] = v_i - v_j, [[u]] = u_i - u_j
                # Average {∇u·n} = 0.5*(∇u_i·n + ∇u_j·n)
                
                # Term 1: - <[[v]], {∇u·n}> = - <v_i - v_j, 0.5(∇u_i·n + ∇u_j·n)>
                term1_ii = -0.5 * phi_i[i] * grad_n_i[j] * h_e
                term1_ij = -0.5 * phi_i[i] * grad_n_j[j] * h_e
                term1_ji = 0.5 * phi_j[i] * grad_n_i[j] * h_e
                term1_jj = 0.5 * phi_j[i] * grad_n_j[j] * h_e
                
                # Term 2: - <{∇v·n}, [[u]]> = - <0.5(∇v_i·n + ∇v_j·n), u_i - u_j>
                term2_ii = -0.5 * grad_n_i[i] * phi_i[j] * h_e
                term2_ij = 0.5 * grad_n_i[i] * phi_j[j] * h_e
                term2_ji = -0.5 * grad_n_j[i] * phi_i[j] * h_e
                term2_jj = 0.5 * grad_n_j[i] * phi_j[j] * h_e
                
                # Term 3: + <σ[[v]], [[u]]> = <σ(v_i - v_j), u_i - u_j>
                term3_ii = sigma * phi_i[i] * phi_i[j] * h_e
                term3_ij = -sigma * phi_i[i] * phi_j[j] * h_e
                term3_ji = -sigma * phi_j[i] * phi_i[j] * h_e
                term3_jj = sigma * phi_j[i] * phi_j[j] * h_e
                
                A[i_i, j_i] += term1_ii + term2_ii + term3_ii
                A[i_i, j_j] += term1_ij + term2_ij + term3_ij
                A[i_j, j_i] += term1_ji + term2_ji + term3_ji
                A[i_j, j_j] += term1_jj + term2_jj + term3_jj
    
    def _assemble_dirichlet_face(self, A, b, edge_id, cell_i, h_e, edge_mid, bc):
        """Assemble Dirichlet boundary face terms"""
        n = self.mesh.edge_normal(edge_id, cell_i)
        phi_i, grad_x_i, grad_y_i = self.evaluate_basis(cell_i, edge_mid, derivatives=True)
        grad_n_i = grad_x_i * n[0] + grad_y_i * n[1]
        
        h = self.mesh.cell_diameter(cell_i)
        sigma = self.penalty / h
        
        g_val = bc.evaluate(edge_mid[0], edge_mid[1])
        
        # Boundary terms: [[u]] = u, {∇u·n} = ∇u·n
        for i in range(self.n_dofs_per_cell):
            for j in range(self.n_dofs_per_cell):
                i_i = self.local_to_global(cell_i, i)
                j_i = self.local_to_global(cell_i, j)
                
                # - <v, ∇u·n>
                A[i_i, j_i] -= phi_i[i] * grad_n_i[j] * h_e
                # - <∇v·n, u>
                A[i_i, j_i] -= grad_n_i[i] * phi_i[j] * h_e
                # + <σv, u>
                A[i_i, j_i] += sigma * phi_i[i] * phi_i[j] * h_e
            
            # RHS boundary terms
            i_i = self.local_to_global(cell_i, i)
            # - <∇v·n, g>
            b[i_i] -= grad_n_i[i] * g_val * h_e
            # + <σv, g>
            b[i_i] += sigma * phi_i[i] * g_val * h_e
    
    def _assemble_neumann_face(self, A, b, edge_id, cell_i, h_e, edge_mid, bc):
        """Assemble Neumann boundary face terms"""
        phi_i = self.evaluate_basis(cell_i, edge_mid, derivatives=False)
        g_val = bc.evaluate(edge_mid[0], edge_mid[1])
        
        # Neumann BC: ∇u·n = g
        # Weak form: -∫ v * g ds
        for i in range(self.n_dofs_per_cell):
            i_i = self.local_to_global(cell_i, i)
            b[i_i] += phi_i[i] * g_val * h_e
    
    def solve(self, f_func):
        """Solve the Poisson problem with BC from bc_manager"""
        A, b = self.assemble_system(f_func)
        u_dofs = spsolve(A, b)
        return u_dofs
    
    def evaluate_solution(self, u_dofs, point, cell_id):
        """Evaluate solution at a point in a given cell"""
        phi = self.evaluate_basis(cell_id, point)
        u_val = 0.0
        for i in range(self.n_dofs_per_cell):
            i_global = self.local_to_global(cell_id, i)
            u_val += u_dofs[i_global] * phi[i]
        return u_val