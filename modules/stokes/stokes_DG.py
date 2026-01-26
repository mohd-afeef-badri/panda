import sys
from pathlib import Path

# Add grandparent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from panda.lib import boundary_conditions

class P1DGStokesSolver:
    def __init__(self, mesh, viscosity=1.0, penalty_u=40.0, penalty_p=1.0):
        self.mesh = mesh
        self.mu = viscosity
        self.gamma_u = penalty_u    
        self.gamma_p = penalty_p    
        self.dofs_per_cell = 9      
        self.n_dofs = mesh.n_cells * self.dofs_per_cell
        self.regularization = 1e-6  # Small epsilon to fix pressure nullspace

    def get_indices(self, cell_id):
        base = cell_id * self.dofs_per_cell
        return {
            'u': np.arange(base, base + 3),
            'v': np.arange(base + 3, base + 6),
            'p': np.arange(base + 6, base + 9)
        }

    def evaluate_basis(self, cell_id, point, derivatives=False):
        cent = self.mesh.cell_centroid(cell_id)
        x_rel, y_rel = point[0] - cent[0], point[1] - cent[1]
        phi = np.array([1.0, x_rel, y_rel])
        if not derivatives: return phi
        return phi, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])

    def assemble(self, f_func, g_func):
        # LIL is fast for construction
        A = lil_matrix((self.n_dofs, self.n_dofs))
        b = np.zeros(self.n_dofs)

        # --- 1. Volume Terms ---
        for c_id in range(self.mesh.n_cells):
            area = self.mesh.cell_area(c_id)
            cent = self.mesh.cell_centroid(c_id)
            idx = self.get_indices(c_id)
            phi, gx, gy = self.evaluate_basis(c_id, cent, True)
            fx, fy = f_func(cent[0], cent[1])

            for i in range(3):
                # Force Terms
                b[idx['u'][i]] += fx * phi[i] * area
                b[idx['v'][i]] += fy * phi[i] * area
                
                # Pressure Regularization (Epsilon * p * q)
                # Adds a small diagonal term to pressure block to ensure invertibility
                A[idx['p'][i], idx['p'][i]] -= self.regularization * area

                for j in range(3):
                    # Viscosity: (grad u, grad v)
                    visc = self.mu * (gx[i]*gx[j] + gy[i]*gy[j]) * area
                    A[idx['u'][i], idx['u'][j]] += visc
                    A[idx['v'][i], idx['v'][j]] += visc
                    
                    # Divergence: - p * div(u)
                    A[idx['p'][i], idx['u'][j]] -= phi[i] * gx[j] * area
                    A[idx['p'][i], idx['v'][j]] -= phi[i] * gy[j] * area
                    
                    # Gradient: - q * div(u) (Symmetric part)
                    A[idx['u'][j], idx['p'][i]] -= phi[i] * gx[j] * area
                    A[idx['v'][j], idx['p'][i]] -= phi[i] * gy[j] * area

        # --- 2. Face Terms ---
        for e_id, edge in enumerate(self.mesh.edges):
            cells = self.mesh.edge_to_cells[edge]
            h_e = self.mesh.edge_length(e_id)
            mid = self.mesh.edge_midpoint(e_id)
            
            # Penalties
            sig_u = (self.gamma_u * self.mu) / h_e
            sig_p = self.gamma_p * h_e

            if len(cells) == 2: # Interior Edge
                c_i, c_j = cells
                idx_i, idx_j = self.get_indices(c_i), self.get_indices(c_j)
                n = self.mesh.edge_normal(e_id, c_i)
                phi_i, gx_i, gy_i = self.evaluate_basis(c_i, mid, True)
                phi_j, gx_j, gy_j = self.evaluate_basis(c_j, mid, True)
                gn_i = gx_i*n[0]+gy_i*n[1] 
                gn_j = gx_j*n[0]+gy_j*n[1]

                for i in range(3):
                    for j in range(3):
                        # Velocity: SIPG (Symmetric Interior Penalty Galerkin)
                        # Term: < {mu grad u}, [[v]] > + < {mu grad v}, [[u]] > + penalty
                        
                        # A_ii terms (Self)
                        term_ii = sig_u*phi_i[i]*phi_i[j] - 0.5*self.mu*(gn_i[j]*phi_i[i] + gn_i[i]*phi_i[j])
                        A[idx_i['u'][i], idx_i['u'][j]] += term_ii * h_e
                        A[idx_i['v'][i], idx_i['v'][j]] += term_ii * h_e

                        # A_ij terms (Neighbor)
                        term_ij = -sig_u*phi_i[i]*phi_j[j] - 0.5*self.mu*(gn_j[j]*phi_i[i] - gn_i[i]*phi_j[j])
                        A[idx_i['u'][i], idx_j['u'][j]] += term_ij * h_e
                        A[idx_i['v'][i], idx_j['v'][j]] += term_ij * h_e

                        # A_ji terms (Neighbor to Self)
                        term_ji = -sig_u*phi_j[i]*phi_i[j] - 0.5*self.mu*(-gn_i[j]*phi_j[i] + gn_j[i]*phi_i[j])
                        A[idx_j['u'][i], idx_i['u'][j]] += term_ji * h_e
                        A[idx_j['v'][i], idx_i['v'][j]] += term_ji * h_e

                        # A_jj terms (Neighbor Self)
                        term_jj = sig_u*phi_j[i]*phi_j[j] - 0.5*self.mu*(-gn_j[j]*phi_j[i] - gn_j[i]*phi_j[j])
                        A[idx_j['u'][i], idx_j['u'][j]] += term_jj * h_e
                        A[idx_j['v'][i], idx_j['v'][j]] += term_jj * h_e
                        
                        # Pressure Stabilization (Jump Penalty)
                        p_stab = sig_p * (phi_i[i]-phi_j[i]) * (phi_i[j]-phi_j[j]) * h_e
                        A[idx_i['p'][i], idx_i['p'][j]] += p_stab
                        A[idx_i['p'][i], idx_j['p'][j]] -= p_stab
                        A[idx_j['p'][i], idx_i['p'][j]] -= p_stab
                        A[idx_j['p'][i], idx_j['p'][j]] += p_stab

            else: # Boundary Edge
                c_i = cells[0]
                idx_i = self.get_indices(c_i)
                n = self.mesh.edge_normal(e_id, c_i)
                gu, gv = g_func(mid[0], mid[1])
                phi_i, gx_i, gy_i = self.evaluate_basis(c_i, mid, True)
                gn_i = gx_i*n[0] + gy_i*n[1]

                for i in range(3):
                    # RHS (Nitsche)
                    b[idx_i['u'][i]] += (sig_u*phi_i[i] - self.mu*gn_i[i])*gu*h_e
                    b[idx_i['v'][i]] += (sig_u*phi_i[i] - self.mu*gn_i[i])*gv*h_e
                    b[idx_i['p'][i]] -= phi_i[i]*(gu*n[0] + gv*n[1])*h_e

                    for j in range(3):
                        # LHS Penalty
                        val = (sig_u*phi_i[i]*phi_i[j] - self.mu*phi_i[i]*gn_i[j] - self.mu*gn_i[i]*phi_i[j]) * h_e
                        A[idx_i['u'][i], idx_i['u'][j]] += val
                        A[idx_i['v'][i], idx_i['v'][j]] += val

        return A.tocsr(), b

    def solve(self, f_func, g_func):
        print(f"Assembling system (Grid: {self.mesh.n_cells} cells)...")
        A, b = self.assemble(f_func, g_func)
        
        print(f"Solving linear system (DOFs: {self.n_dofs})...")
        try:
            u_dofs = spsolve(A, b)
        except Exception as e:
            print(f"Solver failed: {e}")
            return None
            
        if np.any(np.isnan(u_dofs)):
            print("ERROR: Solver returned NaNs. The system is unstable.")
            print("Try increasing penalty_u or checking mesh connectivity.")
            return None
            
        return u_dofs

    def evaluate_solution(self, solution_vector, point, cell_id):
        """
        Evaluate (u, v, p) at a specific point within a cell.
        Returns: tuple (u_val, v_val, p_val)
        """
        idx = self.get_indices(cell_id)
        phi = self.evaluate_basis(cell_id, point) # Returns [1, x, y]
        
        # Reconstruct values using the basis functions
        # u(x) = sum(u_i * phi_i)
        u_val = np.sum(solution_vector[idx['u']] * phi)
        v_val = np.sum(solution_vector[idx['v']] * phi)
        p_val = np.sum(solution_vector[idx['p']] * phi)
        
        return u_val, v_val, p_val
