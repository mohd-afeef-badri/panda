import sys
from pathlib import Path

# Add grandparent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from panda.lib import boundary_conditions

class P1DGStokesSolver:
    """
    P1 Discontinuous Galerkin solver for Stokes equations with proper numerical integration.
    
    Solves: -μ Δu + ∇p = f
            div(u) = 0
    
    Key improvements over basic version:
    - Multi-point Gaussian quadrature for volume integrals
    - 2-point Gauss quadrature for edge integrals
    - Proper SIPG formulation for velocity
    - Pressure jump stabilization
    """
    
    def __init__(self, mesh, bc_manager, viscosity=1.0, penalty_u=40.0, penalty_p=0.5):
        """
        Parameters
        ----------
        mesh : PolygonalMesh
        bc_manager : BoundaryConditionManager
        viscosity : float
            Dynamic viscosity μ
        penalty_u : float
            SIPG penalty parameter for velocity (γ_u)
        penalty_p : float
            Pressure jump stabilization parameter (γ_p)
        """
        self.mesh = mesh
        self.bc_manager = bc_manager
        self.mu = viscosity
        self.gamma_u = penalty_u
        self.gamma_p = penalty_p
        self.dofs_per_cell = 9  # 3 for u, 3 for v, 3 for p
        self.n_dofs = mesh.n_cells * self.dofs_per_cell
        self.regularization = 1e-6  # Pressure nullspace fix

    def get_indices(self, cell_id):
        """Get DOF indices for u, v, p components"""
        base = cell_id * self.dofs_per_cell
        return {
            'u': np.arange(base, base + 3),
            'v': np.arange(base + 3, base + 6),
            'p': np.arange(base + 6, base + 9)
        }

    def evaluate_basis(self, cell_id, point, derivatives=False):
        """Evaluate P1 basis: [1, x-x_c, y-y_c] and optionally its gradients"""
        cent = self.mesh.cell_centroid(cell_id)
        x_rel, y_rel = point[0] - cent[0], point[1] - cent[1]
        phi = np.array([1.0, x_rel, y_rel])
        
        if not derivatives:
            return phi
        
        grad_x = np.array([0.0, 1.0, 0.0])
        grad_y = np.array([0.0, 0.0, 1.0])
        return phi, grad_x, grad_y

    def get_triangle_quadrature(self, vertices, order=2):
        """
        Gaussian quadrature on a triangle
        
        Parameters
        ----------
        vertices : array (3, 2) - triangle vertices
        order : int - quadrature order
        
        Returns
        -------
        points : array (n_quad, 2)
        weights : array (n_quad,)
        """
        area = 0.5 * abs((vertices[1,0] - vertices[0,0]) * (vertices[2,1] - vertices[0,1]) -
                         (vertices[2,0] - vertices[0,0]) * (vertices[1,1] - vertices[0,1]))
        
        if order == 1:
            ref_pts = np.array([[1/3, 1/3]])
            ref_wts = np.array([1.0])
        elif order == 2:
            ref_pts = np.array([
                [1/6, 1/6],
                [2/3, 1/6],
                [1/6, 2/3]
            ])
            ref_wts = np.array([1/3, 1/3, 1/3])
        elif order == 3:
            ref_pts = np.array([
                [1/3, 1/3],
                [0.6, 0.2],
                [0.2, 0.6],
                [0.2, 0.2]
            ])
            ref_wts = np.array([-27/48, 25/48, 25/48, 25/48])
        else:
            raise ValueError(f"Order {order} not supported")
        
        # Map to physical triangle
        points = []
        for xi, eta in ref_pts:
            x = vertices[0] + xi * (vertices[1] - vertices[0]) + eta * (vertices[2] - vertices[0])
            points.append(x)
        points = np.array(points)
        
        weights = ref_wts * 2 * area
        return points, weights

    def get_polygon_quadrature(self, cell_id, order=2):
        """
        Quadrature on general polygon by triangulation
        """
        vertices = self.mesh.vertices[self.mesh.cells[cell_id]]
        n_verts = len(vertices)
        
        if n_verts == 3:
            return self.get_triangle_quadrature(vertices, order)
        
        # Triangulate from centroid
        cent = self.mesh.cell_centroid(cell_id)
        all_points = []
        all_weights = []
        
        for i in range(n_verts):
            v1 = vertices[i]
            v2 = vertices[(i+1) % n_verts]
            tri_verts = np.array([cent, v1, v2])
            pts, wts = self.get_triangle_quadrature(tri_verts, order)
            all_points.append(pts)
            all_weights.append(wts)
        
        return np.vstack(all_points), np.concatenate(all_weights)

    def get_edge_quadrature(self, edge_id, order=2):
        """
        Gaussian quadrature on an edge
        """
        edge = self.mesh.edges[edge_id]
        p1 = self.mesh.vertices[edge[0]]
        p2 = self.mesh.vertices[edge[1]]
        h_e = self.mesh.edge_length(edge_id)
        
        if order == 1:
            ref_pts = np.array([0.5])
            ref_wts = np.array([1.0])
        elif order == 2:
            ref_pts = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
            ref_wts = np.array([0.5, 0.5])
        elif order == 3:
            ref_pts = np.array([0.5 - np.sqrt(15)/10, 0.5, 0.5 + np.sqrt(15)/10])
            ref_wts = np.array([5/18, 8/18, 5/18])
        else:
            raise ValueError(f"Order {order} not supported")
        
        points = np.array([p1 + t * (p2 - p1) for t in ref_pts])
        weights = ref_wts * h_e
        
        return points, weights

    def assemble(self, f_func):
        """Assemble Stokes system with proper numerical integration"""
        A = lil_matrix((self.n_dofs, self.n_dofs))
        b = np.zeros(self.n_dofs)

        # 1. Volume Terms with Quadrature
        for c_id in range(self.mesh.n_cells):
            idx = self.get_indices(c_id)
            
            # Get quadrature points and weights
            quad_pts, quad_wts = self.get_polygon_quadrature(c_id, order=2)
            
            for qp, qw in zip(quad_pts, quad_wts):
                phi, gx, gy = self.evaluate_basis(c_id, qp, derivatives=True)
                
                # Body force
                fx, fy = f_func(qp[0], qp[1])
                for i in range(3):
                    b[idx['u'][i]] += fx * phi[i] * qw
                    b[idx['v'][i]] += fy * phi[i] * qw
                
                # Pressure regularization (integrated over cell)
                for i in range(3):
                    A[idx['p'][i], idx['p'][i]] -= self.regularization * qw
                
                for i in range(3):
                    for j in range(3):
                        # Viscosity: μ (∇u, ∇v)
                        visc = self.mu * (gx[i]*gx[j] + gy[i]*gy[j]) * qw
                        A[idx['u'][i], idx['u'][j]] += visc
                        A[idx['v'][i], idx['v'][j]] += visc
                        
                        # Pressure-divergence coupling: -(p, div v)
                        A[idx['p'][i], idx['u'][j]] -= phi[i] * gx[j] * qw
                        A[idx['p'][i], idx['v'][j]] -= phi[i] * gy[j] * qw
                        
                        # Gradient: -(q, div u) (symmetric part)
                        A[idx['u'][j], idx['p'][i]] -= phi[i] * gx[j] * qw
                        A[idx['v'][j], idx['p'][i]] -= phi[i] * gy[j] * qw

        # 2. Face Terms with Quadrature
        for e_id in range(len(self.mesh.edges)):
            cells = self.mesh.edge_to_cells[self.mesh.edges[e_id]]
            h_e = self.mesh.edge_length(e_id)
            
            # Penalty parameters
            sig_u = (self.gamma_u * self.mu) / h_e
            sig_p = self.gamma_p * h_e

            if len(cells) == 2:  # Interior edge
                self._assemble_interior_face_quad(A, e_id, cells, h_e, sig_u, sig_p)
            else:  # Boundary edge
                bc = self.bc_manager.get_bc(e_id)
                if bc.bc_type == 'dirichlet':
                    self._assemble_dirichlet_face_quad(A, b, e_id, cells[0], h_e, sig_u, bc)
                elif bc.bc_type == 'neumann':
                    self._assemble_neumann_face_quad(A, b, e_id, cells[0], h_e, bc)

        return A.tocsr(), b

    def _assemble_interior_face_quad(self, A, edge_id, cells, h_e, sig_u, sig_p):
        """Interior face assembly with 2-point Gauss quadrature"""
        c_i, c_j = cells
        idx_i = self.get_indices(c_i)
        idx_j = self.get_indices(c_j)
        n = self.mesh.edge_normal(edge_id, c_i)
        
        # Get quadrature points
        quad_pts, quad_wts = self.get_edge_quadrature(edge_id, order=2)
        
        for qp, qw in zip(quad_pts, quad_wts):
            phi_i, gx_i, gy_i = self.evaluate_basis(c_i, qp, derivatives=True)
            phi_j, gx_j, gy_j = self.evaluate_basis(c_j, qp, derivatives=True)
            
            # Normal derivatives: ∂/∂n = n·∇
            gn_i = gx_i * n[0] + gy_i * n[1]
            gn_j = gx_j * n[0] + gy_j * n[1]
            
            for i in range(3):
                for j in range(3):
                    # ===== VELOCITY: SIPG =====
                    # Bilinear form: -<{μ ∂u/∂n}, [[v]]> - <{μ ∂v/∂n}, [[u]]> + (σ/h)<[[u]], [[v]]>
                    
                    # Penalty term: (σ/h) ∫ [[u]]·[[v]] ds
                    # [[u]] = u_i - u_j, [[v]] = v_i - v_j
                    pen = sig_u * qw
                    
                    # Self-self (i-i)
                    term_ii = pen * phi_i[i] * phi_i[j] - 0.5 * self.mu * qw * (gn_i[j]*phi_i[i] + gn_i[i]*phi_i[j])
                    A[idx_i['u'][i], idx_i['u'][j]] += term_ii
                    A[idx_i['v'][i], idx_i['v'][j]] += term_ii
                    
                    # Self-neighbor (i-j)
                    term_ij = -pen * phi_i[i] * phi_j[j] - 0.5 * self.mu * qw * (gn_j[j]*phi_i[i] - gn_i[i]*phi_j[j])
                    A[idx_i['u'][i], idx_j['u'][j]] += term_ij
                    A[idx_i['v'][i], idx_j['v'][j]] += term_ij
                    
                    # Neighbor-self (j-i)
                    term_ji = -pen * phi_j[i] * phi_i[j] - 0.5 * self.mu * qw * (-gn_i[j]*phi_j[i] + gn_j[i]*phi_i[j])
                    A[idx_j['u'][i], idx_i['u'][j]] += term_ji
                    A[idx_j['v'][i], idx_i['v'][j]] += term_ji
                    
                    # Neighbor-neighbor (j-j)
                    term_jj = pen * phi_j[i] * phi_j[j] - 0.5 * self.mu * qw * (-gn_j[j]*phi_j[i] - gn_j[i]*phi_j[j])
                    A[idx_j['u'][i], idx_j['u'][j]] += term_jj
                    A[idx_j['v'][i], idx_j['v'][j]] += term_jj
                    
                    # ===== PRESSURE: Jump Stabilization =====
                    # (γ_p h) ∫ [[p]]·[[q]] ds
                    jump_p_i = phi_i[i] - phi_j[i]
                    jump_p_j = phi_i[j] - phi_j[j]
                    p_stab = sig_p * qw * jump_p_i * jump_p_j
                    
                    A[idx_i['p'][i], idx_i['p'][j]] += p_stab
                    A[idx_i['p'][i], idx_j['p'][j]] -= p_stab
                    A[idx_j['p'][i], idx_i['p'][j]] -= p_stab
                    A[idx_j['p'][i], idx_j['p'][j]] += p_stab

    def _assemble_dirichlet_face_quad(self, A, b, edge_id, cell_i, h_e, sig_u, bc):
        """Dirichlet BC assembly with 2-point Gauss quadrature"""
        idx_i = self.get_indices(cell_i)
        n = self.mesh.edge_normal(edge_id, cell_i)
        
        quad_pts, quad_wts = self.get_edge_quadrature(edge_id, order=2)
        
        for qp, qw in zip(quad_pts, quad_wts):
            phi_i, gx_i, gy_i = self.evaluate_basis(cell_i, qp, derivatives=True)
            gn_i = gx_i * n[0] + gy_i * n[1]
            
            # Evaluate BC
            bc_val = bc.evaluate(qp[0], qp[1])
            if bc.is_vector:
                gu, gv = bc_val
            else:
                gu, gv = bc_val, bc_val
            
            for i in range(3):
                # RHS: Nitsche method
                b[idx_i['u'][i]] += (sig_u * phi_i[i] - self.mu * gn_i[i]) * gu * qw
                b[idx_i['v'][i]] += (sig_u * phi_i[i] - self.mu * gn_i[i]) * gv * qw
                
                # Pressure BC contribution: -∫ q (g·n) ds
                b[idx_i['p'][i]] -= phi_i[i] * (gu*n[0] + gv*n[1]) * qw
                
                for j in range(3):
                    # LHS: Penalty + consistency
                    val = (sig_u * phi_i[i] * phi_i[j] 
                           - self.mu * phi_i[i] * gn_i[j] 
                           - self.mu * gn_i[i] * phi_i[j]) * qw
                    A[idx_i['u'][i], idx_i['u'][j]] += val
                    A[idx_i['v'][i], idx_i['v'][j]] += val

    def _assemble_neumann_face_quad(self, A, b, edge_id, cell_i, h_e, bc):
        """Neumann BC assembly with 2-point Gauss quadrature"""
        idx_i = self.get_indices(cell_i)
        
        quad_pts, quad_wts = self.get_edge_quadrature(edge_id, order=2)
        
        for qp, qw in zip(quad_pts, quad_wts):
            phi_i = self.evaluate_basis(cell_i, qp, derivatives=False)
            
            bc_val = bc.evaluate(qp[0], qp[1])
            if bc.is_vector:
                trac_x, trac_y = bc_val
            else:
                trac_x, trac_y = bc_val, bc_val
            
            for i in range(3):
                b[idx_i['u'][i]] += phi_i[i] * trac_x * qw
                b[idx_i['v'][i]] += phi_i[i] * trac_y * qw

    def solve(self, f_func):
        """Solve the Stokes system"""
        print(f"Assembling system (Grid: {self.mesh.n_cells} cells)...")
        A, b = self.assemble(f_func)
        
        print(f"Solving linear system (DOFs: {self.n_dofs})...")
        
        # Diagnostics
        print(f"  Matrix size: {A.shape}")
        print(f"  Nonzeros: {A.nnz}")
        print(f"  Sparsity: {100 * (1 - A.nnz / (A.shape[0] * A.shape[1])):.1f}%")
        print(f"  RHS norm: {np.linalg.norm(b):.6e}")
        
        try:
            u_dofs = spsolve(A, b)
        except Exception as e:
            print(f"Solver failed: {e}")
            return None
        
        if np.any(np.isnan(u_dofs)):
            print("ERROR: Solver returned NaNs. The system is unstable.")
            print("Try increasing penalty_u or checking mesh connectivity.")
            return None
        
        # Check residual
        residual = np.linalg.norm(A @ u_dofs - b)
        print(f"  Solution residual: {residual:.6e}")
        print(f"  Solution norm: {np.linalg.norm(u_dofs):.6e}")
        
        return u_dofs

    def evaluate_solution(self, solution_vector, point, cell_id):
        """
        Evaluate (u, v, p) at a specific point within a cell.
        
        Parameters
        ----------
        solution_vector : ndarray
            Solution DOF vector
        point : array (2,)
            Point coordinates (x, y)
        cell_id : int
            Cell index
        
        Returns
        -------
        tuple : (u_val, v_val, p_val)
        """
        idx = self.get_indices(cell_id)
        phi = self.evaluate_basis(cell_id, point, derivatives=False)
        
        u_val = np.dot(solution_vector[idx['u']], phi)
        v_val = np.dot(solution_vector[idx['v']], phi)
        p_val = np.dot(solution_vector[idx['p']], phi)
        
        return u_val, v_val, p_val
    
    def compute_velocity_divergence(self, solution_vector, cell_id):
        """
        Compute the divergence of velocity in a cell.
        For P1: div(u) is constant in each cell.
        
        Returns
        -------
        float : div(u) = ∂u/∂x + ∂v/∂y
        """
        idx = self.get_indices(cell_id)
        
        # For P1 basis [1, x-x_c, y-y_c]:
        # u = u_0 + u_1*(x-x_c) + u_2*(y-y_c)
        # ∂u/∂x = u_1
        # v = v_0 + v_1*(x-x_c) + v_2*(y-y_c)
        # ∂v/∂y = v_2
        
        du_dx = solution_vector[idx['u'][1]]  # Coefficient of (x-x_c)
        dv_dy = solution_vector[idx['v'][2]]  # Coefficient of (y-y_c)
        
        return du_dx + dv_dy
