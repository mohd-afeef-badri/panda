import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from panda.lib import boundary_conditions

class P1DGLinearElasticitySolver:
    """
    P1 Discontinuous Galerkin solver for linear elasticity with proper numerical integration.
    
    Features
    --------:
    - Multi-point Gaussian quadrature for volume integrals
    - 2-point Gauss quadrature for edge integrals
    - Full SIPG formulation with consistency terms
    """
    
    def __init__(self, mesh, bc_manager, lame_lambda=1.0, lame_mu=1.0, penalty_param=10.0):
        self.mesh = mesh
        self.bc_manager = bc_manager
        self.lam = lame_lambda
        self.mu = lame_mu
        self.penalty = penalty_param
        self.n_dofs_per_cell = 6
        self.n_dofs = mesh.n_cells * self.n_dofs_per_cell

    def get_indices(self, cell_id):
        """Get DOF indices for displacement components u_x and u_y"""
        base = cell_id * self.n_dofs_per_cell
        return {
            'ux': np.arange(base, base + 3),
            'uy': np.arange(base + 3, base + 6),
        }

    def evaluate_basis(self, cell_id, point, derivatives=False):
        """Evaluate P1 basis: [1, x-x_c, y-y_c] and optionally their gradients"""
        cent = self.mesh.cell_centroid(cell_id)
        x_rel = point[0] - cent[0]
        y_rel = point[1] - cent[1]
        
        phi = np.array([1.0, x_rel, y_rel])
        
        if not derivatives:
            return phi
        else:
            grad_x = np.array([0.0, 1.0, 0.0])
            grad_y = np.array([0.0, 0.0, 1.0])
            return phi, grad_x, grad_y

    def evaluate_solution(self, u_dofs, point, cell_id):
        phi = self.evaluate_basis(cell_id, point, derivatives=False)
        idx = self.get_indices(cell_id)
        ux = np.dot(u_dofs[idx['ux']], phi)
        uy = np.dot(u_dofs[idx['uy']], phi)
        return ux, uy

    def compute_stress(self, grad_ux, grad_uy, normal):
        """
        Compute σ(u) · n
        
        Parameters
        ----------
        grad_ux : tuple (du_x/dx, du_x/dy)
        grad_uy : tuple (du_y/dx, du_y/dy)
        normal : array (n_x, n_y)
        
        Returns
        -------
        (σ·n)_x, (σ·n)_y
        """
        du_x_dx, du_x_dy = grad_ux
        du_y_dx, du_y_dy = grad_uy
        
        # Strain
        eps_xx = du_x_dx
        eps_yy = du_y_dy
        eps_xy = 0.5 * (du_x_dy + du_y_dx)
        
        # Stress: σ = λ(div u)I + 2μ ε
        div_u = eps_xx + eps_yy
        sigma_xx = self.lam * div_u + 2 * self.mu * eps_xx
        sigma_yy = self.lam * div_u + 2 * self.mu * eps_yy
        sigma_xy = 2 * self.mu * eps_xy
        
        # σ · n
        sigma_n_x = sigma_xx * normal[0] + sigma_xy * normal[1]
        sigma_n_y = sigma_xy * normal[0] + sigma_yy * normal[1]
        
        return sigma_n_x, sigma_n_y

    def get_triangle_quadrature(self, vertices, order=2):
        """
        Gaussian quadrature on a triangle
        
        Parameters
        ----------
        vertices : array (3, 2) - triangle vertices
        order : int - quadrature order (1, 2, or 3)
        
        Returns
        -------
        points : array (n_quad, 2) - quadrature points in physical space
        weights : array (n_quad,) - quadrature weights (already scaled by |J|)
        """
        area = 0.5 * abs((vertices[1,0] - vertices[0,0]) * (vertices[2,1] - vertices[0,1]) -
                         (vertices[2,0] - vertices[0,0]) * (vertices[1,1] - vertices[0,1]))
        
        if order == 1:
            # 1-point (centroid)
            ref_pts = np.array([[1/3, 1/3]])
            ref_wts = np.array([1.0])
        elif order == 2:
            # 3-point quadrature
            ref_pts = np.array([
                [1/6, 1/6],
                [2/3, 1/6],
                [1/6, 2/3]
            ])
            ref_wts = np.array([1/3, 1/3, 1/3])
        elif order == 3:
            # 4-point quadrature
            ref_pts = np.array([
                [1/3, 1/3],
                [0.6, 0.2],
                [0.2, 0.6],
                [0.2, 0.2]
            ])
            ref_wts = np.array([-27/48, 25/48, 25/48, 25/48])
        else:
            raise ValueError(f"Order {order} not supported")
        
        # Map reference points to physical triangle
        points = []
        for xi, eta in ref_pts:
            x = vertices[0] + xi * (vertices[1] - vertices[0]) + eta * (vertices[2] - vertices[0])
            points.append(x)
        points = np.array(points)
        
        # Scale weights by triangle area (Jacobian = 2*area)
        weights = ref_wts * 2 * area
        
        return points, weights

    def get_polygon_quadrature(self, cell_id, order=2):
        """
        Quadrature on general polygon by triangulation
        
        Returns
        -------
        points : array (n_quad, 2)
        weights : array (n_quad,)
        """
        vertices = self.mesh.vertices[self.mesh.cells[cell_id]]
        n_verts = len(vertices)
        
        if n_verts == 3:
            # Already a triangle
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
        
        Returns
        -------
        points : array (n_quad, 2)
        weights : array (n_quad,)
        """
        edge = self.mesh.edges[edge_id]
        p1 = self.mesh.vertices[edge[0]]
        p2 = self.mesh.vertices[edge[1]]
        h_e = self.mesh.edge_length(edge_id)
        
        if order == 1:
            # Midpoint rule
            ref_pts = np.array([0.5])
            ref_wts = np.array([1.0])
        elif order == 2:
            # 2-point Gauss
            ref_pts = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
            ref_wts = np.array([0.5, 0.5])
        elif order == 3:
            # 3-point Gauss
            ref_pts = np.array([0.5 - np.sqrt(15)/10, 0.5, 0.5 + np.sqrt(15)/10])
            ref_wts = np.array([5/18, 8/18, 5/18])
        else:
            raise ValueError(f"Order {order} not supported")
        
        # Map to physical edge
        points = np.array([p1 + t * (p2 - p1) for t in ref_pts])
        weights = ref_wts * h_e
        
        return points, weights

    def assemble_system(self, f_func):
        """Assemble SIPG system with proper numerical integration"""
        A = lil_matrix((self.n_dofs, self.n_dofs))
        b = np.zeros(self.n_dofs)
        
        # 1. Volume integrals with quadrature
        for cell_id in range(self.mesh.n_cells):
            idx = self.get_indices(cell_id)
            
            # Get quadrature points and weights
            quad_pts, quad_wts = self.get_polygon_quadrature(cell_id, order=2)
            
            for qp, qw in zip(quad_pts, quad_wts):
                phi, grad_x, grad_y = self.evaluate_basis(cell_id, qp, derivatives=True)
                
                # Body force
                f_x, f_y = f_func(qp[0], qp[1])
                for i in range(3):
                    b[idx['ux'][i]] += f_x * phi[i] * qw
                    b[idx['uy'][i]] += f_y * phi[i] * qw
                
                # Stiffness matrix
                for i in range(3):
                    for j in range(3):
                        # λ div(v) div(u)
                        A[idx['ux'][i], idx['ux'][j]] += self.lam * grad_x[i] * grad_x[j] * qw
                        A[idx['uy'][i], idx['uy'][j]] += self.lam * grad_y[i] * grad_y[j] * qw
                        A[idx['ux'][i], idx['uy'][j]] += self.lam * grad_x[i] * grad_y[j] * qw
                        A[idx['uy'][i], idx['ux'][j]] += self.lam * grad_y[i] * grad_x[j] * qw
                        
                        # 2μ ε(v) : ε(u)
                        A[idx['ux'][i], idx['ux'][j]] += 2*self.mu * grad_x[i] * grad_x[j] * qw
                        A[idx['uy'][i], idx['uy'][j]] += 2*self.mu * grad_y[i] * grad_y[j] * qw
                        A[idx['ux'][i], idx['ux'][j]] += self.mu * grad_y[i] * grad_y[j] * qw
                        A[idx['uy'][i], idx['uy'][j]] += self.mu * grad_x[i] * grad_x[j] * qw
                        A[idx['ux'][i], idx['uy'][j]] += self.mu * grad_y[i] * grad_x[j] * qw
                        A[idx['uy'][i], idx['ux'][j]] += self.mu * grad_x[i] * grad_y[j] * qw
        
        # 2. Face integrals with quadrature
        for edge_id in range(len(self.mesh.edges)):
            cells = self.mesh.edge_to_cells[self.mesh.edges[edge_id]]
            h_e = self.mesh.edge_length(edge_id)
            
            if len(cells) == 2:  # Interior edge
                self._assemble_interior_face_quad(A, edge_id, cells, h_e)
            else:  # Boundary edge
                bc = self.bc_manager.get_bc(edge_id)
                if bc.bc_type == 'dirichlet':
                    self._assemble_dirichlet_face_quad(A, b, edge_id, cells[0], h_e, bc)
                elif bc.bc_type == 'neumann':
                    self._assemble_neumann_face_quad(A, b, edge_id, cells[0], h_e, bc)
        
        return csr_matrix(A), b

    def _assemble_interior_face_quad(self, A, edge_id, cells, h_e):
        """Interior face with 2-point Gauss quadrature"""
        cell_i, cell_j = cells
        n = self.mesh.edge_normal(edge_id, cell_i)
        
        idx_i = self.get_indices(cell_i)
        idx_j = self.get_indices(cell_j)
        
        # Get quadrature points
        quad_pts, quad_wts = self.get_edge_quadrature(edge_id, order=2)
        
        sigma = self.penalty / h_e
        
        for qp, qw in zip(quad_pts, quad_wts):
            phi_i, gx_i, gy_i = self.evaluate_basis(cell_i, qp, derivatives=True)
            phi_j, gx_j, gy_j = self.evaluate_basis(cell_j, qp, derivatives=True)
            
            for i in range(3):
                for j in range(3):
                    # Penalty term
                    pen_term = sigma * qw
                    A[idx_i['ux'][i], idx_i['ux'][j]] += pen_term * phi_i[i] * phi_i[j]
                    A[idx_i['ux'][i], idx_j['ux'][j]] -= pen_term * phi_i[i] * phi_j[j]
                    A[idx_j['ux'][i], idx_i['ux'][j]] -= pen_term * phi_j[i] * phi_i[j]
                    A[idx_j['ux'][i], idx_j['ux'][j]] += pen_term * phi_j[i] * phi_j[j]
                    
                    A[idx_i['uy'][i], idx_i['uy'][j]] += pen_term * phi_i[i] * phi_i[j]
                    A[idx_i['uy'][i], idx_j['uy'][j]] -= pen_term * phi_i[i] * phi_j[j]
                    A[idx_j['uy'][i], idx_i['uy'][j]] -= pen_term * phi_j[i] * phi_i[j]
                    A[idx_j['uy'][i], idx_j['uy'][j]] += pen_term * phi_j[i] * phi_j[j]
                    
                    # Consistency terms with coupling
                    grad_ux_i = (gx_i[j], gy_i[j])
                    grad_uy_i = (0.0, 0.0)
                    sigma_ux_i_x, sigma_ux_i_y = self.compute_stress(grad_ux_i, grad_uy_i, n)
                    
                    grad_ux_j = (gx_j[j], gy_j[j])
                    grad_uy_j = (0.0, 0.0)
                    sigma_ux_j_x, sigma_ux_j_y = self.compute_stress(grad_ux_j, grad_uy_j, n)
                    
                    grad_ux_i_y = (0.0, 0.0)
                    grad_uy_i_y = (gx_i[j], gy_i[j])
                    sigma_uy_i_x, sigma_uy_i_y = self.compute_stress(grad_ux_i_y, grad_uy_i_y, n)
                    
                    grad_ux_j_y = (0.0, 0.0)
                    grad_uy_j_y = (gx_j[j], gy_j[j])
                    sigma_uy_j_x, sigma_uy_j_y = self.compute_stress(grad_ux_j_y, grad_uy_j_y, n)
                    
                    avg_sigma_ux_x = 0.5 * (sigma_ux_i_x + sigma_ux_j_x)
                    avg_sigma_ux_y = 0.5 * (sigma_ux_i_y + sigma_ux_j_y)
                    avg_sigma_uy_x = 0.5 * (sigma_uy_i_x + sigma_uy_j_x)
                    avg_sigma_uy_y = 0.5 * (sigma_uy_i_y + sigma_uy_j_y)
                    
                    # Consistency: -∫{σ(u)·n}·[[v]]
                    cons_xx = -qw * avg_sigma_ux_x
                    cons_xy = -qw * avg_sigma_uy_x
                    cons_yx = -qw * avg_sigma_ux_y
                    cons_yy = -qw * avg_sigma_uy_y
                    
                    A[idx_i['ux'][i], idx_i['ux'][j]] += cons_xx * phi_i[i]
                    A[idx_i['ux'][i], idx_j['ux'][j]] -= cons_xx * phi_i[i]
                    A[idx_j['ux'][i], idx_i['ux'][j]] -= cons_xx * phi_j[i]
                    A[idx_j['ux'][i], idx_j['ux'][j]] += cons_xx * phi_j[i]
                    
                    A[idx_i['ux'][i], idx_i['uy'][j]] += cons_xy * phi_i[i]
                    A[idx_i['ux'][i], idx_j['uy'][j]] -= cons_xy * phi_i[i]
                    A[idx_j['ux'][i], idx_i['uy'][j]] -= cons_xy * phi_j[i]
                    A[idx_j['ux'][i], idx_j['uy'][j]] += cons_xy * phi_j[i]
                    
                    A[idx_i['uy'][i], idx_i['ux'][j]] += cons_yx * phi_i[i]
                    A[idx_i['uy'][i], idx_j['ux'][j]] -= cons_yx * phi_i[i]
                    A[idx_j['uy'][i], idx_i['ux'][j]] -= cons_yx * phi_j[i]
                    A[idx_j['uy'][i], idx_j['ux'][j]] += cons_yx * phi_j[i]
                    
                    A[idx_i['uy'][i], idx_i['uy'][j]] += cons_yy * phi_i[i]
                    A[idx_i['uy'][i], idx_j['uy'][j]] -= cons_yy * phi_i[i]
                    A[idx_j['uy'][i], idx_i['uy'][j]] -= cons_yy * phi_j[i]
                    A[idx_j['uy'][i], idx_j['uy'][j]] += cons_yy * phi_j[i]
                    
                    # Symmetry: swap i,j
                    A[idx_i['ux'][j], idx_i['ux'][i]] += cons_xx * phi_i[i]
                    A[idx_j['ux'][j], idx_i['ux'][i]] -= cons_xx * phi_i[i]
                    A[idx_i['ux'][j], idx_j['ux'][i]] -= cons_xx * phi_j[i]
                    A[idx_j['ux'][j], idx_j['ux'][i]] += cons_xx * phi_j[i]
                    
                    A[idx_i['uy'][j], idx_i['ux'][i]] += cons_xy * phi_i[i]
                    A[idx_j['uy'][j], idx_i['ux'][i]] -= cons_xy * phi_i[i]
                    A[idx_i['uy'][j], idx_j['ux'][i]] -= cons_xy * phi_j[i]
                    A[idx_j['uy'][j], idx_j['ux'][i]] += cons_xy * phi_j[i]
                    
                    A[idx_i['ux'][j], idx_i['uy'][i]] += cons_yx * phi_i[i]
                    A[idx_j['ux'][j], idx_i['uy'][i]] -= cons_yx * phi_i[i]
                    A[idx_i['ux'][j], idx_j['uy'][i]] -= cons_yx * phi_j[i]
                    A[idx_j['ux'][j], idx_j['uy'][i]] += cons_yx * phi_j[i]
                    
                    A[idx_i['uy'][j], idx_i['uy'][i]] += cons_yy * phi_i[i]
                    A[idx_j['uy'][j], idx_i['uy'][i]] -= cons_yy * phi_i[i]
                    A[idx_i['uy'][j], idx_j['uy'][i]] -= cons_yy * phi_j[i]
                    A[idx_j['uy'][j], idx_j['uy'][i]] += cons_yy * phi_j[i]

    def _assemble_dirichlet_face_quad(self, A, b, edge_id, cell_i, h_e, bc):
        """Dirichlet BC with 2-point Gauss quadrature"""
        idx_i = self.get_indices(cell_i)
        n = self.mesh.edge_normal(edge_id, cell_i)
        
        quad_pts, quad_wts = self.get_edge_quadrature(edge_id, order=2)
        sigma = self.penalty / h_e
        
        for qp, qw in zip(quad_pts, quad_wts):
            phi_i, gx_i, gy_i = self.evaluate_basis(cell_i, qp, derivatives=True)
            
            bc_val = bc.evaluate(qp[0], qp[1])
            if bc.is_vector:
                g_x, g_y = bc_val
            else:
                g_x = bc_val
                g_y = 0.0
            
            for i in range(3):
                # Penalty RHS
                b[idx_i['ux'][i]] += sigma * qw * phi_i[i] * g_x
                b[idx_i['uy'][i]] += sigma * qw * phi_i[i] * g_y
                
                for j in range(3):
                    # Penalty matrix
                    pen_term = sigma * qw * phi_i[i] * phi_i[j]
                    A[idx_i['ux'][i], idx_i['ux'][j]] += pen_term
                    A[idx_i['uy'][i], idx_i['uy'][j]] += pen_term
                    
                    # Consistency terms
                    grad_ux = (gx_i[j], gy_i[j])
                    grad_uy = (0.0, 0.0)
                    sigma_ux_x, sigma_ux_y = self.compute_stress(grad_ux, grad_uy, n)
                    
                    grad_ux_y = (0.0, 0.0)
                    grad_uy_y = (gx_i[j], gy_i[j])
                    sigma_uy_x, sigma_uy_y = self.compute_stress(grad_ux_y, grad_uy_y, n)
                    
                    cons_xx = -qw * sigma_ux_x * phi_i[i]
                    cons_xy = -qw * sigma_uy_x * phi_i[i]
                    cons_yx = -qw * sigma_ux_y * phi_i[i]
                    cons_yy = -qw * sigma_uy_y * phi_i[i]
                    
                    A[idx_i['ux'][i], idx_i['ux'][j]] += cons_xx
                    A[idx_i['ux'][i], idx_i['uy'][j]] += cons_xy
                    A[idx_i['uy'][i], idx_i['ux'][j]] += cons_yx
                    A[idx_i['uy'][i], idx_i['uy'][j]] += cons_yy
                    
                    # Symmetry (goes to RHS)
                    grad_vx = (gx_i[i], gy_i[i])
                    grad_vy = (0.0, 0.0)
                    sigma_vx_x, sigma_vx_y = self.compute_stress(grad_vx, grad_vy, n)
                    
                    grad_vx_y = (0.0, 0.0)
                    grad_vy_y = (gx_i[i], gy_i[i])
                    sigma_vy_x, sigma_vy_y = self.compute_stress(grad_vx_y, grad_vy_y, n)
                    
                    b[idx_i['ux'][i]] -= (-qw * sigma_vx_x * phi_i[j]) * g_x
                    b[idx_i['ux'][i]] -= (-qw * sigma_vx_y * phi_i[j]) * g_y
                    b[idx_i['uy'][i]] -= (-qw * sigma_vy_x * phi_i[j]) * g_x
                    b[idx_i['uy'][i]] -= (-qw * sigma_vy_y * phi_i[j]) * g_y

    def _assemble_neumann_face_quad(self, A, b, edge_id, cell_i, h_e, bc):
        """Neumann BC with 2-point Gauss quadrature"""
        idx_i = self.get_indices(cell_i)
        
        quad_pts, quad_wts = self.get_edge_quadrature(edge_id, order=2)
        
        for qp, qw in zip(quad_pts, quad_wts):
            phi_i = self.evaluate_basis(cell_i, qp, derivatives=False)
            
            bc_val = bc.evaluate(qp[0], qp[1])
            if bc.is_vector:
                t_x, t_y = bc_val
            else:
                t_x = bc_val
                t_y = 0.0
            
            for i in range(3):
                b[idx_i['ux'][i]] += t_x * qw * phi_i[i]
                b[idx_i['uy'][i]] += t_y * qw * phi_i[i]

    def solve(self, f_func):
        """Solve the system"""
        A, b = self.assemble_system(f_func)
        A_csr = A.tocsr()
        
        print(f"\nMatrix diagnostics:")
        print(f"  Shape: {A_csr.shape}")
        print(f"  Nonzeros: {A_csr.nnz}")
        print(f"  Sparsity: {100 * (1 - A_csr.nnz / (A_csr.shape[0] * A_csr.shape[1])):.1f}%")
        
        u_dofs = spsolve(A_csr, b)
        residual = np.linalg.norm(A_csr @ u_dofs - b)
        print(f"  Solution residual: ||A*u - b||_2 = {residual:.6e}")
        
        return u_dofs
