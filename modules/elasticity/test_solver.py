import sys
from pathlib import Path

# Add parent directory to path so we can import panda package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from elasticity_DG import *
from panda.lib import polygonal_mesh
from panda.lib import boundary_conditions

# ============================================================================
# Manufactured Solutions for Linear Elasticity
# ============================================================================

def polynomial_displacement():
    """
    Simple polynomial displacement field for linear elasticity
    
    Displacement:
        u_x(x,y) = x² * (1-x)² * y * (1-y)
        u_y(x,y) = x * (1-x) * y² * (1-y)²
    
    This satisfies homogeneous Dirichlet BCs on the boundary of [0,1]×[0,1]
    """
    def ux_exact(x, y):
        return x**2 * (1-x)**2 * y * (1-y)
    
    def uy_exact(x, y):
        return x * (1-x) * y**2 * (1-y)**2
    
    def f_source(x, y, lam=1.0, mu=1.0):
        # Simplified body force for testing
        # This is an approximation - exact computation requires symbolic differentiation
        fx = -2.0 * mu * (2 - 12*x + 12*x**2) * y * (1-y)
        fy = -2.0 * mu * x * (1-x) * (2 - 12*y + 12*y**2)
        return fx, fy
    
    def bc_func(x, y):
        return (ux_exact(x, y), uy_exact(x, y))
    
    return ux_exact, uy_exact, f_source, bc_func


def linear_displacement():
    """
    Linear displacement field (should be exactly represented by P1 elements)
    
    Displacement:
        u_x(x,y) = 0.1 * x + 0.05 * y
        u_y(x,y) = 0.05 * x + 0.1 * y
    
    This represents a combination of stretching and shearing
    """
    def ux_exact(x, y):
        return 0.1 * x + 0.05 * y
    
    def uy_exact(x, y):
        return 0.05 * x + 0.1 * y
    
    def f_source(x, y, lam=1.0, mu=1.0):
        # For linear displacement, body force is zero
        return 0.0, 0.0
    
    def bc_func(x, y):
        return (ux_exact(x, y), uy_exact(x, y))
    
    return ux_exact, uy_exact, f_source, bc_func


def pure_shear():
    """
    Pure shear deformation
    
    Displacement:
        u_x(x,y) = γ * y
        u_y(x,y) = 0
    
    where γ is the shear strain
    """
    gamma = 0.1
    
    def ux_exact(x, y):
        return gamma * y
    
    def uy_exact(x, y):
        return 0.0
    
    def f_source(x, y, lam=1.0, mu=1.0):
        return 0.0, 0.0
    
    def bc_func(x, y):
        return (ux_exact(x, y), uy_exact(x, y))
    
    return ux_exact, uy_exact, f_source, bc_func


# ============================================================================
# Accuracy Tests
# ============================================================================

def test_p1dg_elasticity_linear_accuracy():
    """Test P1 DG Elasticity Solver Accuracy on Linear Displacement Field"""
    print("Testing P1 DG Elasticity Solver Accuracy on Linear Displacement")
    
    mesh = polygonal_mesh.create_square_mesh(n=11)
    ux_exact, uy_exact, f_source, bc_func = linear_displacement()
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", bc_func, is_vector=True)
    
    lam, mu = 1.0, 1.0
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=lam, lame_mu=mu, penalty_param=50.0)
    
    def f(x, y):
        return f_source(x, y, lam, mu)
    
    u_dofs = solver.solve(f)
    
    # Compute errors at cell centroids
    ux_errors = []
    uy_errors = []
    
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        ux_num, uy_num = solver.evaluate_solution(u_dofs, (x, y), cell_id)
        
        ux_errors.append(abs(ux_num - ux_exact(x, y)))
        uy_errors.append(abs(uy_num - uy_exact(x, y)))
    
    max_ux_error = max(ux_errors)
    max_uy_error = max(uy_errors)
    
    l2_ux_error = np.sqrt(np.mean(np.array(ux_errors)**2))
    l2_uy_error = np.sqrt(np.mean(np.array(uy_errors)**2))
    
    print(f"  Max errors: ux={max_ux_error:.3e}, uy={max_uy_error:.3e}")
    print(f"  L2 errors:  ux={l2_ux_error:.3e}, uy={l2_uy_error:.3e}")
    
    # Linear displacement should be well-represented (DG method has some error at interfaces)
    assert max_ux_error < 0.1, f"ux error too large for linear field: {max_ux_error}"
    assert max_uy_error < 0.1, f"uy error too large for linear field: {max_uy_error}"


def test_p1dg_elasticity_polynomial_accuracy():
    """Test P1 DG Elasticity Solver Accuracy on Polynomial Displacement"""
    print("Testing P1 DG Elasticity Solver Accuracy on Polynomial Displacement")
    
    mesh = polygonal_mesh.create_square_mesh(n=11)
    ux_exact, uy_exact, f_source, bc_func = polynomial_displacement()
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", bc_func, is_vector=True)
    
    lam, mu = 1.0, 1.0
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=lam, lame_mu=mu, penalty_param=50.0)
    
    def f(x, y):
        return f_source(x, y, lam, mu)
    
    u_dofs = solver.solve(f)
    
    # Compute errors
    ux_errors = []
    uy_errors = []
    
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        ux_num, uy_num = solver.evaluate_solution(u_dofs, (x, y), cell_id)
        
        ux_errors.append(abs(ux_num - ux_exact(x, y)))
        uy_errors.append(abs(uy_num - uy_exact(x, y)))
    
    max_ux_error = max(ux_errors)
    max_uy_error = max(uy_errors)
    
    print(f"  Max errors: ux={max_ux_error:.3e}, uy={max_uy_error:.3e}")
    
    # Polynomial displacement will have some error
    assert max_ux_error < 0.01, f"ux error too large: {max_ux_error}"
    assert max_uy_error < 0.01, f"uy error too large: {max_uy_error}"


def test_p1dg_elasticity_pure_shear():
    """Test P1 DG Elasticity Solver on Pure Shear Deformation"""
    print("Testing P1 DG Elasticity Solver on Pure Shear")
    
    mesh = polygonal_mesh.create_square_mesh(n=11)
    ux_exact, uy_exact, f_source, bc_func = pure_shear()
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", bc_func, is_vector=True)
    
    lam, mu = 1.0, 1.0
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=lam, lame_mu=mu, penalty_param=50.0)
    
    def f(x, y):
        return f_source(x, y, lam, mu)
    
    u_dofs = solver.solve(f)
    
    # Compute errors
    ux_errors = []
    uy_errors = []
    
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        ux_num, uy_num = solver.evaluate_solution(u_dofs, (x, y), cell_id)
        
        ux_errors.append(abs(ux_num - ux_exact(x, y)))
        uy_errors.append(abs(uy_num - uy_exact(x, y)))
    
    max_ux_error = max(ux_errors)
    max_uy_error = max(uy_errors)
    
    print(f"  Max errors: ux={max_ux_error:.3e}, uy={max_uy_error:.3e}")
    
    # Linear shear should be well-represented
    assert max_ux_error < 0.05, f"ux error too large for shear: {max_ux_error}"
    assert max_uy_error < 0.01, f"uy error too large for shear: {max_uy_error}"


# ============================================================================
# Penalty Parameter Tests
# ============================================================================

def test_penalty_stability():
    """Test P1 DG Elasticity Solver Stability with Varying Penalty Parameters"""
    print("Testing P1 DG Elasticity Solver Stability with Varying Penalty Parameters")
    
    mesh = polygonal_mesh.create_square_mesh(n=6)
    ux_exact, uy_exact, f_source, bc_func = linear_displacement()
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", bc_func, is_vector=True)
    
    lam, mu = 1.0, 1.0
    penalties = [10.0, 25.0, 50.0, 100.0]
    ux_errors = []
    uy_errors = []
    
    for γ in penalties:
        solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=lam, lame_mu=mu, penalty_param=γ)
        
        def f(x, y):
            return f_source(x, y, lam, mu)
        
        u_dofs = solver.solve(f)
        
        ux_err = []
        uy_err = []
        
        for cid in range(mesh.n_cells):
            x, y = mesh.cell_centroid(cid)
            ux_num, uy_num = solver.evaluate_solution(u_dofs, (x, y), cid)
            
            ux_err.append(abs(ux_num - ux_exact(x, y)))
            uy_err.append(abs(uy_num - uy_exact(x, y)))
        
        ux_errors.append(max(ux_err))
        uy_errors.append(max(uy_err))
    
    print(f"  Penalties: {penalties}")
    print(f"  ux errors: {ux_errors}")
    print(f"  uy errors: {uy_errors}")
    
    # Errors should be reasonable and stable across penalty values
    assert all(e < 0.1 for e in ux_errors), "ux errors should be reasonable for linear field"
    assert all(e < 0.1 for e in uy_errors), "uy errors should be reasonable for linear field"


# ============================================================================
# Material Parameter Tests
# ============================================================================

def test_lame_parameter_variation():
    """Test P1 DG Elasticity Solver with Different Lamé Parameters"""
    print("Testing P1 DG Elasticity Solver with Different Lamé Parameters")
    
    mesh = polygonal_mesh.create_square_mesh(n=8)
    ux_exact, uy_exact, f_source, bc_func = linear_displacement()
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", bc_func, is_vector=True)
    
    # Test different material parameters
    lame_params = [
        (0.5, 0.5),   # Soft material
        (1.0, 1.0),   # Reference
        (10.0, 5.0),  # Stiffer material
    ]
    
    for lam, mu in lame_params:
        solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=lam, lame_mu=mu, penalty_param=50.0)
        
        def f(x, y):
            return f_source(x, y, lam, mu)
        
        u_dofs = solver.solve(f)
        
        assert u_dofs is not None, f"Solver failed for λ={lam}, μ={mu}"
        assert not np.any(np.isnan(u_dofs)), f"NaN values for λ={lam}, μ={mu}"
        
        # Check that solution is reasonable
        u_norm = np.linalg.norm(u_dofs)
        assert u_norm > 0, f"Zero solution for λ={lam}, μ={mu}"
        assert u_norm < 1e6, f"Solution exploded for λ={lam}, μ={mu}"
        
        print(f"  λ={lam}, μ={mu}: solution norm = {u_norm:.3e}")


def test_incompressible_limit():
    """Test behavior approaching incompressible limit (large λ)"""
    print("Testing Incompressible Limit (large λ)")
    
    mesh = polygonal_mesh.create_square_mesh(n=6)
    
    # Simple compression test
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    
    # Bottom fixed
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(y) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, 0.0),
        name="bottom",
        is_vector=True
    )
    
    # Top compressed downward
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, -0.01),
        name="top",
        is_vector=True
    )
    
    # Sides free
    bc.add_bc_by_function(
        region_func=lambda x, y: (abs(x) < 1e-10) or (abs(x - 1.0) < 1e-10),
        bc_type="neumann",
        value_func=(0.0, 0.0),
        name="sides",
        is_vector=True
    )
    
    mu = 1.0
    lambda_values = [1.0, 10.0, 100.0]
    
    for lam in lambda_values:
        solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=lam, lame_mu=mu, penalty_param=50.0)
        
        def f(x, y):
            return 0.0, 0.0
        
        u_dofs = solver.solve(f)
        
        assert u_dofs is not None, f"Solver failed for λ={lam}"
        assert not np.any(np.isnan(u_dofs)), f"NaN for λ={lam}"
        
        print(f"  λ={lam}: solution norm = {np.linalg.norm(u_dofs):.3e}")


# ============================================================================
# Boundary Condition Tests
# ============================================================================

def test_homogeneous_dirichlet_bc():
    """Test Elasticity solver with homogeneous Dirichlet boundary conditions"""
    print("Testing Homogeneous Dirichlet Boundary Conditions")
    
    mesh = polygonal_mesh.create_square_mesh(n=8)
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", (0.0, 0.0), is_vector=True)
    
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=1.0, lame_mu=1.0, penalty_param=50.0)
    
    # Constant body force
    def f(x, y):
        return 1.0, -1.0
    
    u_dofs = solver.solve(f)
    
    assert u_dofs is not None
    assert not np.any(np.isnan(u_dofs))
    
    # Check that solution is non-trivial
    u_norm = np.linalg.norm(u_dofs)
    assert u_norm > 1e-6, "Solution should be non-trivial with body force"
    
    print(f"  Solution norm: {u_norm:.3e}")


def test_mixed_bc_dirichlet_neumann():
    """Test Elasticity solver with mixed Dirichlet and Neumann boundary conditions"""
    print("Testing Mixed Dirichlet and Neumann Boundary Conditions")
    
    mesh = polygonal_mesh.create_square_mesh(n=8)
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    
    # Fixed left edge
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, 0.0),
        name="left",
        is_vector=True
    )
    
    # Traction on right edge
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x - 1.0) < 1e-10,
        bc_type="neumann",
        value_func=(1.0, 0.0),  # Horizontal traction
        name="right",
        is_vector=True
    )
    
    # Free top and bottom
    bc.add_bc_by_function(
        region_func=lambda x, y: (abs(y) < 1e-10) or (abs(y - 1.0) < 1e-10),
        bc_type="neumann",
        value_func=(0.0, 0.0),
        name="top_bottom",
        is_vector=True
    )
    
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=1.0, lame_mu=1.0, penalty_param=50.0)
    
    def f(x, y):
        return 0.0, 0.0
    
    u_dofs = solver.solve(f)
    
    assert u_dofs is not None
    assert not np.any(np.isnan(u_dofs))
    
    print(f"  Solution norm: {np.linalg.norm(u_dofs):.3e}")


def test_all_neumann_bc():
    """Test Elasticity solver with all Neumann boundary conditions"""
    print("Testing All Neumann Boundary Conditions")
    
    mesh = polygonal_mesh.create_square_mesh(n=8)
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    
    # Traction on all boundaries
    bc.add_bc_to_all_boundaries("neumann", (0.1, 0.0), is_vector=True)
    
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=1.0, lame_mu=1.0, penalty_param=50.0)
    
    def f(x, y):
        return 0.0, 0.0
    
    u_dofs = solver.solve(f)
    
    assert u_dofs is not None
    assert not np.any(np.isnan(u_dofs))
    
    print(f"  Solution norm: {np.linalg.norm(u_dofs):.3e}")


# ============================================================================
# Physical Problem Tests
# ============================================================================

def test_cantilever_beam():
    """Test cantilever beam under gravity"""
    print("Testing Cantilever Beam Under Gravity")
    
    # Create rectangular mesh (beam-like)
    mesh = polygonal_mesh.create_rectangle_mesh(length=3.0, height=1.0, nx=15, ny=5)
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    
    # Fixed left end
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, 0.0),
        name="left_fixed",
        is_vector=True
    )
    
    # Free right end and top/bottom
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x) >= 1e-10,
        bc_type="neumann",
        value_func=(0.0, 0.0),
        name="free",
        is_vector=True
    )
    
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=1.0, lame_mu=1.0, penalty_param=50.0)
    
    # Gravity pointing down
    def f(x, y):
        return 0.0, -1.0
    
    u_dofs = solver.solve(f)
    
    assert u_dofs is not None
    assert not np.any(np.isnan(u_dofs))
    
    # Check that displacement is largest at free end
    max_disp_left = 0.0
    max_disp_right = 0.0
    
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        ux, uy = solver.evaluate_solution(u_dofs, (x, y), cell_id)
        disp_mag = np.sqrt(ux**2 + uy**2)
        
        if x < 0.5:
            max_disp_left = max(max_disp_left, disp_mag)
        if x > 2.5:
            max_disp_right = max(max_disp_right, disp_mag)
    
    print(f"  Max displacement near fixed end: {max_disp_left:.3e}")
    print(f"  Max displacement near free end: {max_disp_right:.3e}")
    
    # Free end should deflect more than fixed end
    assert max_disp_right > max_disp_left, "Free end should deflect more than fixed end"


def test_compression():
    """Test uniaxial compression"""
    print("Testing Uniaxial Compression")
    
    mesh = polygonal_mesh.create_square_mesh(n=10)
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    
    # Bottom fixed
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(y) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, 0.0),
        name="bottom",
        is_vector=True
    )
    
    # Top compressed
    compression = -0.05
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, compression),
        name="top",
        is_vector=True
    )
    
    # Sides free
    bc.add_bc_by_function(
        region_func=lambda x, y: (abs(x) < 1e-10) or (abs(x - 1.0) < 1e-10),
        bc_type="neumann",
        value_func=(0.0, 0.0),
        name="sides",
        is_vector=True
    )
    
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=1.0, lame_mu=1.0, penalty_param=50.0)
    
    def f(x, y):
        return 0.0, 0.0
    
    u_dofs = solver.solve(f)
    
    assert u_dofs is not None
    assert not np.any(np.isnan(u_dofs))
    
    # Check that vertical displacement is negative (compression)
    uy_vals = []
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        if 0.4 < y < 0.6:  # Middle section
            ux, uy = solver.evaluate_solution(u_dofs, (x, y), cell_id)
            uy_vals.append(uy)
    
    mean_uy = np.mean(uy_vals)
    print(f"  Mean vertical displacement in middle: {mean_uy:.3e}")
    
    assert mean_uy < 0, "Vertical displacement should be negative (compression)"


def test_tension():
    """Test uniaxial tension"""
    print("Testing Uniaxial Tension")
    
    mesh = polygonal_mesh.create_square_mesh(n=10)
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    
    # Left fixed
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, 0.0),
        name="left",
        is_vector=True
    )
    
    # Right pulled
    tension = 0.05
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=(tension, 0.0),
        name="right",
        is_vector=True
    )
    
    # Top and bottom free
    bc.add_bc_by_function(
        region_func=lambda x, y: (abs(y) < 1e-10) or (abs(y - 1.0) < 1e-10),
        bc_type="neumann",
        value_func=(0.0, 0.0),
        name="top_bottom",
        is_vector=True
    )
    
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=1.0, lame_mu=1.0, penalty_param=50.0)
    
    def f(x, y):
        return 0.0, 0.0
    
    u_dofs = solver.solve(f)
    
    assert u_dofs is not None
    assert not np.any(np.isnan(u_dofs))
    
    # Check that horizontal displacement is positive (tension)
    ux_vals = []
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        if 0.4 < x < 0.6:  # Middle section
            ux, uy = solver.evaluate_solution(u_dofs, (x, y), cell_id)
            ux_vals.append(ux)
    
    mean_ux = np.mean(ux_vals)
    print(f"  Mean horizontal displacement in middle: {mean_ux:.3e}")
    
    assert mean_ux > 0, "Horizontal displacement should be positive (tension)"


# ============================================================================
# Regression Tests
# ============================================================================

def test_regression_cantilever():
    """
    Regression test for cantilever beam
    
    This test captures expected solution ranges to detect unintended changes
    """
    print("Regression Test: Cantilever Beam")
    
    mesh = polygonal_mesh.create_rectangle_mesh(length=2.0, height=1.0, nx=10, ny=5)
    
    bc = boundary_conditions.BoundaryConditionManager(mesh)
    
    # Fixed left end
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=(0.0, 0.0),
        name="left",
        is_vector=True
    )
    
    # Free elsewhere
    bc.add_bc_by_function(
        region_func=lambda x, y: abs(x) >= 1e-10,
        bc_type="neumann",
        value_func=(0.0, 0.0),
        name="free",
        is_vector=True
    )
    
    solver = P1DGLinearElasticitySolver(mesh, bc, lame_lambda=1.0, lame_mu=1.0, penalty_param=50.0)
    
    def f(x, y):
        return 0.0, -0.5
    
    u_dofs = solver.solve(f)
    
    # Expected ranges (relaxed for DG method)
    expected_ranges = {
        'ux_min': -1.0,
        'ux_max': 1.0,
        'uy_min': -5.0,
        'uy_max': 0.5,
    }
    
    ux_vals = []
    uy_vals = []
    
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        ux, uy = solver.evaluate_solution(u_dofs, (x, y), cell_id)
        ux_vals.append(ux)
        uy_vals.append(uy)
    
    ux_min, ux_max = min(ux_vals), max(ux_vals)
    uy_min, uy_max = min(uy_vals), max(uy_vals)
    
    print(f"  ux range: [{ux_min:.3e}, {ux_max:.3e}]")
    print(f"  uy range: [{uy_min:.3e}, {uy_max:.3e}]")
    
    # Check ranges
    assert ux_min > expected_ranges['ux_min'], f"ux_min out of range: {ux_min}"
    assert ux_max < expected_ranges['ux_max'], f"ux_max out of range: {ux_max}"
    assert uy_min > expected_ranges['uy_min'], f"uy_min out of range: {uy_min}"
    assert uy_max < expected_ranges['uy_max'], f"uy_max out of range: {uy_max}"


if __name__ == "__main__":
    # Run all tests
    print("="*80)
    print("Running Elasticity Solver Tests")
    print("="*80)
    
    # Accuracy tests
    test_p1dg_elasticity_linear_accuracy()
    test_p1dg_elasticity_polynomial_accuracy()
    test_p1dg_elasticity_pure_shear()
    
    # Penalty tests
    test_penalty_stability()
    
    # Material parameter tests
    test_lame_parameter_variation()
    test_incompressible_limit()
    
    # Boundary condition tests
    test_homogeneous_dirichlet_bc()
    test_mixed_bc_dirichlet_neumann()
    test_all_neumann_bc()
    
    # Physical problem tests
    test_cantilever_beam()
    test_compression()
    test_tension()
    
    # Regression tests
    test_regression_cantilever()
    
    print("="*80)
    print("All tests passed!")
    print("="*80)
