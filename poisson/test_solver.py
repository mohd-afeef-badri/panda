import numpy as np
from DG_P1 import *
import manufactured_solutions as manufactured_solutions
from med_io import *

def test_p1dg_poisson_accuracy():
    print("Testing P1 DG Poisson Solver Accuracy on Boundary Layer Problem")
    mesh = create_square_mesh(n=51)
    u_exact, f, g, _ = manufactured_solutions.boundary_layer()

    bc = BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", g)

    solver = P1DGPoissonSolver(mesh, bc, penalty_param=10.0)
    u_dofs = solver.solve(f)

    errors = []
    for cell_id in range(mesh.n_cells):
        x, y = mesh.cell_centroid(cell_id)
        u_num = solver.evaluate_solution(u_dofs, (x, y), cell_id)
        errors.append(abs(u_num - u_exact(x, y)))

    max_error = max(errors)
    l2_error = np.sqrt(np.mean(np.array(errors)**2))

    assert max_error < 2e-2
    assert l2_error < 1e-2

def test_convergence_rate():
    print("Testing P1 DG Poisson Solver Convergence Rate on Smooth Problem")
    hs = []
    errors = []

    for n in [4, 8, 16]:
        mesh = create_square_mesh(n=n)
        u_exact, f, g, _ = manufactured_solutions.smooth_sin_cos()

        bc = BoundaryConditionManager(mesh)
        bc.add_bc_to_all_boundaries("dirichlet", g)

        solver = P1DGPoissonSolver(mesh, bc, penalty_param=10)
        u = solver.solve(f)

        err = []
        for cid in range(mesh.n_cells):
            x, y = mesh.cell_centroid(cid)
            err.append((solver.evaluate_solution(u, (x,y), cid)
                        - u_exact(x,y))**2)

        errors.append(np.sqrt(np.mean(err)))
        hs.append(1.0 / n)

    rate = np.log(errors[0]/errors[-1]) / np.log(hs[0]/hs[-1])
    assert rate > 1.7

def test_penalty_stability():
    print("Testing P1 DG Poisson Solver Stability with Varying Penalty Parameters")
    mesh = create_square_mesh(n=5)
    u_exact, f, g, _ = manufactured_solutions.smooth_sin_cos()

    bc = BoundaryConditionManager(mesh)
    bc.add_bc_to_all_boundaries("dirichlet", g)

    penalties = [2.0, 5.0, 10.0]
    errors = []

    for γ in penalties:
        solver = P1DGPoissonSolver(mesh, bc, penalty_param=γ)
        u = solver.solve(f)

        e = []
        for cid in range(mesh.n_cells):
            x, y = mesh.cell_centroid(cid)
            e.append(abs(solver.evaluate_solution(u, (x,y), cid)
                         - u_exact(x,y)))
        errors.append(max(e))

    # Regression condition: no penalty blows up
    assert max(errors) / min(errors) < 3.0

def test_all_boundaries_simple():
    """
    Example: Apply the same BC to ALL boundaries
    Simplest case - homogeneous Dirichlet everywhere
    
    This test verifies that the solver produces consistent results
    for a simple test case with homogeneous Dirichlet BCs on all boundaries.
    The test checks that solution values at specific locations match expected values
    within a tolerance to detect regressions.
    """
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Apply Dirichlet BC u=0 to ALL boundaries
    bc_manager.add_bc_to_all_boundaries("dirichlet", 0.0)
    
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: 1.0
    u_dofs = solver.solve(f)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured on 2026-01-21 with penalty_param=10.0
    expected_values = {
        0: 7.427013e-04,    # Cell 0 centroid
        1: 7.812928e-04,    # Cell 1 centroid
        2: 7.410023e-04,    # Cell 2 centroid
        3: 7.625856e-04,    # Cell 3 centroid
        4: 7.386026e-04,    # Cell 4 centroid
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check global statistics
    assert np.min(u_dofs) > -0.4, "Minimum solution value out of expected range"
    assert np.max(u_dofs) < 0.4, "Maximum solution value out of expected range"


def test_all_boundaries_function():
    """
    Example: Apply a function-based BC to all boundaries
    
    Test case with spatially-varying Dirichlet BC: u = x^2 + y^2 on all boundaries.
    Source term: f = -4.0 (Laplacian of x^2 + y^2 is 4, so RHS is -4).
    
    This test verifies that the solver correctly handles function-based BCs
    and produces consistent results across the domain.
    """
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Apply spatially-varying Dirichlet BC to all boundaries
    bc_manager.add_bc_to_all_boundaries(
        bc_type="dirichlet",
        value_func=lambda x, y: x**2 + y**2
    )
    
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: -4.0  # Laplacian of x^2 + y^2
    u_dofs = solver.solve(f)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured on 2026-01-21 with penalty_param=10.0
    expected_values = {
        0: 1.073821e-02,    # Cell 0 centroid
        1: 8.044295e-01,    # Cell 1 centroid
        2: 1.000862e+00,    # Cell 2 centroid
        3: 1.794314e+00,    # Cell 3 centroid
        4: 1.794513e+00,    # Cell 4 centroid
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check global statistics
    assert np.min(u_dofs) > 0.0, "Minimum solution value out of expected range"
    assert np.max(u_dofs) < 2.1, "Maximum solution value out of expected range"


def test_global_with_override():
    """
    Example: Set a global BC, then override specific boundaries
    Priority: specific > global
    
    This test verifies that boundary conditions can be set globally and then
    overridden for specific boundary groups, with the overrides taking precedence.
    Uses a meaningful source term f = x + y to test non-trivial behavior.
    """
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    edge_groups = extract_edge_groups_from_med("./mesh/mesh.med")
    
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # Set default: all boundaries have Dirichlet u=0
    bc_manager.add_bc_to_all_boundaries("dirichlet", 0.0)
    
    # Override specific boundaries
    # "boundary" group contains all boundary edges
    bc_manager.add_bc_by_group("boundary", "dirichlet", 1.0)
    
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    # Use a meaningful source term: f = x + y
    f = lambda x, y: x + y
    u_dofs = solver.solve(f)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured on 2026-01-21 with penalty_param=10.0 and f=x+y
    expected_values = {
        0: 1.000329e+00,    # Cell 0 centroid
        1: 1.000759e+00,    # Cell 1 centroid
        2: 1.000762e+00,    # Cell 2 centroid
        3: 1.001188e+00,    # Cell 3 centroid
        4: 1.001150e+00,    # Cell 4 centroid
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check global statistics
    assert np.min(u_dofs) > -0.5, "Minimum solution value out of expected range"
    assert np.max(u_dofs) < 1.2, "Maximum solution value out of expected range"


def test_global_with_analytical_override():
    """
    Example: Global BC with analytical function overrides
    
    This test verifies that a global Neumann BC can be overridden by analytical
    function-based Dirichlet BCs on specific boundaries (left wall in this case).
    Uses a meaningful source term f = sin(π*x)*cos(π*y).
    """
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Default: Neumann BC (no flux) everywhere
    bc_manager.add_bc_to_all_boundaries("neumann", 0.0)
    
    # Override: Dirichlet on left wall only (x ≈ 0)
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=1.0,
        name="left_inlet"
    )
    
    # All other boundaries keep the Neumann BC
    
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    # Use a meaningful source term
    f = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
    u_dofs = solver.solve(f)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured on 2026-01-21 with penalty_param=10.0
    expected_values = {
        0: 1.024934e+00,    # Cell 0 centroid
        1: 1.053037e+00,    # Cell 1 centroid
        2: 1.047975e+00,    # Cell 2 centroid
        3: 9.520214e-01,    # Cell 3 centroid
        4: 9.469653e-01,    # Cell 4 centroid
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check global statistics
    assert np.min(u_dofs) > -0.4, "Minimum solution value out of expected range"
    assert np.max(u_dofs) < 1.2, "Maximum solution value out of expected range"


def test_insulated_boundaries():
    """
    Example: Thermal problem with insulated (Neumann) boundaries by default
    
    This test models a thermal problem where most boundaries are insulated (zero heat flux),
    but specific boundaries (inlet and outlet) have fixed temperatures.
    Simulates heat conduction from a hot inlet (100°) to a cold outlet (20°)
    with insulated top and bottom boundaries.
    """
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Most boundaries are insulated (zero heat flux)
    bc_manager.add_bc_to_all_boundaries("neumann", 0.0)
    
    # Specify temperature only at inlet and outlet
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=100.0,  # Hot inlet
        name="hot_inlet"
    )
    
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=20.0,  # Cold outlet
        name="cold_outlet"
    )
    
    # Top and bottom are insulated (use global Neumann BC)
    
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: 0.0  # No internal heat source
    u_dofs = solver.solve(f)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured on 2026-01-21 with penalty_param=10.0
    # Solution should show temperature gradient from hot inlet (100) to cold outlet (20)
    expected_values = {
        0: 8.834902e+01,    # Cell 0 centroid (near hot inlet)
        1: 3.164885e+01,    # Cell 1 centroid (moving toward outlet)
        2: 2.174169e+01,    # Cell 2 centroid (near cold outlet)
        3: 2.177707e+01,    # Cell 3 centroid (near cold outlet)
        4: 3.164952e+01,    # Cell 4 centroid (moving toward outlet)
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check global statistics
    assert np.min(u_dofs) > -150.0, "Minimum temperature out of expected range"
    assert np.max(u_dofs) < 150.0, "Maximum temperature out of expected range"
    # Solution should show temperature between hot inlet (100) and cold outlet (20)
    assert np.mean(u_dofs) < 50.0, "Mean temperature should be influenced by boundary temps"


def test_priority_demonstration():
    """
    Demonstrate the priority system for boundary conditions:
    1. MED groups (highest priority)
    2. Analytical functions
    3. Global boundary BC
    4. Default (lowest priority)
    
    This test verifies that when multiple BCs are defined on overlapping regions,
    the higher priority BC takes precedence.
    """
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    edge_groups = extract_edge_groups_from_med("./mesh/mesh.med")
    
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # Priority 4 (lowest): Default is homogeneous Dirichlet
    # This is built-in, no need to set
    
    # Priority 3: Set global BC (Neumann on all boundaries)
    bc_manager.add_bc_to_all_boundaries("neumann", 0.5)
    
    # Priority 2: Analytical function (Dirichlet on top wall y=1)
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=10.0,
        name="top_wall"
    )
    
    # Priority 1 (highest): MED group (Dirichlet on "boundary" group)
    bc_manager.add_bc_by_group("boundary", "dirichlet", 100.0)
    
    # Result:
    # - "boundary" group edges: u=100 (Dirichlet) [Priority 1 - MED groups]
    # - Top wall (y=1): u=10 (Dirichlet) [Priority 2 - Analytical]
    # - All other boundaries: ∂u/∂n=0.5 (Neumann) [Priority 3 - Global]
    
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: 0.0
    u_dofs = solver.solve(f)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured on 2026-01-21 with penalty_param=10.0
    # Since "boundary" group (all 256 boundary edges) has u=100 and f=0,
    # the solution should be u=100 everywhere
    expected_values = {
        0: 1.000000e+02,    # Cell 0 centroid
        1: 1.000000e+02,    # Cell 1 centroid
        2: 1.000000e+02,    # Cell 2 centroid
        3: 1.000000e+02,    # Cell 3 centroid
        4: 1.000000e+02,    # Cell 4 centroid
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check that the mean solution value is close to the expected BC value
    # In DG methods, DOF values can vary widely, so we check evaluated solution instead
    evaluated_solutions = []
    for cell_id in range(min(20, solver.mesh.n_cells)):
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        evaluated_solutions.append(u_val)
    
    mean_solution = np.mean(evaluated_solutions)
    assert mean_solution > 99.0, f"Mean solution should be close to 100, got {mean_solution:.2f}"


def test_analytical_boundaries():
    """
    Example using analytical functions to define boundaries
    No MED groups required!
    
    This test defines boundaries analytically based on coordinate conditions:
    - Left boundary (x≈0): Dirichlet u=1
    - Right boundary (x≈1): Dirichlet u=0
    - Bottom boundary (y≈0): Dirichlet u=0
    - Top boundary (y≈1): Neumann ∂u/∂n=1
    
    Source term: f = -1.0
    """
    
    # Load mesh (no need for edge groups)
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    
    # Create BC manager without edge groups
    bc_manager = BoundaryConditionManager(mesh)
    
    # Define boundaries analytically
    # Assume domain is [0,1] x [0,1]
    
    # Left boundary: x ≈ 0
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x) < 1e-10,
        bc_type="dirichlet",
        value_func=1.0,
        name="left_wall"
    )
    
    # Right boundary: x ≈ 1
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(x - 1.0) < 1e-10,
        bc_type="dirichlet",
        value_func=0.0,
        name="right_wall"
    )
    
    # Bottom boundary: y ≈ 0
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y) < 1e-10,
        bc_type="dirichlet",
        value_func=0.0,
        name="bottom_wall"
    )
    
    # Top boundary: y ≈ 1
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,
        bc_type="neumann",
        value_func=1.0,
        name="top_wall"
    )
    
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: -1.0
    u_dofs = solver.solve(f)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured on 2026-01-21 with penalty_param=10.0
    expected_values = {
        0: 1.217212e-02,    # Cell 0 centroid (near left wall, low y)
        1: -5.738125e-05,   # Cell 1 centroid (near right wall, low y)
        2: -6.606651e-05,   # Cell 2 centroid (near right wall)
        3: 2.955387e-03,    # Cell 3 centroid (near right wall, high y)
        4: 5.122327e-02,    # Cell 4 centroid (near right wall, top)
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check global statistics
    # Solution should be influenced by left BC (u=1) and right BC (u=0)
    assert np.min(u_dofs) < 0.0, "Some DOF values should be negative due to source term"
    assert np.max(u_dofs) > 0.0, "Some DOF values should be positive"


def test_med_groups():
    """
    Example using boundary conditions from MED file groups
    
    This test demonstrates using pre-defined boundary groups from a MED file.
    The square_poly.med mesh has groups: left, right, top, bottom
    
    Boundary conditions:
    - left group: Dirichlet u=1
    - right group: Dirichlet u=0
    - top group: Neumann ∂u/∂n=0.5
    - bottom group: Dirichlet u=sin(π*x)
    
    Source term: f = 2π²*sin(π*x)*sin(π*y)
    """
    
    # 1. Load mesh with groups
    mesh = load_med_mesh_mc("./mesh/square_poly.med")
    
    # 2. Extract edge groups from MED file
    edge_groups = extract_edge_groups_from_med("./mesh/square_poly.med")
    
    # 3. Create boundary condition manager
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # 4. Define boundary conditions for each group using add_bc_by_group
    bc_manager.add_bc_by_group("left", "dirichlet", 1.0)
    bc_manager.add_bc_by_group("right", "dirichlet", 0.0)
    bc_manager.add_bc_by_group("top", "neumann", 0.5)
    bc_manager.add_bc_by_group("bottom", "dirichlet", lambda x, y: np.sin(np.pi * x))
    
    # 5. Create solver and solve
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    
    def source_term(x, y):
        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    u_dofs = solver.solve(source_term)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured with correct named groups (left, right, top, bottom)
    expected_values = {
        0: 5.022583e-01,    # Cell 0 centroid
        1: 3.219115e-03,    # Cell 1 centroid
        2: 1.039536e-01,    # Cell 2 centroid
        3: 8.198034e-02,    # Cell 3 centroid
        4: 6.903238e-02,    # Cell 4 centroid
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check global statistics
    assert np.min(u_dofs) < 0.0, "Solution should have some negative values"
    assert np.max(u_dofs) > 0.0, "Solution should have some positive values"

def test_mixed_mode_bc_groups_analytical():
    """
    Test combining MED groups and analytical boundaries with priority system
    
    Priority: MED groups (highest) > Analytical functions > Default
    
    Boundary conditions:
    - left group: Dirichlet u=4*y*(1-y) (parabolic profile)
    - right group: Neumann ∂u/∂n=0 (insulated)
    - bottom: Dirichlet u=0 (via analytical function)
    - top: Dirichlet u=0 (via analytical function)
    
    Source term: f=0 (Laplace equation)
    
    This demonstrates the priority system where MED groups take precedence
    over analytical function boundaries.
    """
    
    # 1. Load mesh and groups
    mesh = load_med_mesh_mc("./mesh/square_poly.med")
    edge_groups = extract_edge_groups_from_med("./mesh/square_poly.med")
    
    # 2. Create boundary condition manager
    bc_manager = BoundaryConditionManager(mesh, edge_groups)
    
    # 3. MED group boundaries (highest priority)
    bc_manager.add_bc_by_group("left", "dirichlet", lambda x, y: 4*y*(1-y))
    bc_manager.add_bc_by_group("right", "neumann", 0.0)
    
    # 4. Analytical function boundaries (only apply where MED groups don't cover)
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y) < 1e-10,  # Bottom boundary
        bc_type="dirichlet",
        value_func=0.0,
        name="bottom"
    )
    
    bc_manager.add_bc_by_function(
        region_func=lambda x, y: abs(y - 1.0) < 1e-10,  # Top boundary
        bc_type="dirichlet",
        value_func=0.0,
        name="top"
    )
    
    # 5. Create solver and solve
    solver = P1DGPoissonSolver(mesh, bc_manager, penalty_param=10.0)
    
    def source_term(x, y):
        return 0.0  # Laplace equation
    
    u_dofs = solver.solve(source_term)
    
    # Regression test: verify solution values at specific locations
    # Expected values captured with mixed mode BCs (MED groups + analytical)
    expected_values = {
        0: 6.851020e-02,    # Cell 0 centroid (near parabolic left wall)
        1: 5.287351e-03,    # Cell 1 centroid (near insulated right wall)
        2: 3.482101e-02,    # Cell 2 centroid (upper right)
        3: 2.935371e-02,    # Cell 3 centroid (near parabolic left wall)
        4: 2.351983e-02,    # Cell 4 centroid
    }
    
    tolerance = 1e-6  # Relative tolerance
    
    for cell_id, expected_val in expected_values.items():
        cent = solver.mesh.cell_centroid(cell_id)
        u_val = solver.evaluate_solution(u_dofs, cent, cell_id)
        rel_error = abs(u_val - expected_val) / (abs(expected_val) + 1e-15)
        assert rel_error < tolerance, \
            f"Cell {cell_id}: expected {expected_val:.6e}, got {u_val:.6e}, rel_error={rel_error:.6e}"
    
    # Check that solution respects boundary conditions
    # Left wall should have higher values (parabolic profile max at y=0.5)
    assert np.max(u_dofs) > 3.0, "Solution should have significant positive values from left BC"
    # Right wall is insulated, bottom/top are zero, so we expect reasonable range
    assert np.min(u_dofs) < 0.0, "Solution should have negative values (extension from boundaries)"
