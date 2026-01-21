

## EXAMPLE USAGE - ANALYTICAL FUNCTIONS

```python



def example_circular_domain():
    """
    Example with circular boundary detection
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/circle.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Circular boundary: r ≈ 1 (centered at origin)
    def is_on_circle(x, y, radius=1.0, center=(0.0, 0.0), tol=0.05):
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return abs(r - radius) < tol
    
    bc_manager.add_bc_by_function(
        region_func=is_on_circle,
        bc_type="dirichlet",
        value_func=lambda x, y: x**2 + y**2,
        name="circle_boundary"
    )
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: -4.0  # Laplacian of x^2 + y^2
    u_dofs = solver.solve(f)
    
    return u_dofs
```

## ADVANCED EXAMPLES

```python
def example_complex_analytical():
    """
    Complex analytical boundary detection examples
    """
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Example 1: Parabolic boundary
    def parabolic_boundary(x, y):
        return abs(y - x**2) < 0.01
    
    bc_manager.add_bc_by_function(
        region_func=parabolic_boundary,
        bc_type="dirichlet",
        value_func=100.0,
        name="parabola"
    )
    
    # Example 2: Multiple line segments
    def left_or_bottom(x, y):
        return (abs(x) < 1e-10) or (abs(y) < 1e-10)
    
    bc_manager.add_bc_by_function(
        region_func=left_or_bottom,
        bc_type="dirichlet",
        value_func=0.0,
        name="left_or_bottom"
    )
    
    # Example 3: Angular sector
    def angular_sector(x, y):
        angle = np.arctan2(y, x)
        return (angle > 0) and (angle < np.pi/4)
    
    bc_manager.add_bc_by_function(
        region_func=angular_sector,
        bc_type="neumann",
        value_func=lambda x, y: np.sin(5*x),
        name="sector"
    )
    
    # Example 4: Distance-based
    def near_point(x, y, px=0.5, py=0.5, radius=0.1):
        dist = np.sqrt((x-px)**2 + (y-py)**2)
        return abs(dist - radius) < 0.01
    
    bc_manager.add_bc_by_function(
        region_func=near_point,
        bc_type="dirichlet",
        value_func=50.0,
        name="circle_around_point"
    )
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager, penalty_param=10.0)
    f = lambda x, y: 1.0
    u_dofs = solver.solve(f)
    
    return u_dofs
```

## EXAMPLES WITH HELPER FUNCTIONS FOR COMMON BOUNDARIES

```python
def create_rectangle_bc(xmin, xmax, ymin, ymax, tol=1e-10):
    """
    Helper to create boundary detection functions for a rectangle
    
    Returns:
    --------
    dict with keys 'left', 'right', 'bottom', 'top'
    """
    return {
        'left': lambda x, y: abs(x - xmin) < tol,
        'right': lambda x, y: abs(x - xmax) < tol,
        'bottom': lambda x, y: abs(y - ymin) < tol,
        'top': lambda x, y: abs(y - ymax) < tol,
    }

def create_circle_bc(center, radius, tol=0.01):
    """Helper to create circular boundary detection function"""
    cx, cy = center
    return lambda x, y: abs(np.sqrt((x-cx)**2 + (y-cy)**2) - radius) < tol


def example_using_helpers():
    """Example using helper functions"""
    from med_reader import load_med_mesh_mc
    from bc_handler import BoundaryConditionManager, P1DGPoissonSolverWithBC
    
    mesh = load_med_mesh_mc("./mesh/mesh.med")
    bc_manager = BoundaryConditionManager(mesh)
    
    # Create rectangle boundaries
    rect_funcs = create_rectangle_bc(0.0, 1.0, 0.0, 1.0)
    
    bc_manager.add_bc_by_function(rect_funcs['left'], "dirichlet", 1.0, "left")
    bc_manager.add_bc_by_function(rect_funcs['right'], "dirichlet", 0.0, "right")
    bc_manager.add_bc_by_function(rect_funcs['bottom'], "dirichlet", 0.0, "bottom")
    bc_manager.add_bc_by_function(rect_funcs['top'], "neumann", 0.0, "top")
    
    solver = P1DGPoissonSolverWithBC(mesh, bc_manager)
    f = lambda x, y: 1.0
    u_dofs = solver.solve(f)
    
    return u_dofs
```