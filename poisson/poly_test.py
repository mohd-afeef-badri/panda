import numpy as np

# ==============================================================================
# Test Case 1: Boundary Layer (tanh-based)
# ==============================================================================
def test_boundary_layer():
    """
    Solution with steep boundary layer near x=0
    Gradient concentrated in thin region
    """
    epsilon = 0.05  # Controls layer thickness (smaller = steeper)
    
    def u_exact(x, y):
        return np.tanh((x - 0.1) / epsilon) * np.sin(np.pi * y)
    
    def f(x, y):
        # Computed via: f = -Δu
        tanh_term = np.tanh((x - 0.1) / epsilon)
        sech_term = 1.0 / np.cosh((x - 0.1) / epsilon)
        sin_term = np.sin(np.pi * y)
        
        # d²u/dx²
        d2udx2 = (-2.0 / epsilon**2) * sech_term**2 * tanh_term * sin_term
        
        # d²u/dy²
        d2udy2 = -np.pi**2 * tanh_term * sin_term
        
        return -(d2udx2 + d2udy2)
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "Boundary Layer (tanh)"



# ==============================================================================
# Test Case 2: Internal Layer (crossing the domain)
# ==============================================================================
def test_internal_layer():
    """
    Solution with steep internal layer along diagonal
    """
    epsilon = 0.03
    
    def u_exact(x, y):
        # Layer along line x + y = 1
        return np.tanh((x + y - 1.0) / epsilon)
    
    def f(x, y):
        sech_term = 1.0 / np.cosh((x + y - 1.0) / epsilon)
        tanh_term = np.tanh((x + y - 1.0) / epsilon)
        
        # Both d²u/dx² and d²u/dy² are the same due to symmetry
        d2u = (-2.0 / epsilon**2) * sech_term**2 * tanh_term
        
        return -2.0 * d2u  # -Δu = -(d²u/dx² + d²u/dy²)
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "Internal Layer (diagonal)"


# ==============================================================================
# Test Case 3: Corner Singularity (exponential peak)
# ==============================================================================
def test_corner_peak():
    """
    Solution with sharp peak in corner
    Very high gradients radiating from (x0, y0)
    """
    x0, y0 = 0.5, 0.5  # Peak location
    alpha = 50.0  # Controls sharpness (higher = sharper)
    
    def u_exact(x, y):
        r2 = (x - x0)**2 + (y - y0)**2
        return np.exp(-alpha * r2)
    
    def f(x, y):
        r2 = (x - x0)**2 + (y - y0)**2
        exp_term = np.exp(-alpha * r2)
        
        # -Δu = 2*alpha*(2*alpha*r² - 2)*exp(-alpha*r²)
        return -2.0 * alpha * (2.0 * alpha * r2 - 2.0) * exp_term
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "Corner Peak (exponential)"


# ==============================================================================
# Test Case 4: Multiple Peaks (Gaussian-like)
# ==============================================================================
def test_multiple_peaks():
    """
    Solution with multiple sharp peaks at different locations
    """
    alpha = 30.0
    centers = [(0.3, 0.3), (0.7, 0.7), (0.3, 0.7)]
    
    def u_exact(x, y):
        u = 0.0
        for x0, y0 in centers:
            r2 = (x - x0)**2 + (y - y0)**2
            u += np.exp(-alpha * r2)
        return u
    
    def f(x, y):
        source = 0.0
        for x0, y0 in centers:
            r2 = (x - x0)**2 + (y - y0)**2
            exp_term = np.exp(-alpha * r2)
            source += -2.0 * alpha * (2.0 * alpha * r2 - 2.0) * exp_term
        return source
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "Multiple Peaks"


# ==============================================================================
# Test Case 5: Sharp Front (propagating wave)
# ==============================================================================
def test_sharp_front():
    """
    Solution with sharp transition front
    Mimics wave propagation or phase transition
    """
    epsilon = 0.04
    
    def u_exact(x, y):
        # Front moves from left to right
        dist = x - 0.5 + 0.2 * np.sin(4 * np.pi * y)
        return 0.5 * (1.0 + np.tanh(dist / epsilon))
    
    def f(x, y):
        dist = x - 0.5 + 0.2 * np.sin(4 * np.pi * y)
        sech2 = 1.0 / np.cosh(dist / epsilon)**2
        tanh_term = np.tanh(dist / epsilon)
        
        cos_term = np.cos(4 * np.pi * y)
        sin_term = np.sin(4 * np.pi * y)
        
        # d²u/dx²
        d2udx2 = -(1.0 / epsilon**2) * sech2 * tanh_term
        
        # d²u/dy² (more complex due to sin term)
        dy_dist = 0.2 * 4 * np.pi * cos_term
        d2y_dist = -0.2 * 16 * np.pi**2 * sin_term
        
        d2udy2 = (0.5 / epsilon) * (
            -sech2 * tanh_term * dy_dist**2 +
            sech2 * d2y_dist
        )
        
        return -(d2udx2 + d2udy2)
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "Sharp Front (wave)"


# ==============================================================================
# Test Case 6: Circular Boundary Layer
# ==============================================================================
def test_circular_layer():
    """
    Solution with circular boundary layer around center
    """
    xc, yc = 0.5, 0.5
    R = 0.3  # Radius of circular layer
    epsilon = 0.02
    
    def u_exact(x, y):
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        return np.tanh((r - R) / epsilon)
    
    def f(x, y):
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        if r < 1e-10:  # Avoid division by zero at center
            return 0.0
        
        dist = (r - R) / epsilon
        sech2 = 1.0 / np.cosh(dist)**2
        tanh_term = np.tanh(dist)
        
        # Laplacian in polar coordinates, then convert back
        # Δu = (1/r) * d(r*du/dr)/dr
        dudr = sech2 / epsilon
        
        d2udr2 = (-2.0 / epsilon**2) * sech2 * tanh_term
        
        laplacian = d2udr2 + (1.0 / r) * dudr
        
        return -laplacian
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "Circular Layer"


# ==============================================================================
# Test Case 7: Extreme Corner Layer
# ==============================================================================
def test_extreme_corner():
    """
    Very steep gradients near multiple corners
    Challenges the solver significantly
    """
    epsilon = 0.02
    
    def u_exact(x, y):
        # Product of tanh functions creates steep corners
        return (np.tanh((x - 0.2) / epsilon) * 
                np.tanh((y - 0.2) / epsilon) *
                np.tanh((0.8 - x) / epsilon) * 
                np.tanh((0.8 - y) / epsilon))
    
    def f(x, y):
        # This is complex - numerical differentiation recommended
        # or use finite differences for verification
        h = 1e-6
        u_xx = (u_exact(x+h, y) - 2*u_exact(x, y) + u_exact(x-h, y)) / h**2
        u_yy = (u_exact(x, y+h) - 2*u_exact(x, y) + u_exact(x, y-h)) / h**2
        return -(u_xx + u_yy)
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "Extreme Corner Layers"

# ==============================================================================
# Test Case 8: Smooth sin cos
# ==============================================================================
def smooth_sin_cos():
    """
    Define problem: -Δu = 2π²sin(πx)sin(πy) with u=0 on boundary
    Exact solution: u = sin(πx)sin(πy)
    """
    epsilon = 0.02
    
    def u_exact(x, y):
        return np.sin(np.pi*x) * np.sin(np.pi*y)
    
    def f(x, y):
        return 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
    
    def g(x, y):
        return u_exact(x, y)
    
    return u_exact, f, g, "smooth sin cos"