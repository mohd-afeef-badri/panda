"""Boundary condition management utilities."""


class BoundaryCondition:
    """Base class for boundary conditions"""
    def __init__(self, bc_type, value_func, is_vector=False):
        """
        Parameters:
        -----------
        bc_type : str
            'dirichlet' or 'neumann'
        value_func : callable
            Function f(x, y) that returns the BC value at point (x, y)
            For scalars: returns float
            For vectors: returns tuple (val_x, val_y)
        is_vector : bool
            True if this BC returns vector values (for Stokes velocity)
        """
        self.bc_type = bc_type.lower()
        self.value_func = value_func
        self.is_vector = is_vector
    
    def evaluate(self, *coordinates):
        """Evaluate the boundary condition at a 2D or 3D point."""
        return self.value_func(*coordinates)


class BoundaryConditionManager:
    """Manage boundary conditions on 2D edges or 3D faces."""
    def __init__(self, mesh, edge_groups=None, face_groups=None):
        """
        Parameters:
        -----------
        mesh : PolygonalMesh
            The mesh object
        edge_groups : dict, optional
            Dictionary mapping group names to arrays of boundary edge indices
            Format: {'group_name': np.array([edge_indices...])}
            If None, only analytical boundary detection will be used
        """
        self.mesh = mesh
        if edge_groups is not None and face_groups is not None:
            raise ValueError("Pass either edge_groups or face_groups, not both")
        self.edge_groups = (
            face_groups if face_groups is not None
            else edge_groups if edge_groups is not None
            else {}
        )
        self.conditions = {}
        self.analytical_conditions = []
        self.global_boundary_condition = None  # For all boundaries
        self.default_dirichlet = BoundaryCondition('dirichlet', lambda *coordinates: 0.0)
        
        # Map each boundary edge to its group
        self.edge_to_group = {}
        for group_name, edge_indices in self.edge_groups.items():
            for edge_idx in edge_indices:
                self.edge_to_group[edge_idx] = group_name

    def add_bc_by_group(self, group_name, bc_type, value_func, is_vector=False):
       """
       Add a boundary condition to a specific edge group from MED file

       Parameters:
       -----------
       group_name : str
           Name of the edge group
       bc_type : str
           'dirichlet' or 'neumann'
       value_func : callable or float/tuple
           Function f(x, y) or constant value for the BC
           For vectors: can be tuple (ux, uy) or function returning (ux, uy)
       is_vector : bool
           True if this is a vector-valued BC (for Stokes)
       """
       if isinstance(value_func, (int, float)):
           const_val = value_func
           value_func = lambda *coordinates, v=const_val: v
       elif isinstance(value_func, tuple):
           const_vals = value_func
           value_func = lambda *coordinates, v=const_vals: v

       self.conditions[group_name] = BoundaryCondition(bc_type, value_func, is_vector)

    def add_bc_by_face_group(self, group_name, bc_type, value_func, is_vector=False):
        """3D spelling of :meth:`add_bc_by_group`."""
        self.add_bc_by_group(group_name, bc_type, value_func, is_vector)

    def add_bc_by_function(self, region_func, bc_type, value_func, name=None, tolerance=1e-10, is_vector=False):
        """
        Add a boundary condition defined by an analytical function

        Parameters:
        -----------
        region_func : callable
            Function f(x, y) that returns True if point (x, y) is on this boundary
        bc_type : str
            'dirichlet' or 'neumann'
        value_func : callable or float/tuple
            Function f(x, y) or constant value for the BC
        name : str, optional
            Name for this boundary region
        tolerance : float, default=1e-10
            Tolerance for region function evaluation
        is_vector : bool
            True if this is a vector-valued BC
        """
        if isinstance(value_func, (int, float)):
            const_val = value_func
            value_func = lambda *coordinates, v=const_val: v
        elif isinstance(value_func, tuple):
            const_vals = value_func
            value_func = lambda *coordinates, v=const_vals: v

        bc = BoundaryCondition(bc_type, value_func, is_vector)
        self.analytical_conditions.append((region_func, bc, name, tolerance))

        if name:
            print(f"Added analytical BC '{name}' ({bc_type})")

    def add_bc_to_all_boundaries(self, bc_type, value_func, is_vector=False):
        """
        Add a boundary condition to ALL boundary edges (global fallback)

        Parameters:
        -----------
        bc_type : str
            'dirichlet' or 'neumann'
        value_func : callable or float/tuple
            Function f(x, y) or constant value for the BC
        is_vector : bool
            True if this is a vector-valued BC
        """
        if isinstance(value_func, (int, float)):
            const_val = value_func
            value_func = lambda *coordinates, v=const_val: v
        elif isinstance(value_func, tuple):
            const_vals = value_func
            value_func = lambda *coordinates, v=const_vals: v

        self.global_boundary_condition = BoundaryCondition(bc_type, value_func, is_vector)
        print(f"Set global boundary condition: {bc_type}")
    
    def set_default_bc(self, bc_type, value_func):
        """
        Alias for add_bc_to_all_boundaries for clarity
        Sets the default boundary condition for unspecified boundaries
        """
        self.add_bc_to_all_boundaries(bc_type, value_func)
    
    def get_bc(self, edge_id):
        """
        Get boundary condition for a specific edge
        Priority: 1) MED group, 2) Analytical function, 3) Global boundary BC, 4) Default
        
        Returns:
        --------
        BoundaryCondition or None
        """
        # First check if edge belongs to a MED group
        group_name = self.edge_to_group.get(edge_id)
        if group_name is not None:
            bc = self.conditions.get(group_name)
            if bc is not None:
                return bc
        
        # Check analytical conditions
        if getattr(self.mesh, "dimension", 2) == 3:
            edge_mid = self.mesh.face_centroid(edge_id)
        else:
            edge_mid = self.mesh.edge_midpoint(edge_id)
        
        for region_func, bc, name, tol in self.analytical_conditions:
            try:
                if region_func(*edge_mid):
                    return bc
            except:
                # If region_func fails, skip this condition
                continue
        
        # Check if there's a global boundary condition
        if self.global_boundary_condition is not None:
            return self.global_boundary_condition
        
        # Default to homogeneous Dirichlet
        return self.default_dirichlet


__all__ = ["BoundaryCondition", "BoundaryConditionManager"]
