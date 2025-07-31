import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Callable, Dict, Any
from .boundary_conditions import get_neighbor_indices, BoundaryType

# taken from evoxels

# Shorthands in slicing logic
__ = slice(None)    # all elements [:]
_i_ = slice(1, -1)  # inner elements [1:-1]

CENTER = (__,  _i_, _i_, _i_)
LEFT   = (__, slice(None,-2), _i_, _i_)
RIGHT  = (__, slice(2, None), _i_, _i_)
BOTTOM = (__, _i_, slice(None,-2), _i_)
TOP    = (__, _i_, slice(2, None), _i_)
BACK   = (__, _i_, _i_, slice(None,-2))
FRONT  = (__, _i_, _i_, slice(2, None))

def laplacian(dx: Union[float, Tuple[float, ...]] = 1.0,
              boundary_conditions: Optional[Dict[str, Any]] = None) -> Callable[[jax.Array, Optional[float]], jax.Array]:
    """
    Create a Laplacian operator function.
    
    Args:
        dx: Grid spacing. Can be a scalar (uniform spacing) or tuple of scalars 
            for each dimension (non-uniform spacing)
        boundary_conditions: Dictionary specifying boundary conditions for each dimension.
            Format: {axis: (boundary_type, boundary_values)} where:
            - axis: int (0, 1, 2 for x, y, z dimensions)
            - boundary_type: str ('periodic', 'dirichlet', 'neumann', 'time_dependent_dirichlet')
            - boundary_values: depends on boundary_type
                - 'dirichlet': 
                    - float: Constant value for both boundaries
                    - tuple: (left_val, right_val) for different boundary values
                    - callable: Function f(pos) that returns boundary value at position
                    - jax.Array: Spatially dependent boundary values
                - 'time_dependent_dirichlet': callable f(time) that returns boundary values
                - 'neumann': float or (left_grad, right_grad) 
                - 'periodic': None (ignored)
    
    Returns:
        Callable function that computes the Laplacian of a JAX array.
        The function signature is: laplacian_op(field, time=None)
        
    Example:
        >>> import jax.numpy as jnp
        >>> # Create Laplacian operator with periodic BCs
        >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('periodic', None)})
        >>> field = jnp.sin(2 * jnp.pi * jnp.linspace(0, 1, 10))
        >>> result = lap_op(field)
        
        # Spatially dependent Dirichlet BCs
        >>> def boundary_func(pos):
        ...     return jnp.sin(2 * jnp.pi * pos)
        >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('dirichlet', boundary_func)})
        
        # Vector-based boundary values
        >>> boundary_values = jnp.array([1.0, 2.0, 3.0])  # for 1D field
        >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('dirichlet', boundary_values)})
        
        # Time-dependent Dirichlet BCs
        >>> def time_boundary_func(t):
        ...     return jnp.array([jnp.sin(t), jnp.cos(t), jnp.sin(2*t)])
        >>> lap_op = laplacian(dx=0.1, boundary_conditions={0: ('time_dependent_dirichlet', time_boundary_func)})
        >>> result = lap_op(field, time=1.5)
    """
    def _laplacian_operator(field: jax.Array, time: Optional[float] = None) -> jax.Array:
        """Compute the Laplacian of the input field."""
        if field.ndim not in [1, 2, 3]:
            raise ValueError(f"Field must be 1D, 2D, or 3D, got {field.ndim}D")
        
        # Handle dx parameter
        if isinstance(dx, (int, float)):
            dx_array = (dx,) * field.ndim
        elif len(dx) != field.ndim:
            raise ValueError(f"dx must have length {field.ndim}, got {len(dx)}")
        else:
            dx_array = dx
        
        # Convert to JAX arrays
        dx_array = jnp.array(dx_array)
        dx2 = dx_array ** 2
        
        # Default boundary conditions (zero-padding)
        if boundary_conditions is None:
            boundary_conditions = {}
        
        if field.ndim == 1:
            return _compute_1d_laplacian(field, dx2[0], boundary_conditions, time)
        elif field.ndim == 2:
            return _compute_2d_laplacian(field, dx2, boundary_conditions, time)
        else:  # 3D
            return _compute_3d_laplacian(field, dx2, boundary_conditions, time)
    
    return _laplacian_operator

def _compute_1d_laplacian(field: jax.Array, dx2: float, boundary_conditions: Dict[str, Any], time: Optional[float] = None) -> jax.Array:
    """Compute 1D Laplacian using central finite differences."""
    axis = 0
    boundary_config = boundary_conditions.get(axis, ('dirichlet', 0.0))
    boundary_type, boundary_values = boundary_config
    
    left, right = get_neighbor_indices(field, boundary_type, boundary_values, axis, time)
    return (left + right - 2 * field) / dx2

def _compute_2d_laplacian(field: jax.Array, dx2: Tuple[float, float], boundary_conditions: Dict[str, Any], time: Optional[float] = None) -> jax.Array:
    """Compute 2D Laplacian using central finite differences."""
    dx2_x, dx2_y = dx2
    
    # Get boundary configurations for each axis
    x_bc = boundary_conditions.get(1, ('dirichlet', 0.0))  # x-axis (axis=1)
    y_bc = boundary_conditions.get(0, ('dirichlet', 0.0))  # y-axis (axis=0)
    
    # Get neighbors for x-direction
    left, right = get_neighbor_indices(field, x_bc[0], x_bc[1], axis=1, time=time)
    
    # Get neighbors for y-direction  
    bottom, top = get_neighbor_indices(field, y_bc[0], y_bc[1], axis=0, time=time)
    
    return (left + right) / dx2_x + (bottom + top) / dx2_y - 2 * field * (1/dx2_x + 1/dx2_y)

def _compute_3d_laplacian(field: jax.Array, dx2: Tuple[float, float, float], boundary_conditions: Dict[str, Any], time: Optional[float] = None) -> jax.Array:
    """Compute 3D Laplacian using central finite differences."""
    dx2_x, dx2_y, dx2_z = dx2
    
    # Get boundary configurations for each axis
    x_bc = boundary_conditions.get(2, ('dirichlet', 0.0))  # x-axis (axis=2)
    y_bc = boundary_conditions.get(1, ('dirichlet', 0.0))  # y-axis (axis=1)
    z_bc = boundary_conditions.get(0, ('dirichlet', 0.0))  # z-axis (axis=0)
    
    # Get neighbors for each direction
    left, right = get_neighbor_indices(field, x_bc[0], x_bc[1], axis=2, time=time)
    bottom, top = get_neighbor_indices(field, y_bc[0], y_bc[1], axis=1, time=time)
    back, front = get_neighbor_indices(field, z_bc[0], z_bc[1], axis=0, time=time)
    
    return (left + right) / dx2_x + (bottom + top) / dx2_y + (back + front) / dx2_z - 2 * field * (1/dx2_x + 1/dx2_y + 1/dx2_z)

# Keep the original laplace method for backward compatibility
def laplace(self, field):
    r"""Calculate laplace based on compact 2nd order stencil.

    Laplace given as $\nabla\cdot(\nabla u)$ which in 3D is given by
    $\partial^2 u/\partial^2 x + \partial^2 u/\partial^2 y+ \partial^2 u/\partial^2 z$
    Returned field has same shape as the input field (padded with zeros)
    """
    # Manual indexing is ~10x faster than conv3d with laplace kernel in torch
    laplace = \
        (field[RIGHT] + field[LEFT]) * self.div_dx2[0] + \
        (field[TOP] + field[BOTTOM]) * self.div_dx2[1] + \
        (field[FRONT] + field[BACK]) * self.div_dx2[2] - \
            2 * field[CENTER] * self.lib.sum(self.div_dx2)
    return laplace