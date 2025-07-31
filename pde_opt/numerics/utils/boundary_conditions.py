import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Callable, Any
from enum import Enum

class BoundaryType(Enum):
    """Enumeration of boundary condition types."""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    TIME_DEPENDENT_DIRICHLET = "time_dependent_dirichlet"

def apply_boundary_conditions(field: jax.Array, 
                            boundary_type: Union[str, BoundaryType],
                            boundary_values: Optional[Union[float, Tuple[float, ...], Callable, jax.Array]] = None,
                            axis: int = 0,
                            time: Optional[float] = None) -> jax.Array:
    """
    Apply boundary conditions to a field along a specified axis.
    
    Args:
        field: Input array
        boundary_type: Type of boundary condition ('periodic', 'dirichlet', 'neumann', 'time_dependent_dirichlet')
        boundary_values: Values for boundary conditions. Can be:
            - float: Constant value for Dirichlet/Neumann
            - tuple: Different values for each boundary (left/right, bottom/top, etc.)
            - callable: Function that takes position and returns boundary value
            - jax.Array: Spatially dependent boundary values (must match boundary shape)
            - callable: For time_dependent_dirichlet, function that takes time and returns boundary values
        axis: Axis along which to apply boundary conditions
        time: Current time (required for time_dependent_dirichlet)
    
    Returns:
        Array with boundary conditions applied
    """
    if isinstance(boundary_type, str):
        boundary_type = BoundaryType(boundary_type)
    
    if boundary_type == BoundaryType.PERIODIC:
        return apply_periodic_boundary(field, axis)
    elif boundary_type == BoundaryType.DIRICHLET:
        return apply_dirichlet_boundary(field, boundary_values, axis)
    elif boundary_type == BoundaryType.NEUMANN:
        return apply_neumann_boundary(field, boundary_values, axis)
    elif boundary_type == BoundaryType.TIME_DEPENDENT_DIRICHLET:
        if time is None:
            raise ValueError("time parameter is required for time_dependent_dirichlet boundary conditions")
        return apply_time_dependent_dirichlet_boundary(field, boundary_values, axis, time)
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")

def apply_periodic_boundary(field: jax.Array, axis: int = 0) -> jax.Array:
    """Apply periodic boundary conditions using roll operation."""
    return jnp.roll(field, 1, axis=axis)

def apply_dirichlet_boundary(field: jax.Array, 
                           boundary_values: Union[float, Tuple[float, ...], Callable, jax.Array],
                           axis: int = 0) -> jax.Array:
    """
    Apply Dirichlet boundary conditions (fixed values at boundaries).
    
    Args:
        field: Input array
        boundary_values: Can be:
            - float: Constant value for both boundaries
            - tuple: (left_val, right_val) for different boundary values
            - callable: Function f(pos) that returns boundary value at position
            - jax.Array: Spatially dependent boundary values
        axis: Axis along which to apply boundary conditions
    """
    if isinstance(boundary_values, (int, float)):
        # Constant value for both boundaries
        left_val = right_val = boundary_values
        left_boundary = jnp.full((1,) + field.shape[1:], left_val, dtype=field.dtype) if axis == 0 else \
                       jnp.full((field.shape[0], 1) + field.shape[2:], left_val, dtype=field.dtype) if axis == 1 else \
                       jnp.full(field.shape[:-1] + (1,), left_val, dtype=field.dtype)
        right_boundary = left_boundary.copy()
        
    elif isinstance(boundary_values, (tuple, list)):
        # Different values for left and right boundaries
        if len(boundary_values) == 1:
            left_val = right_val = boundary_values[0]
        elif len(boundary_values) == 2:
            left_val, right_val = boundary_values
        else:
            raise ValueError("boundary_values tuple must have length 1 or 2")
        
        # Create boundary arrays
        left_shape = list(field.shape)
        left_shape[axis] = 1
        right_shape = list(field.shape)
        right_shape[axis] = 1
        
        left_boundary = jnp.full(left_shape, left_val, dtype=field.dtype)
        right_boundary = jnp.full(right_shape, right_val, dtype=field.dtype)
        
    elif callable(boundary_values):
        # Function-based boundary values
        left_boundary, right_boundary = _create_functional_boundaries(field, boundary_values, axis)
        
    elif isinstance(boundary_values, jax.Array):
        # Spatially dependent boundary values
        left_boundary, right_boundary = _create_spatial_boundaries(field, boundary_values, axis)
        
    else:
        raise ValueError("boundary_values must be float, tuple, callable, or jax.Array")
    
    # Concatenate with original field
    return jnp.concatenate([left_boundary, field, right_boundary], axis=axis)

def apply_time_dependent_dirichlet_boundary(field: jax.Array, 
                                          boundary_func: Callable, 
                                          axis: int = 0, 
                                          time: float = 0.0) -> jax.Array:
    """
    Apply time-dependent Dirichlet boundary conditions.
    
    Args:
        field: Input array
        boundary_func: Function f(time) that returns boundary values at given time
        axis: Axis along which to apply boundary conditions
        time: Current time
    
    Returns:
        Array with time-dependent boundary conditions applied
    """
    # Get boundary values at current time
    boundary_values = boundary_func(time)
    
    # Apply the boundary values using the standard Dirichlet method
    return apply_dirichlet_boundary(field, boundary_values, axis)

def _create_functional_boundaries(field: jax.Array, 
                                boundary_func: Callable, 
                                axis: int = 0) -> Tuple[jax.Array, jax.Array]:
    """Create boundary arrays using a function that takes position and returns value."""
    # Create position arrays for boundary evaluation
    if axis == 0:
        # y-axis boundary (left and right boundaries are at y=0 and y=1)
        left_pos = jnp.zeros(field.shape[1:])
        right_pos = jnp.ones(field.shape[1:])
        left_boundary = jnp.expand_dims(boundary_func(left_pos), 0)
        right_boundary = jnp.expand_dims(boundary_func(right_pos), 0)
    elif axis == 1:
        # x-axis boundary (left and right boundaries are at x=0 and x=1)
        left_pos = jnp.zeros(field.shape[0])
        right_pos = jnp.ones(field.shape[0])
        left_boundary = jnp.expand_dims(boundary_func(left_pos), 1)
        right_boundary = jnp.expand_dims(boundary_func(right_pos), 1)
    else:  # axis == 2
        # z-axis boundary (left and right boundaries are at z=0 and z=1)
        left_pos = jnp.zeros(field.shape[:2])
        right_pos = jnp.ones(field.shape[:2])
        left_boundary = jnp.expand_dims(boundary_func(left_pos), 2)
        right_boundary = jnp.expand_dims(boundary_func(right_pos), 2)
    
    return left_boundary, right_boundary

def _create_spatial_boundaries(field: jax.Array, 
                             boundary_values: jax.Array, 
                             axis: int = 0) -> Tuple[jax.Array, jax.Array]:
    """Create boundary arrays using spatially dependent boundary values."""
    # Check if boundary_values is a single array (same for both boundaries)
    if boundary_values.ndim == field.ndim - 1:
        # Single array for both boundaries
        left_boundary = jnp.expand_dims(boundary_values, axis)
        right_boundary = left_boundary.copy()
    elif boundary_values.ndim == field.ndim:
        # Check if it's a tuple-like structure (first and last slices)
        if boundary_values.shape[axis] == 2:
            # Array with shape (..., 2, ...) - first slice is left, second is right
            left_boundary = jnp.take(boundary_values, 0, axis=axis)
            right_boundary = jnp.take(boundary_values, 1, axis=axis)
        else:
            raise ValueError(f"Boundary array shape {boundary_values.shape} is incompatible with field shape {field.shape}")
    else:
        raise ValueError(f"Boundary array dimensions {boundary_values.ndim} must be {field.ndim - 1} or {field.ndim}")
    
    return left_boundary, right_boundary

def apply_neumann_boundary(field: jax.Array, 
                          boundary_values: Union[float, Tuple[float, ...]],
                          axis: int = 0) -> jax.Array:
    """Apply Neumann boundary conditions (fixed gradients at boundaries)."""
    if isinstance(boundary_values, (int, float)):
        left_grad = right_grad = boundary_values
    elif isinstance(boundary_values, (tuple, list)):
        if len(boundary_values) == 1:
            left_grad = right_grad = boundary_values[0]
        elif len(boundary_values) == 2:
            left_grad, right_grad = boundary_values
        else:
            raise ValueError("boundary_values tuple must have length 1 or 2")
    else:
        raise ValueError("boundary_values must be float or tuple")
    
    # For zero gradient (most common case)
    if left_grad == 0 and right_grad == 0:
        left_boundary = jnp.take(field, 0, axis=axis, mode='clip')
        right_boundary = jnp.take(field, -1, axis=axis, mode='clip')
    else:
        # For non-zero gradients, we need to extrapolate
        # This is a simplified implementation - in practice you might want more sophisticated extrapolation
        left_boundary = jnp.take(field, 0, axis=axis, mode='clip') - left_grad
        right_boundary = jnp.take(field, -1, axis=axis, mode='clip') + right_grad
    
    # Reshape for concatenation
    left_shape = list(field.shape)
    left_shape[axis] = 1
    right_shape = list(field.shape)
    right_shape[axis] = 1
    
    left_boundary = jnp.expand_dims(left_boundary, axis)
    right_boundary = jnp.expand_dims(right_boundary, axis)
    
    return jnp.concatenate([left_boundary, field, right_boundary], axis=axis)

def get_neighbor_indices(field: jax.Array, 
                        boundary_type: Union[str, BoundaryType],
                        boundary_values: Optional[Union[float, Tuple[float, ...], Callable, jax.Array]] = None,
                        axis: int = 0,
                        time: Optional[float] = None) -> Tuple[jax.Array, jax.Array]:
    """
    Get left and right neighbor arrays for finite difference calculations.
    
    Returns:
        Tuple of (left_neighbors, right_neighbors) arrays
    """
    if isinstance(boundary_type, str):
        boundary_type = BoundaryType(boundary_type)
    
    if boundary_type == BoundaryType.PERIODIC:
        left = jnp.roll(field, 1, axis=axis)
        right = jnp.roll(field, -1, axis=axis)
    elif boundary_type == BoundaryType.DIRICHLET:
        if boundary_values is None:
            boundary_values = 0.0
        
        # Handle different types of boundary values
        if isinstance(boundary_values, (int, float)):
            left_val = right_val = boundary_values
        elif isinstance(boundary_values, (tuple, list)):
            if len(boundary_values) == 1:
                left_val = right_val = boundary_values[0]
            elif len(boundary_values) == 2:
                left_val, right_val = boundary_values
            else:
                left_val = right_val = boundary_values[0]
        elif callable(boundary_values):
            # For functional boundaries, we need to evaluate at boundary positions
            left_boundary, right_boundary = _create_functional_boundaries(field, boundary_values, axis)
            left = jnp.concatenate([left_boundary, field[:, :-1]] if axis == 1 else 
                                  [left_boundary, field[:-1, :]] if axis == 0 else
                                  [left_boundary, field[:, :, :-1]], axis=axis)
            right = jnp.concatenate([field[:, 1:], right_boundary] if axis == 1 else
                                   [field[1:, :], right_boundary] if axis == 0 else
                                   [field[:, :, 1:], right_boundary], axis=axis)
            return left, right
        elif isinstance(boundary_values, jax.Array):
            # For spatial boundaries, we need to extract the boundary values
            left_boundary, right_boundary = _create_spatial_boundaries(field, boundary_values, axis)
            left = jnp.concatenate([left_boundary, field[:, :-1]] if axis == 1 else 
                                  [left_boundary, field[:-1, :]] if axis == 0 else
                                  [left_boundary, field[:, :, :-1]], axis=axis)
            right = jnp.concatenate([field[:, 1:], right_boundary] if axis == 1 else
                                   [field[1:, :], right_boundary] if axis == 0 else
                                   [field[:, :, 1:], right_boundary], axis=axis)
            return left, right
        else:
            raise ValueError("boundary_values must be float, tuple, callable, or jax.Array")
        
        # Create boundary arrays for simple cases
        left_shape = list(field.shape)
        left_shape[axis] = 1
        right_shape = list(field.shape)
        right_shape[axis] = 1
        
        left_boundary = jnp.full(left_shape, left_val, dtype=field.dtype)
        right_boundary = jnp.full(right_shape, right_val, dtype=field.dtype)
        
        left = jnp.concatenate([left_boundary, field[:, :-1]] if axis == 1 else 
                              [left_boundary, field[:-1, :]] if axis == 0 else
                              [left_boundary, field[:, :, :-1]], axis=axis)
        right = jnp.concatenate([field[:, 1:], right_boundary] if axis == 1 else
                               [field[1:, :], right_boundary] if axis == 0 else
                               [field[:, :, 1:], right_boundary], axis=axis)
    elif boundary_type == BoundaryType.TIME_DEPENDENT_DIRICHLET:
        if time is None:
            raise ValueError("time parameter is required for time_dependent_dirichlet boundary conditions")
        if not callable(boundary_values):
            raise ValueError("boundary_values must be callable for time_dependent_dirichlet")
        
        # Get boundary values at current time
        current_boundary_values = boundary_values(time)
        
        # Apply the boundary values using the standard Dirichlet method
        return get_neighbor_indices(field, BoundaryType.DIRICHLET, current_boundary_values, axis)
    elif boundary_type == BoundaryType.NEUMANN:
        # Zero gradient (most common case)
        if boundary_values is None or (isinstance(boundary_values, (int, float)) and boundary_values == 0):
            left = jnp.concatenate([jnp.expand_dims(jnp.take(field, 0, axis=axis), axis), 
                                   field[:, :-1]] if axis == 1 else
                                  [jnp.expand_dims(jnp.take(field, 0, axis=axis), axis), 
                                   field[:-1, :]] if axis == 0 else
                                  [jnp.expand_dims(jnp.take(field, 0, axis=axis), axis), 
                                   field[:, :, :-1]], axis=axis)
            right = jnp.concatenate([field[:, 1:], 
                                    jnp.expand_dims(jnp.take(field, -1, axis=axis), axis)] if axis == 1 else
                                   [field[1:, :], 
                                    jnp.expand_dims(jnp.take(field, -1, axis=axis), axis)] if axis == 0 else
                                   [field[:, :, 1:], 
                                    jnp.expand_dims(jnp.take(field, -1, axis=axis), axis)], axis=axis)
        else:
            # Non-zero gradient - simplified implementation
            left = jnp.concatenate([jnp.expand_dims(jnp.take(field, 0, axis=axis), axis), 
                                   field[:, :-1]] if axis == 1 else
                                  [jnp.expand_dims(jnp.take(field, 0, axis=axis), axis), 
                                   field[:-1, :]] if axis == 0 else
                                  [jnp.expand_dims(jnp.take(field, 0, axis=axis), axis), 
                                   field[:, :, :-1]], axis=axis)
            right = jnp.concatenate([field[:, 1:], 
                                    jnp.expand_dims(jnp.take(field, -1, axis=axis), axis)] if axis == 1 else
                                   [field[1:, :], 
                                    jnp.expand_dims(jnp.take(field, -1, axis=axis), axis)] if axis == 0 else
                                   [field[:, :, 1:], 
                                    jnp.expand_dims(jnp.take(field, -1, axis=axis), axis)], axis=axis)
    else:
        raise ValueError(f"Unsupported boundary type: {boundary_type}")
    
    return left, right 