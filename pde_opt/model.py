from typing import Type, Dict, Any, List

from pde_opt.numerics.equations import BaseEquation
from pde_opt.numerics import domains
import diffrax as dfx
import jax


class OptimizationModel:
    def __init__(
        self,
        equation_type: Type[BaseEquation],
        domain: domains.Domain,
        solver_type: Type[dfx.AbstractSolver],
    ):
        """
        equation_type: The class of the equation to optimize (from numerics/equations.py)
        domain: The domain to use for the equation
        parameters: Dictionary of all parameters (except domain) to instantiate the equation
        optimize_params: List of parameter names to optimize over
        solver_type: The class of the solver to use (e.g., diffrax.Tsit5)
        """
        self.equation_type = equation_type
        self.domain = domain
        self.solver_type = solver_type

    def solve(
        self,
        parameters: Dict[str, Any],
        y0,
        ts,
        solver_parameters: Dict[str, Any],
        adjoint=dfx.ForwardMode(),
        dt0=0.000001,
        max_steps=100000,
    ):
        """
        Solve the equation with given parameters.
        
        Args:
            parameters: Dictionary of parameters to use for the equation
            y0: Initial condition
            saveat: SaveAt object specifying when to save solution
            solver_parameters: Dictionary of parameters for the solver
            solver_equation_attrs: List of equation attribute names to pass to solver
                                 (if None, will try to get from solver.required_equation_attrs)
            adjoint: Adjoint mode for differentiation
            dt0: Initial time step
            max_steps: Maximum number of steps
        """
        # Initialize the equation with the given parameters
        equation = self.equation_type(domain=self.domain, **parameters)
        
        
        # Try to get required attributes from solver class
        if hasattr(self.solver_type, 'required_equation_attrs'):
            solver_equation_attrs = self.solver_type.required_equation_attrs
        else:
            solver_equation_attrs = []
        
        # Prepare solver parameters including equation attributes if specified
        full_solver_params = solver_parameters.copy()
        if solver_equation_attrs:
            for attr_name in solver_equation_attrs:
                if hasattr(equation, attr_name):
                    full_solver_params[attr_name] = getattr(equation, attr_name)
        
        # Initialize the solver with solver_parameters and equation attributes
        solver = self.solver_type(**full_solver_params)
        
        # Solve with diffrax
        solution = dfx.diffeqsolve(
            dfx.ODETerm(jax.jit(lambda t, y, args: equation.rhs(y, t))),
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            saveat=dfx.SaveAt(ts=ts),
            max_steps=max_steps,
            throw=False,
            adjoint=adjoint,
        )
        
        return solution.ys

    # Methods for running the optimization will be added later
