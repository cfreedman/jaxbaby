from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import optimistix as optx


class Model:
    def __init__(
        self,
        endogenous: List[str],
        exogenous: List[str],
        parameters: dict[str, float],
        bounds: dict[str, jnp.ndarray],
        equations: List[Callable[[dict[str, float]], jnp.ndarray]],
        ss_equations: List[Callable[[dict[str, float]], jnp.ndarray]],
    ):
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.variables = endogenous + exogenous
        self.parameters = parameters
        self.bounds = bounds
        self.equations = equations
        self.ss_equations = ss_equations

    def make_ss_context(self, x: jnp.ndarray):
        ctx = {}
        for i, v in enumerate(self.variables):
            bounds = self.bounds[v]
            value = jnp.minimum(jnp.maximum(x[i], bounds[0]), bounds[1])
            ctx[v] = value
        ctx.update(self.parameters)
        return ctx

    def ss_solver(self, init_guess=None, tol=1e-8, max_iterations=200):
        def ss_residual(input: jnp.ndarray, *args) -> jnp.ndarray:
            ctx = self.make_ss_context(input)
            jax.debug.print("Context: {}", ctx)
            residuals = [equation(ctx) for equation in self.ss_equations]
            jax.debug.print("Residuals: {}", residuals)
            return jnp.stack(residuals)

        init = init_guess if init_guess else jnp.ones(len(self.variables))

        print(ss_residual(init))

        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(fn=ss_residual, solver=solver, y0=init)

        return solution.value
