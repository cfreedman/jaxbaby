from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import optimistix as optx


class Model:
    def __init__(
        self,
        endogenous: List[str],
        exogenous: List[str],
        parameters: dict[str, ArrayLike],
        bounds: dict[str, jnp.ndarray],
        equations: List[Callable[[dict[str, ArrayLike]], jnp.ndarray]],
        ss_equations: List[Callable[[dict[str, ArrayLike]], jnp.ndarray]],
    ):
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.variables = endogenous + exogenous
        self.parameters = parameters
        self.bounds = bounds
        self.equations = equations
        self.ss_equations = ss_equations

    def make_ss_context(self, x: jnp.ndarray):
        # ctx = {}
        # for i, v in enumerate(self.variables):
        #     bounds = self.bounds[v]
        #     value = jnp.minimum(jnp.maximum(x[i], bounds[0]), bounds[1])
        #     ctx[v] = value
        ctx: dict[str, ArrayLike] = {v: x[i] for i, v in enumerate(self.variables)}
        ctx.update(self.parameters)
        return ctx

    def ss_solver(self, init_guess=None, tol=1e-6, max_iterations=200):
        def ss_residual(input: jnp.ndarray, *args) -> jnp.ndarray:
            ctx = self.make_ss_context(input)
            jax.debug.print("Context: {}", ctx)
            residuals = [equation(ctx) for equation in self.ss_equations]
            jax.debug.print("Residuals: {}", residuals)
            return jnp.stack(residuals)

        init = (
            init_guess
            if init_guess is not None
            else 0.5 * jnp.ones(len(self.variables))
        )

        solver = optx.Newton(rtol=tol, atol=tol)
        solution = optx.root_find(fn=ss_residual, solver=solver, y0=init)

        return solution.value
