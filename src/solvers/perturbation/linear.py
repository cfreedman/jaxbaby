from typing import Callable

import jax.numpy as jnp
import jax

# Representation equilbrium equilibrium model as E_{t}[H(y,y',x,x')] = 0 where y are controls, x are states
# (both endogenous and exogenous) and primes denote an increment of one period

type EquilibriumModel = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]


def perturbation_solve(
    model: EquilibriumModel, controls_ss: jnp.ndarray, state_ss: jnp.ndarray
):
    H_y, H_yprime, H_x, H_xprime, *_ = jax.vjp(
        model, controls_ss, controls_ss, state_ss, state_ss
    )
