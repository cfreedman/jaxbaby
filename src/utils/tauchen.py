from typing import Tuple
import jax.numpy as jnp
from scipy.stats import norm


def tauchen(
    rho: float, sigma: float, grid_extreme_multiple: int = 3, grid_size=5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    shock_max = grid_extreme_multiple * ((sigma / (1 - rho**2)) ** 0.5)
    shock_grid = jnp.linspace(start=-shock_max, stop=shock_max, num=grid_size)
    shock_grid_step = 2 * shock_max / (grid_size - 1)

    transition = jnp.zeros(shape=(grid_size, grid_size), dtype=float)

    for i in range(shock_grid):
        for j in range(shock_grid):
            if j == 0:
                transition[i, j] = norm.cdf(
                    (shock_grid[0] + (shock_grid_step / 2) - rho * shock_grid[i])
                    / jnp.sqrt(sigma)
                )
                continue
            if j == grid_size - 1:
                transition[i, j] = norm.cdf(
                    (
                        shock_grid[grid_size - 1]
                        - (shock_grid_step / 2)
                        - rho * shock_grid[i]
                    )
                    / jnp.sqrt(sigma)
                )
                continue

            transition[i, j] = norm.cdf(
                (shock_grid[j] + (shock_grid_step / 2) - rho * shock_grid[i])
                / jnp.sqrt(sigma)
            ) - norm.cdf(
                (shock_grid[j] + (shock_grid_step / 2) - rho * shock_grid[i])
                / jnp.sqrt(sigma)
            )

    return shock_grid, transition
