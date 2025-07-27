import jax
import jax.numpy as jnp

from src.core.model import Model

endogenous = [
    "c",
    "k",
    "l",
]
exogenous = ["z"]
parameters = {
    "alpha": 0.6,
    "beta": 0.98,
    "delta": 0.1,
    "eta": 1.2,
    "rho": 0.8,
    "sigma": 1.0,
    "psi": 3.0,
    "phi": 1.4,
}


def resource_constraint(ctx: dict[str, float]) -> jnp.ndarray:
    c = ctx["c"]
    k = ctx["k"]
    l = ctx["l"]

    alpha = ctx["alpha"]
    delta = ctx["delta"]

    res = (k**alpha) * (l ** (1 - alpha)) + (1 - delta) * k - c - k
    return jnp.array(res)


def euler(ctx: dict[str, float]) -> jnp.ndarray:
    k = ctx["k"]
    l = ctx["l"]

    alpha = ctx["alpha"]
    beta = ctx["beta"]
    delta = ctx["delta"]

    res = 1 - beta * (alpha * (k ** (alpha - 1)) * (l ** (1 - alpha)) + 1 - delta)
    return jnp.array(res)


def labor_optimality(ctx: dict[str, float]) -> jnp.ndarray:
    c = ctx["c"]
    k = ctx["k"]
    l = ctx["l"]

    alpha = ctx["alpha"]
    eta = ctx["eta"]
    psi = ctx["psi"]
    phi = ctx["phi"]

    res = psi * phi * (l ** (phi - 1)) * (c**eta) - (1 - alpha) * (k**alpha) * (
        l ** (-alpha)
    )
    return jnp.array(res)


ss_equations = [resource_constraint, euler, labor_optimality]

model = Model(
    endogenous=endogenous,
    exogenous=exogenous,
    parameters=parameters,
    bounds={
        "c": jnp.array([1e-8, float("inf")]),
        "k": jnp.array([1e-8, float("inf")]),
        "l": jnp.array([1e-8, float("inf")]),
        "z": jnp.array([float("-inf"), float("inf")]),
    },
    equations=[],
    ss_equations=ss_equations,
)

print(model.ss_solver(init_guess=jnp.array([2, 13, 0.3])))
