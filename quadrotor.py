import jax
import jax.numpy as jnp
from numpy.linalg import cholesky
import numpy as onp
import immrax as irx
from control import lqr
from pathlib import Path

g = 9.81

BASE = Path(".")

NCM = BASE / "NCM"
CONTROLLER = BASE / "Controller"


class Quadrotor(irx.System):
    def __init__(self):
        self.evolution = "continuous"
        self.xlen = 10

    def f(self, t, x, u):
        px, py, pz, vx, vy, vz, f, phi, theta, psi = x
        u1, u2, u3, u4 = u
        return jnp.array(
            [
                vx,
                vy,
                vz,
                -f * jnp.sin(theta),
                f * jnp.cos(theta) * jnp.sin(phi),
                g - f * jnp.cos(theta) * jnp.cos(phi),
                u1,
                u2,
                u3,
                u4,
            ]
        )


_sys = Quadrotor()

x_eq = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
u_eq = jnp.array([0.0, 0.0, 0.0, 0.0])

A_eq = jax.jacfwd(_sys.f, argnums=1)(0.0, x_eq, u_eq)
B_eq = jax.jacfwd(_sys.f, argnums=2)(0.0, x_eq, u_eq)
Q = jnp.diag(jnp.array([0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
R = jnp.eye(4)
K_eq, S_eq, _ = lqr(A_eq, B_eq, Q, R)
K_eq = -jnp.array(K_eq)
Th_eq = jnp.asarray(cholesky(S_eq, upper=True))


def ncm(x, ncm_net):
    # Killing field condition, no dependence in actuation directions
    Th_flat = ncm_net(x[:-4])
    Th = jnp.zeros((_sys.xlen, _sys.xlen), dtype=Th_flat.dtype)
    Th = Th.at[onp.triu_indices(_sys.xlen)].set(Th_flat)
    return Th + Th_eq


def control(x, control_net):
    return control_net(x) + K_eq @ (x - x_eq) + u_eq


if __name__ == "__main__":
    from neural_contraction.Quadrotor import CONTROLLER

    jnp.savez(
        CONTROLLER / "eq.npz", K_eq=K_eq, x_eq=x_eq, u_eq=u_eq, Th_eq=Th_eq, Q=Q, R=R
    )
