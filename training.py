# %%

import jax
import jax.numpy as jnp
import immrax as irx
import equinox as eqx
from functools import partial
import optax

from ncm_trainer import NCMTrainer
from quadrotor import (
    Quadrotor,
    x_eq,
    u_eq,
    ncm,
    control,
    K_eq,
    S_eq,
    A_eq,
    B_eq,
    NCM,
    CONTROLLER,
)

device = "gpu"
jit = partial(eqx.filter_jit, backend=device)

sys = Quadrotor()
print(sys.f(0.0, x_eq, u_eq))

# Derive a, b, c from the LQR initialization (50% reduction/bloating)
a = 0.01
b = 100.0
c = 0.001
lr = 1e-5
print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}")

trainer = NCMTrainer(
    sys=sys,
    ncm_fn=ncm,
    control_fn=control,
    a=a,
    b=b,
    c=c,
    partition_indices=[6, 7, 8],
    device=device,
)

# %%

# ix_gen: linearly grows perturbation from 1% to 100% of max over num_pert levels.
_pert_max = jnp.array(
    [
        10.0,
        10.0,
        10.0,
        5.0,
        5.0,
        5.0,
        9.81 / 3.0,
        jnp.pi / 8.0,
        jnp.pi / 8.0,
        jnp.pi / 2.0,
    ]
)
_num_levels = 100


def ix_gen(i):
    alpha = i / (_num_levels - 1)
    pert = _pert_max * (0.01 + alpha * 0.99)
    return irx.icentpert(x_eq, pert)


# %%
ncm_net = irx.NeuralNetwork(NCM, load=False)
control_net = irx.NeuralNetwork(CONTROLLER, load=False)
ncm_net = ncm_net.loadzeros()
control_net = control_net.loadzeros()

params = (ncm_net, control_net)
optim = optax.adamw(lr)

(ncm_net, control_net), ix, perti = trainer.train(
    params,
    optim,
    ix_gen,
    num_samples=0,
    divs=9,
    steps=1_000_000,
    ix_save_path=CONTROLLER / "ix_ut.npy",
    num_pert=_num_levels,
)

# %%

ncm_net = irx.NeuralNetwork(NCM, load=True)
control_net = irx.NeuralNetwork(CONTROLLER, load=True)

aM, bM = trainer.get_bounds_iM(trainer.M_crown(ix, ncm_net))
print(aM, bM)

print(f"Valid: {aM > 0}, Metric contracts at rate {c}")

# %%
