import jax
import jax.numpy as jnp
import immrax as irx
import equinox as eqx
from functools import partial
from time import perf_counter


class NestedIntervals:
    """
    Default perturbation generator for NCMTrainer.

    Returns irx.icentpert(center, (i + 1) * pert) for level i, so that
    level 0 is the smallest region and each subsequent level expands by pert.

    Args:
        center: center point of the interval (e.g. x_eq or x_lin)
        pert: base perturbation half-widths; level i uses (i+1) * pert
    """

    def __init__(self, center, pert):
        self.center = center
        self.pert = pert

    def __call__(self, i):
        return irx.icentpert(self.center, (i + 1) * self.pert)


class NCMTrainer:
    """
    General Neural Contraction Metric (NCM) trainer.

    Trains NCM and control networks to satisfy the contraction condition:

        G(x) = Th(x)^T (LM_f(x) + Th(x) (Df(x) + cI)) <= 0

    and metric bounds:

        a I <= M(x) <= b I

    Args:
        sys: immrax System with .f(t, x, u) and .xlen
        ncm_fn: callable(x, ncm_net) -> upper-triangular Th matrix
        control_fn: callable(x, control_net) -> u
        a: lower bound on metric eigenvalues (M >= a I)
        b: upper bound on metric eigenvalues (M <= b I)
        c: contraction rate numerator (contracts at rate c/b)
        partition_indices: state indices to partition in the loss.
            If None, partitions all dimensions.
        device: JAX backend ("gpu" or "cpu")
    """

    def __init__(
        self,
        sys,
        ncm_fn,
        control_fn,
        a=1.0,
        b=100.0,
        c=1.0,
        partition_indices=None,
        device="gpu",
    ):
        self.sys = sys
        self.ncm_fn = ncm_fn
        self.control_fn = control_fn
        self.a = a
        self.b = b
        self.c = c
        self.partition_indices = (
            list(range(sys.xlen))
            if partition_indices is None
            else list(partition_indices)
        )
        self.device = device
        self._jit = partial(eqx.filter_jit, backend=device)
        self._Dncm = jax.jacfwd(ncm_fn, argnums=0)

    def G(self, x, ncm_net, control_net):
        Th = self.ncm_fn(x, ncm_net)

        def cl_f(x):
            return self.sys.f(0.0, x, self.control_fn(x, control_net))

        LM_f = jnp.einsum("ijk,k", self._Dncm(x, ncm_net), cl_f(x))
        return Th.T @ (
            LM_f + Th @ (jax.jacfwd(cl_f)(x) + self.c * jnp.eye(self.sys.xlen))
        )

    def M(self, x, ncm_net):
        Th = self.ncm_fn(x, ncm_net)
        return Th.T @ Th

    def G_crown(self, ix, ncm_net, control_net):
        lin_bounds = irx.crown(
            partial(self.G, ncm_net=ncm_net, control_net=control_net)
        )(ix)
        return lin_bounds(ix)

    def M_crown(self, ix, ncm_net):
        lin_bounds = irx.crown(partial(self.M, ncm_net=ncm_net))(ix)
        return lin_bounds(ix)

    @staticmethod
    def get_eigs_rohn(iG):
        iG = irx.interval(iG)
        G_rohn = irx.utils.get_rohn_corners(iG, "+")
        return jax.vmap(jnp.linalg.eigvalsh)(G_rohn)

    @staticmethod
    def get_bounds_iM(iM):
        Mcm = irx.utils.get_rohn_corners(iM, "-")
        _a = jnp.min(jax.vmap(jnp.linalg.eigvalsh)(Mcm))
        Mcp = irx.utils.get_rohn_corners(iM, "+")
        _b = jnp.max(jax.vmap(jnp.linalg.eigvalsh)(Mcp))
        return _a, _b

    def sample_loss(self, params, ix, key, num_samples=10):
        ncm_net, control_net = params
        samples = irx.utils.gen_ics(ix, num_samples, key)
        return jnp.sum(
            jax.vmap(
                lambda x: jnp.maximum(
                    jnp.max(jnp.linalg.eigvalsh(self.G(x, ncm_net, control_net))),
                    0.0,
                )
            )(samples)
        )

    def loss(self, params, ix, divs):
        ncm_net, control_net = params
        n = self.sys.xlen
        pidx = self.partition_indices
        k = len(pidx)

        ix_ut = irx.i2ut(ix)
        part_ut = jnp.concatenate(
            [ix_ut[jnp.array(pidx)], ix_ut[jnp.array([n + i for i in pidx])]]
        )
        partitions = irx.utils.get_partitions_ut(part_ut, divs**k)

        def get_sum_p(part_p_ut):
            ixp = irx.interval(
                ix.lower.at[jnp.array(pidx)].set(part_p_ut[:k]),
                ix.upper.at[jnp.array(pidx)].set(part_p_ut[k:]),
            )
            iG = self.G_crown(ixp, ncm_net, control_net)
            iM = self.M_crown(ixp, ncm_net)
            aM, bM = self.get_bounds_iM(iM)
            c_term = jnp.maximum(jnp.max(self.get_eigs_rohn(iG)), 0.0)
            a_term = jnp.maximum(self.a - aM, 0.0)
            b_term = jnp.maximum(bM - self.b, 0.0)
            return c_term + a_term + b_term, (a_term, b_term, c_term)

        results, (a_terms, b_terms, c_terms) = jax.vmap(get_sum_p)(partitions)
        return jnp.sum(results), (jnp.sum(a_terms), jnp.sum(b_terms), jnp.sum(c_terms))

    def train(
        self,
        params,
        optim,
        ix_gen,
        key=jax.random.key(0),
        num_pert=100,
        steps=1_000_000,
        stall_steps=None,
        divs=1,
        num_samples=10,
        print_every=1,
        ix_save_path=None,
    ):
        """
        Train with progressive perturbation growth. Saves networks and ix when
        loss reaches zero, then advances to the next perturbation level.

        Args:
            params: (ncm_net, control_net) tuple
            optim: optax optimizer
            ix_gen: callable(i: int) -> interval; returns the training region
                for perturbation level i. Use NestedIntervals for the default
                expanding-box scheme.
            key: JAX PRNGKey for sample_loss randomness
            num_pert: stop after this many pert levels verified
            steps: maximum training steps
            stall_steps: stop early if no new pert level verified within this many steps
            divs: initial sub-box divisions per partition dimension
            num_samples: number of samples used in sample_loss
            print_every: how often to print progress
            ix_save_path: if provided, saves ix_ut.npy here on each solve

        Returns:
            (params, ix, perti): final parameters, training interval, and pert levels verified
        """

        time0 = perf_counter()

        @self._jit
        def make_step(params, opt_state, ix, divs, key):
            def combined_loss(params):
                crown, (a_term, b_term, c_term) = self.loss(params, ix, divs)
                soft = self.sample_loss(params, ix, key, num_samples)
                return crown + soft, (crown, soft, a_term, b_term, c_term)

            (_, (crown_value, soft_value, a_val, b_val, c_val)), grads = (
                eqx.filter_value_and_grad(combined_loss, has_aux=True)(params)
            )
            updates, opt_state = optim.update(grads, opt_state, params)
            new_params = eqx.apply_updates(params, updates)
            return (
                new_params,
                opt_state,
                crown_value,
                soft_value,
                a_val,
                b_val,
                c_val,
                params,
            )

        perti = 0
        last_reset = 0
        ix = ix_gen(perti)
        opt_state = optim.init(eqx.filter(params, eqx.is_array))

        # JIT Compile
        print("JIT Compiling... ", end="")
        _ = make_step(params, opt_state, ix, divs, key)
        time1 = perf_counter()

        print(f"done in {time1 - time0:.4f} seconds.")
        print(f"Starting training at: {ix}")

        for step in range(steps):
            key, subkey = jax.random.split(key)
            params, opt_state, value, soft_value, a_val, b_val, c_val, old_params = (
                make_step(params, opt_state, ix, divs, subkey)
            )

            if (step % print_every) == 0 or value <= 0.0:
                print(
                    f"\r\033[Kstep={step:7d}, crown={value:.6f}, soft={soft_value:.6f}, "
                    f"a={a_val:.4f}, b={b_val:.4f}, c={c_val:.4f}, divs={divs:3d}, ({perti})",
                    end="",
                    flush=True,
                )

            if value <= 0.0:
                ncm_net, control_net = old_params
                print()
                ncm_net.save(verbose=False)
                control_net.save(verbose=False)
                if ix_save_path is not None:
                    jnp.save(ix_save_path, irx.i2ut(ix))
                last_reset = step
                perti += 1
                if perti >= num_pert:
                    break
                ix = ix_gen(perti)
                print(f"Advancing to pert level {perti}: {ix}")
                params = old_params
                opt_state = optim.init(eqx.filter(params, eqx.is_array))

            if stall_steps is not None and (step - last_reset) > stall_steps:
                print(f"\nStalled: no progress for {stall_steps} steps.")
                break

        time2 = perf_counter()

        print(f"\nJIT compilation time: {time1 - time0:.4f} seconds.")
        print(f"Iterations time: {time2 - time1:.4f} seconds.")
        print(f"Total time spent: {time2 - time0:.4f} seconds.")

        return params, ix, perti
