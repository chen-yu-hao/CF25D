import pathlib
import sys
import unittest

import jax
import jax.numpy as jnp


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


from xclib.CF25D_jax import int_xc_jax  # noqa: E402
from xclib.numint_tools_jax import params as DEFAULT_PARAMS  # noqa: E402


def _as_trainable_params(params_dict, *, dtype=jnp.float32):
    return {k: jnp.asarray(v, dtype=dtype) for k, v in params_dict.items()}


def _sample_rho_weight(key, *, ngrid=8, dtype=jnp.float32):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # rho: (2, 6, ngrid) = (spin, [rho, grad_x, grad_y, grad_z, lapl, tau], grid)
    rho0 = jax.random.uniform(k1, (2, 1, ngrid), minval=0.2, maxval=2.0, dtype=dtype)
    grad = jax.random.normal(k2, (2, 3, ngrid), dtype=dtype) * jnp.asarray(0.01, dtype=dtype)
    lapl = jax.random.normal(k3, (2, 1, ngrid), dtype=dtype) * jnp.asarray(0.01, dtype=dtype)
    tau = jax.random.uniform(k4, (2, 1, ngrid), minval=0.2, maxval=2.0, dtype=dtype)
    rho = jnp.concatenate([rho0, grad, lapl, tau], axis=1)

    weight = jnp.abs(jax.random.normal(k5, (ngrid,), dtype=dtype)) + jnp.asarray(0.1, dtype=dtype)
    return rho, weight


class TestIntXcJax(unittest.TestCase):
    def test_output_shape_and_finite(self):
        params = _as_trainable_params(DEFAULT_PARAMS)
        rho, weight = _sample_rho_weight(jax.random.PRNGKey(0), ngrid=8)

        out = int_xc_jax(params, rho, weight)
        self.assertEqual(out.shape, (76,))
        self.assertTrue(jnp.isfinite(out).all())

    def test_grad_wrt_params_runs(self):
        params = _as_trainable_params(DEFAULT_PARAMS)
        rho, weight = _sample_rho_weight(jax.random.PRNGKey(1), ngrid=8)

        def loss_fn(p):
            return jnp.sum(int_xc_jax(p, rho, weight))

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        self.assertGreater(len(leaves), 0)
        self.assertTrue(all(jnp.isfinite(x).all() for x in leaves))


if __name__ == "__main__":
    unittest.main()

