import pathlib
import sys
import unittest

import jax
import jax.numpy as jnp


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


from xclib.CF25D_jax import int_xc_jax  # noqa: E402
from xclib.cf25d_flax import CF25DXCEnergyModel, CF25DXCModel, flax_params_to_cf25d_params  # noqa: E402


def _sample_rho_weight(key, *, ngrid=8, dtype=jnp.float32):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    rho0 = jax.random.uniform(k1, (2, 1, ngrid), minval=0.2, maxval=2.0, dtype=dtype)
    grad = jax.random.normal(k2, (2, 3, ngrid), dtype=dtype) * jnp.asarray(0.01, dtype=dtype)
    lapl = jax.random.normal(k3, (2, 1, ngrid), dtype=dtype) * jnp.asarray(0.01, dtype=dtype)
    tau = jax.random.uniform(k4, (2, 1, ngrid), minval=0.2, maxval=2.0, dtype=dtype)
    rho = jnp.concatenate([rho0, grad, lapl, tau], axis=1)
    weight = jnp.abs(jax.random.normal(k5, (ngrid,), dtype=dtype)) + jnp.asarray(0.1, dtype=dtype)
    return rho, weight


class TestCF25DXCModel(unittest.TestCase):
    def test_model_output_matches_function(self):
        rho, weight = _sample_rho_weight(jax.random.PRNGKey(0), ngrid=8)
        model = CF25DXCModel()
        variables = model.init(jax.random.PRNGKey(42), rho, weight)
        out_model = model.apply(variables, rho, weight)

        params = flax_params_to_cf25d_params(variables["params"])
        out_fn = int_xc_jax(params, rho, weight)

        self.assertEqual(out_model.shape, (76,))
        self.assertTrue(jnp.isfinite(out_model).all())
        self.assertTrue(jnp.allclose(out_model, out_fn, rtol=1e-6, atol=1e-6))

    def test_grad_wrt_flax_params_runs(self):
        rho, weight = _sample_rho_weight(jax.random.PRNGKey(1), ngrid=8)
        model = CF25DXCModel()
        variables = model.init(jax.random.PRNGKey(0), rho, weight)

        def loss_fn(p):
            return jnp.sum(model.apply({"params": p}, rho, weight))

        grads = jax.grad(loss_fn)(variables["params"])
        leaves = jax.tree_util.tree_leaves(grads)
        self.assertGreater(len(leaves), 0)
        self.assertTrue(all(jnp.isfinite(x).all() for x in leaves))


class TestCF25DXCEnergyModel(unittest.TestCase):
    def test_hf_only_when_linear_coeff_zero(self):
        rho, weight = _sample_rho_weight(jax.random.PRNGKey(2), ngrid=8)
        e_hf = jnp.asarray(-1.234, dtype=jnp.float32)
        model = CF25DXCEnergyModel()
        variables = model.init(jax.random.PRNGKey(0), rho, weight, e_hf)
        out = model.apply(variables, rho, weight, e_hf)
        self.assertTrue(jnp.allclose(out, 0.462805832 * e_hf, rtol=1e-6, atol=1e-6))

    def test_grad_wrt_params_runs(self):
        rho, weight = _sample_rho_weight(jax.random.PRNGKey(3), ngrid=8)
        e_hf = jnp.asarray(-0.5, dtype=jnp.float32)
        model = CF25DXCEnergyModel()
        variables = model.init(jax.random.PRNGKey(0), rho, weight, e_hf)

        def loss_fn(p):
            return model.apply({"params": p}, rho, weight, e_hf)

        grads = jax.grad(loss_fn)(variables["params"])
        leaves = jax.tree_util.tree_leaves(grads)
        self.assertGreater(len(leaves), 0)
        self.assertTrue(all(jnp.isfinite(x).all() for x in leaves))


if __name__ == "__main__":
    unittest.main()
