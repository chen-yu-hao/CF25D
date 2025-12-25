from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from flax import linen as nn

from .CF25D_jax import int_xc_jax
from .numint_tools_jax import params as DEFAULT_CF25D_PARAMS


def _cf25d_key_to_flax_name(key: str) -> str:
    return key.replace("/", "__")


def _find_cf25d_param_mapping(flax_params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Best-effort extraction of CF25D params from a (possibly nested) flax tree."""

    expected_any = {_cf25d_key_to_flax_name(k) for k in DEFAULT_CF25D_PARAMS.keys()}
    if any(k in flax_params for k in expected_any):
        return flax_params

    for value in flax_params.values():
        if isinstance(value, Mapping) and any(k in value for k in expected_any):
            return value

    return flax_params


def flax_params_to_cf25d_params(
    flax_params: Mapping[str, Any],
    *,
    dtype: Any | None = None,
) -> dict[str, jnp.ndarray]:
    """Convert flax `params` tree back into the dict expected by `int_xc_jax`."""

    param_mapping = _find_cf25d_param_mapping(flax_params)
    out: dict[str, jnp.ndarray] = {}
    missing: list[str] = []
    for key in DEFAULT_CF25D_PARAMS.keys():
        name = _cf25d_key_to_flax_name(key)
        if name not in param_mapping:
            missing.append(name)
            continue
        value = param_mapping[name]
        out[key] = jnp.asarray(value, dtype=dtype) if dtype is not None else value

    if missing:
        raise KeyError(
            "Missing CF25D parameter(s) in flax params tree: " + ", ".join(missing)
        )
    return out


class CF25DXCModel(nn.Module):
    """Flax wrapper for `int_xc_jax(params, rho, weight)` (returns 76 features)."""

    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, rho: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
        rho = jnp.asarray(rho, dtype=self.dtype)
        weight = jnp.asarray(weight, dtype=self.dtype)

        cf25d_params: dict[str, jnp.ndarray] = {}
        for key, default_value in DEFAULT_CF25D_PARAMS.items():
            init_value = jnp.asarray(default_value, dtype=self.dtype)
            name = _cf25d_key_to_flax_name(key)

            def init_fn(rng, shape, dtype, *, _init_value=init_value):
                return jnp.asarray(_init_value, dtype=dtype)

            cf25d_params[key] = self.param(name, init_fn, init_value.shape, self.dtype)

        return int_xc_jax(cf25d_params, rho, weight)


class CF25DXCEnergyModel(nn.Module):
    """Trainable XC energy model based on CF25D features.

    E_xc = hf_coeff * E_HF + dot(linear_coeff, int_xc_jax(params, rho, weight))
    """

    dtype: Any = jnp.float32
    hf_coeff: float = 0.462805832

    @nn.compact
    def __call__(
        self, rho: jnp.ndarray, weight: jnp.ndarray, e_hf: jnp.ndarray
    ) -> jnp.ndarray:
        rho = jnp.asarray(rho, dtype=self.dtype)
        weight = jnp.asarray(weight, dtype=self.dtype)
        e_hf = jnp.asarray(e_hf, dtype=self.dtype)

        features = CF25DXCModel(dtype=self.dtype)(rho, weight)
        linear_coeff = self.param(
            "linear_coeff", nn.initializers.zeros, (features.shape[0],), self.dtype
        )
        return jnp.asarray(self.hf_coeff, dtype=self.dtype) * e_hf + jnp.dot(
            linear_coeff, features
        )

