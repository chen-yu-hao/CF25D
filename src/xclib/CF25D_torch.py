from __future__ import annotations

from typing import Any, Mapping

import torch

from .numint_tools_torch import (
    Dsigma,
    E_ijk,
    PW_alpha_c,
    PW_mod_c,
    PBE_pw_c,
    exf,
    h_function,
    uxf,
    vxf,
    wf,
    wf_total,
    xf,
    zetf,
)


def int_xc_torch(
    params: Mapping[str, Any],
    rho: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Torch rewrite of `xclib.CF25D_jax.int_xc_jax` (returns 76 features)."""
    wx = params["wx"]
    gamma = params["gamma"]
    alpha = params["alpha"]
    alpha_anti = params["alpha_anti"]

    ee: list[torch.Tensor] = []

    xa = xf(rho[0])
    xb = xf(rho[1])
    xab = torch.sqrt(xa**2 + xb**2)

    vx_a = vxf(rho[0], wx=wx)
    ux_a = uxf(xa, gamma=gamma)
    w_a = wf(rho[0])
    ex_a = exf(rho[0])

    vx_b = vxf(rho[1], wx=wx)
    ux_b = uxf(xb, gamma=gamma)
    w_b = wf(rho[1])
    ex_b = exf(rho[1])

    for i in range(4):
        for j in range(4 - i):
            for k in range(6 - i - j):
                ee.append(
                    E_ijk(rho[0], ex_a, vx_a, ux_a, w_a, i, j, k, weight)
                    + E_ijk(rho[1], ex_b, vx_b, ux_b, w_b, i, j, k, weight)
                )

    w_total = wf_total(rho)
    lsda = PW_mod_c(rho, params)

    rho_total = rho[:, 0].sum(dim=0)
    for i in range(9):
        ee.append(torch.dot(weight, rho_total * (lsda * (w_total**i))))

    pbe_lsda = PBE_pw_c(rho, lsda, params)
    for i in range(9):
        ee.append(torch.dot(weight, rho_total * (pbe_lsda * (w_total**i))))

    rho_a = rho[0, 0, :]
    rho_b = rho[1, 0, :]

    ueg_ab = lsda
    ueg_a = PW_alpha_c(rho_a, params, dens_threshold=1e-15)
    ueg_b = PW_alpha_c(rho_b, params, dens_threshold=1e-15)

    zeta_a = zetf(rho[0])
    zeta_b = zetf(rho[1])
    zeta_ab = zeta_a + zeta_b

    h_anti = h_function(xa**2 + xb**2, zeta_ab, alpha=alpha_anti)
    h_a = h_function(xa**2, zeta_a, alpha=alpha)
    h_b = h_function(xb**2, zeta_b, alpha=alpha)

    d_a = Dsigma(rho[0])
    d_b = Dsigma(rho[1])

    for value in h_anti:
        ee.append(torch.dot(weight, (ueg_ab * (rho_a + rho_b) - ueg_a * rho_a - ueg_b * rho_b) * value))

    for a, b in zip(h_a, h_b):
        ee.append(torch.dot(weight, ueg_a * rho_a * a * d_a + ueg_b * rho_b * b * d_b))

    ucab = uxf(xab, 0.0031)
    for i in range(5):
        ee.append(torch.dot(weight, (ueg_ab * (rho_a + rho_b) - ueg_a * rho_a - ueg_b * rho_b) * (ucab**i)))

    ua = uxf(xa, 0.06)
    ub = uxf(xb, 0.06)
    for i in range(5):
        ee.append(torch.dot(weight, ueg_a * rho_a * (ua**i) * d_a + ueg_b * rho_b * (ub**i) * d_b))

    return torch.stack(ee, dim=0)

