from __future__ import annotations

import math
from typing import Any, Mapping

import torch


DEFAULT_PARAMS: dict[str, Any] = {
    "PW/params_a_pp": [1, 1, 1],
    "PW/params_a_a": [0.0310907, 0.01554535, 0.0168869],
    "PW/params_a_alpha1": [0.21370, 0.20548, 0.11125],
    "PW/params_a_beta1": [7.5957, 14.1189, 10.357],
    "PW/params_a_beta2": [3.5876, 6.1977, 3.6231],
    "PW/params_a_beta3": [1.6382, 3.3662, 0.88026],
    "PW/params_a_beta4": [0.49294, 0.62517, 0.49671],
    "PW/params_a_fz20": 1.709920934161365617563962776245,
    "fH/params_a_beta": 0.06672455060314922,
    "fH/params_a_gamma": (1.0 - math.log(2.0)) / (math.pi**2),
    "fH/params_a_BB": 1.0,
    "fH/params_a_tscale": 1.0,
    "wx": 2.50,
    "gamma": 0.004,
    "alpha": 0.00515088,
    "alpha_anti": 0.00304966,
}


def make_default_params(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Materialize DEFAULT_PARAMS as torch tensors."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    out: dict[str, torch.Tensor] = {}
    for k, v in DEFAULT_PARAMS.items():
        t = torch.as_tensor(v, dtype=dtype, device=device)
        if requires_grad:
            t = t.clone().detach().requires_grad_(True)
        out[k] = t
    return out


def _to_tensor(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def vxf(rho: torch.Tensor, wx: torch.Tensor | float) -> torch.Tensor:
    return wx * rho[0] ** (1.0 / 3.0) / (1.0 + wx * rho[0] ** (1.0 / 3.0))


def xf(rho: torch.Tensor) -> torch.Tensor:
    return rho[0] ** (-4.0 / 3.0) * torch.linalg.norm(rho[1:4], dim=0)


def uxf(x: torch.Tensor, gamma: torch.Tensor | float) -> torch.Tensor:
    return gamma * x**2 / (1.0 + gamma * x**2)


def yf(rho: torch.Tensor) -> torch.Tensor:
    return (3.0 / 5.0) * (6.0 * math.pi**2) ** (2.0 / 3.0) * rho[0] ** (5.0 / 3.0) / rho[-1] / 2.0


def wf(rho: torch.Tensor) -> torch.Tensor:
    y = yf(rho)
    return (y - 1.0) / (y + 1.0)


def wf_total(rho: torch.Tensor) -> torch.Tensor:
    rho_total = rho[:, 0].sum(dim=0)
    t_total = rho[:, -1].sum(dim=0)
    y_total = (
        (3.0 / 10.0)
        * (3.0 * math.pi**2) ** (2.0 / 3.0)
        * rho_total ** (5.0 / 3.0)
        / t_total
    )
    return (y_total - 1.0) / (y_total + 1.0)


def exf(rho: torch.Tensor) -> torch.Tensor:
    return -(3.0 / 2.0) * (3.0 / 4.0 / math.pi) ** (1.0 / 3.0) * rho[0] ** (1.0 / 3.0)


def E_ijk(
    rho: torch.Tensor,
    ex: torch.Tensor,
    vx: torch.Tensor,
    ux: torch.Tensor,
    wx: torch.Tensor,
    i: int,
    j: int,
    k: int,
    weight: torch.Tensor,
    aijk: float = 1.0,
) -> torch.Tensor:
    integrand = rho[0] * ex * aijk * (vx**i) * (ux**j) * (wx**k)
    return torch.dot(weight, integrand)


def Dsigma(rho: torch.Tensor) -> torch.Tensor:
    return 1.0 - 0.125 * torch.linalg.norm(rho[1:4], dim=0) ** 2 / (rho[0] * rho[-1])


def zetf(rho: torch.Tensor) -> torch.Tensor:
    cf = 0.3 * (3.0 * math.pi**2) ** (2.0 / 3.0)
    scalefactor_tfconst = 3.17480210393640
    return 2.0 * rho[-1] / (rho[0] ** (5.0 / 3.0)) - cf * scalefactor_tfconst


def opz_pow_n(z: torch.Tensor, n: float, *, zeta_threshold: float = 1e-12) -> torch.Tensor:
    """(1+z)^n with zeta screening (mirrors libxc maple/util.mpl style)."""
    safe = torch.as_tensor(zeta_threshold**n, dtype=z.dtype, device=z.device)
    return torch.where(z + 1.0 <= zeta_threshold, safe, (z + 1.0) ** n)


def h_function(
    chi_ab2: torch.Tensor, zet_ab: torch.Tensor, *, alpha: torch.Tensor | float = 0.00304966
) -> torch.Tensor:
    gamma_val = 1.0 + alpha * (chi_ab2 + zet_ab)
    t1 = 1.0 / gamma_val
    t2_chi = chi_ab2 / (gamma_val**2)
    t2_zet = zet_ab / (gamma_val**2)
    t3_chi4 = chi_ab2**2 / (gamma_val**3)
    t3_zet2 = zet_ab**2 / (gamma_val**3)
    t3 = 1.918681e-03 * t3_chi4 - 2.032902e-03 * t3_zet2
    return torch.stack([t1, t2_chi, t2_zet, t3], dim=0)


def PW_alpha_c(rho_up: torch.Tensor, params: Mapping[str, Any], *, dens_threshold: float = 1e-15) -> torch.Tensor:
    params_a_pp = params["PW/params_a_pp"]
    params_a_a = params["PW/params_a_a"]
    params_a_alpha1 = params["PW/params_a_alpha1"]
    params_a_beta1 = params["PW/params_a_beta1"]
    params_a_beta2 = params["PW/params_a_beta2"]
    params_a_beta3 = params["PW/params_a_beta3"]
    params_a_beta4 = params["PW/params_a_beta4"]
    params_a_fz20 = params["PW/params_a_fz20"]

    rho_total = torch.where(rho_up < dens_threshold, dens_threshold, rho_up)
    zeta = 1.0
    rs = (3.0 / (4.0 * math.pi) / rho_total) ** (1.0 / 3.0)

    def f_zeta(z: float) -> torch.Tensor:
        return (opz_pow_n(rs.new_tensor(z), 4.0 / 3.0) + opz_pow_n(rs.new_tensor(-z), 4.0 / 3.0) - 2.0) / (
            2.0 ** (4.0 / 3.0) - 2.0
        )

    def g_aux(idx: int, rs_vals: torch.Tensor) -> torch.Tensor:
        return (
            params_a_beta1[idx] * torch.sqrt(rs_vals)
            + params_a_beta2[idx] * rs_vals
            + params_a_beta3[idx] * rs_vals ** 1.5
            + params_a_beta4[idx] * rs_vals ** (params_a_pp[idx] + 1.0)
        )

    def g(k: int, rs_vals: torch.Tensor) -> torch.Tensor:
        idx = k - 1
        return -2.0 * params_a_a[idx] * (1.0 + params_a_alpha1[idx] * rs_vals) * torch.log(
            1.0 + 1.0 / (2.0 * params_a_a[idx] * g_aux(idx, rs_vals))
        )

    fz = f_zeta(zeta)
    return g(1, rs) + zeta**4 * fz * (g(2, rs) - g(1, rs) + g(3, rs) / params_a_fz20) - fz * g(3, rs) / params_a_fz20


def PW_mod_c(rho: torch.Tensor, params: Mapping[str, Any], *, dens_threshold: float = 1e-15) -> torch.Tensor:
    """PW92 (PW_mod) correlation energy per particle with libxc-like screening."""
    params_a_pp = _to_tensor(params["PW/params_a_pp"], like=rho)
    params_a_a = _to_tensor(params["PW/params_a_a"], like=rho)
    params_a_alpha1 = _to_tensor(params["PW/params_a_alpha1"], like=rho)
    params_a_beta1 = _to_tensor(params["PW/params_a_beta1"], like=rho)
    params_a_beta2 = _to_tensor(params["PW/params_a_beta2"], like=rho)
    params_a_beta3 = _to_tensor(params["PW/params_a_beta3"], like=rho)
    params_a_beta4 = _to_tensor(params["PW/params_a_beta4"], like=rho)
    params_a_fz20 = _to_tensor(params["PW/params_a_fz20"], like=rho)

    rho_up_raw = rho[0, 0, :]
    rho_down_raw = rho[1, 0, :]
    rho_total_raw = rho_up_raw + rho_down_raw

    masked = rho_total_raw < dens_threshold

    rho_up = torch.where(masked, torch.zeros_like(rho_up_raw), rho_up_raw.clamp_min(dens_threshold))
    rho_down = torch.where(masked, torch.zeros_like(rho_down_raw), rho_down_raw.clamp_min(dens_threshold))
    rho_total = rho_up + rho_down

    rho_total_safe = torch.where(masked, torch.ones_like(rho_total), rho_total)
    zeta = (rho_up - rho_down) / rho_total_safe
    rs = (3.0 / (4.0 * math.pi) / rho_total_safe) ** (1.0 / 3.0)

    def f_zeta(z: torch.Tensor) -> torch.Tensor:
        return (opz_pow_n(z, 4.0 / 3.0) + opz_pow_n(-z, 4.0 / 3.0) - 2.0) / (2.0 ** (4.0 / 3.0) - 2.0)

    def g_aux(idx: int, rs_vals: torch.Tensor) -> torch.Tensor:
        return (
            params_a_beta1[idx] * torch.sqrt(rs_vals)
            + params_a_beta2[idx] * rs_vals
            + params_a_beta3[idx] * rs_vals ** 1.5
            + params_a_beta4[idx] * rs_vals ** (params_a_pp[idx] + 1.0)
        )

    def g(k: int, rs_vals: torch.Tensor) -> torch.Tensor:
        idx = k - 1
        return -2.0 * params_a_a[idx] * (1.0 + params_a_alpha1[idx] * rs_vals) * torch.log(
            1.0 + 1.0 / (2.0 * params_a_a[idx] * g_aux(idx, rs_vals))
        )

    fz = f_zeta(zeta)
    eps_c = g(1, rs) + zeta**4 * fz * (g(2, rs) - g(1, rs) + g(3, rs) / params_a_fz20) - fz * g(3, rs) / params_a_fz20
    return torch.where(masked, torch.zeros_like(eps_c), eps_c)


_PBE_GAMMA = float((1.0 - math.log(2.0)) / (math.pi**2))
_PBE_BETA = 0.06672455060314922
_PBE_CT = float(((1.0 / 12.0) * 3.0 ** (5.0 / 6.0) * math.pi ** (1.0 / 6.0)) ** 2)


def _phi(zeta: torch.Tensor, *, zeta_threshold: float = 1e-12) -> torch.Tensor:
    return 0.5 * (
        opz_pow_n(zeta, 2.0 / 3.0, zeta_threshold=zeta_threshold)
        + opz_pow_n(-zeta, 2.0 / 3.0, zeta_threshold=zeta_threshold)
    )


def PBE_corr_c_new(
    rho: torch.Tensor,
    PW: torch.Tensor,
    params: Mapping[str, Any],
    *,
    dens_threshold: float = 1e-12,
    sigma_floor: float = 1e-32,
) -> torch.Tensor:
    """PBE correlation (GGA_C_PBE) energy per particle with libxc-like screening."""
    del PW  # kept for signature parity with the JAX code

    gamma = _to_tensor(params["fH/params_a_gamma"], like=rho)
    beta_gamma = _to_tensor(params["fH/params_a_beta"], like=rho) / gamma

    rho_up_raw = rho[0, 0, :]
    rho_down_raw = rho[1, 0, :]
    rho_total_raw = rho_up_raw + rho_down_raw

    masked = rho_total_raw < dens_threshold

    rho_up = torch.where(masked, torch.zeros_like(rho_up_raw), rho_up_raw.clamp_min(dens_threshold))
    rho_down = torch.where(masked, torch.zeros_like(rho_down_raw), rho_down_raw.clamp_min(dens_threshold))
    rho_total = rho_up + rho_down
    rho_total_safe = torch.where(masked, torch.ones_like(rho_total), rho_total)

    zeta = (rho_up - rho_down) / rho_total_safe
    phi = _phi(zeta)
    phi2 = phi * phi
    phi3 = phi2 * phi

    grad_u = rho[0, 1:4, :]
    grad_d = rho[1, 1:4, :]
    sigma_uu = torch.sum(grad_u * grad_u, dim=0).clamp_min(sigma_floor)
    sigma_dd = torch.sum(grad_d * grad_d, dim=0).clamp_min(sigma_floor)
    sigma_ud = torch.sum(grad_u * grad_d, dim=0)
    sigma = (sigma_uu + sigma_dd + 2.0 * sigma_ud).clamp_min(0.0)

    ct = torch.as_tensor(_PBE_CT, dtype=rho.dtype, device=rho.device)
    d2 = ct * sigma / (phi2 * rho_total_safe ** (7.0 / 3.0))

    eps_lda = PW_mod_c(rho, params, dens_threshold=dens_threshold)

    eps_safe = torch.where(masked, torch.full_like(eps_lda, -1.0), eps_lda)
    d2_safe = torch.where(masked, torch.zeros_like(d2), d2)
    phi3_safe = torch.where(masked, torch.ones_like(phi3), phi3)

    def A(eps: torch.Tensor, u3: torch.Tensor) -> torch.Tensor:
        return beta_gamma / torch.expm1(-eps / (gamma * u3))

    d2A = d2_safe * A(eps_safe, phi3_safe)
    h = gamma * phi3_safe * torch.log(
        1.0 + beta_gamma * d2_safe * (1.0 + d2A) / (1.0 + d2A * (1.0 + d2A))
    )
    exc = eps_lda + h
    return torch.where(masked, torch.zeros_like(exc), exc)


def PBE_pw_c(
    rho: torch.Tensor,
    PW: torch.Tensor,
    params: Mapping[str, Any],
    *,
    pbe_dens_threshold: float = 1e-12,
    sigma_floor: float = 1e-32,
    pw_dens_threshold: float = 1e-15,
) -> torch.Tensor:
    """Match PySCF's `eval_xc(',PBE') - eval_xc(',PW_mod')` pointwise."""
    del pw_dens_threshold  # kept for signature parity with the JAX code
    pbe = PBE_corr_c_new(rho, PW, params, dens_threshold=pbe_dens_threshold, sigma_floor=sigma_floor)
    return pbe - PW

