import jax.numpy as jnp
def vxf(rho,wx):
    return wx*rho[0]**(1.0/3.0)/(1+wx*rho[0]**(1/3))
def xf(rho):
    return rho[0]**(-4.0/3.0)*jnp.linalg.norm(rho[1:4],axis=0)
def uxf(x,gamma):
    return gamma*x**2/(1+gamma*x**2)
def yf(rho):
    return (3/5)*(6*jnp.pi**2)**(2/3)*rho[0]**(5/3)/rho[-1]/2
def wf(rho):
    return (yf(rho)-1)/(yf(rho)+1)
def wf_total(rho):
    rho_total = rho[:,0].sum(axis=0)
    t_total = rho[:,-1].sum(axis=0)
    y_total = 3.0/10.0*(3*jnp.pi**2)**(2.0/3.0)*rho_total**(5.0/3.0)/(t_total)
    return (y_total-1)/(y_total+1)
def exf(rho):
    return -(3/2)*(3/4/jnp.pi)**(1/3)*rho[0]**(1/3)
def E_ijk(rho,ex,vx,ux,wx,i,j,k,weight,aijk=1.0):
    return jnp.dot(weight,rho[0]*ex*aijk*vx**i*ux**j*wx**k)
def ucf(x,gamma):
    return gamma*x**2/(1+gamma*x**2)
def Dsigma(rho):
    return 1.0 - 0.125 * jnp.linalg.norm(rho[1:4],axis=0)**2 / (rho[0] * rho[-1])
def zetf(rho):
    CF = 0.3 * (3.0 * jnp.pi**2)**(2.0/3.0)  # Thomas-Fermi常数
    scalefactorTFconst = 3.17480210393640
    return 2.0 * rho[-1] / (rho[0]**(5.0/3.0)) - CF * scalefactorTFconst
def opz_pow_n(z, n, *, zeta_threshold=1e-12):
    """(1+z)^n with zeta screening (mirrors libxc maple/util.mpl style)."""
    return jnp.where(z + 1.0 <= zeta_threshold, zeta_threshold**n, (z + 1.0) ** n)
def h_function(chi_ab2,zet_ab,alpha=0.00304966):
    gamma_val = 1.0 + alpha * (chi_ab2 + zet_ab)
    t1 = 1.0 / gamma_val
    t2_chi = chi_ab2 / (gamma_val**2)  # χ²/γ²
    t2_zet = zet_ab / (gamma_val**2)   # ζ/γ²

    t3_chi4 = chi_ab2**2 / (gamma_val**3)              # χ⁴/γ³
    # t3_chi2zet = chi_ab2 * zet_ab / (gamma_val**3)     # χ²ζ/γ³
    t3_zet2 = zet_ab**2 / (gamma_val**3)               # ζ²/γ³

    return jnp.array([t1, t2_chi, t2_zet, 1.918681e-03*t3_chi4-2.032902e-03*t3_zet2])

def h_function(chi_ab2,zet_ab,alpha=0.00304966):
    gamma_val = 1.0 + alpha * (chi_ab2 + zet_ab)
    t1 = 1.0 / gamma_val
    t2_chi = chi_ab2 / (gamma_val**2)  # χ²/γ²
    t2_zet = zet_ab / (gamma_val**2)   # ζ/γ²

    t3_chi4 = chi_ab2**2 / (gamma_val**3)              # χ⁴/γ³
    # t3_chi2zet = chi_ab2 * zet_ab / (gamma_val**3)     # χ²ζ/γ³
    t3_zet2 = zet_ab**2 / (gamma_val**3)               # ζ²/γ³

    return jnp.array([t1, t2_chi, t2_zet, 1.918681e-03*t3_chi4-2.032902e-03*t3_zet2])
params = {   "PW/params_a_pp":[1,  1,  1],    
             "PW/params_a_a"      : [0.0310907, 0.01554535, 0.0168869],
             "PW/params_a_alpha1": [0.21370,  0.20548,  0.11125],
             "PW/params_a_beta1"  : [7.5957, 14.1189, 10.357],
             "PW/params_a_beta2"  : [3.5876, 6.1977, 3.6231],
             "PW/params_a_beta3"  : [1.6382, 3.3662,  0.88026],
             "PW/params_a_beta4"  : [0.49294, 0.62517, 0.49671],
             "PW/params_a_fz20"   : 1.709920934161365617563962776245,
             "fH/params_a_beta"   : 0.06672455060314922,
             "fH/params_a_gamma"  : (1 - jnp.log(2))/jnp.pi**2,
             "fH/params_a_BB"     : 1,
             "fH/params_a_tscale" : 1,
             "wx"                 : 2.50,
             "gamma"              : 0.004,
             "alpha"              : 0.00515088,
             'alpha_anti'         : 0.00304966}

def PW_alpha_c(rho_up,params,dens_threshold=1e-15):
    params_a_pp     = params["PW/params_a_pp"]
    params_a_a      = params["PW/params_a_a"]
    params_a_alpha1 = params["PW/params_a_alpha1"]
    params_a_beta1  = params["PW/params_a_beta1"]
    params_a_beta2  = params["PW/params_a_beta2"]
    params_a_beta3  = params["PW/params_a_beta3"]
    params_a_beta4  = params["PW/params_a_beta4"]
    params_a_fz20   = params["PW/params_a_fz20"]
    rho_total = rho_up 
    rho_total = jnp.where(rho_total < dens_threshold, dens_threshold, rho_total)
    zeta = 1
    rs = (3.0 / (4.0 * jnp.pi)/rho_total)**(1.0/3.0)
    def opz_pow_n(z,n):
        return jnp.where(z+1<=1e-12, 1e-12**n, (z+1)**n)
    def f_zeta(z):
        return (opz_pow_n(z,4/3) + opz_pow_n(-z,4/3) - 2)/(2**(4/3) - 2)
    def g_aux(k,rs):
        return params_a_beta1[k]*jnp.sqrt(rs) + params_a_beta2[k]*rs+ params_a_beta3[k]*rs**1.5 + params_a_beta4[k]*rs**(params_a_pp[k] + 1)
    def g(k, rs_vals):
        k-=1
        return -2*params_a_a[k]*(1 + params_a_alpha1[k]*rs)*jnp.log(1.0+1.0/(2.0*params_a_a[k]*g_aux(k, rs)))
    eps_c_per_particle =  g(1, rs) +zeta**4*f_zeta(zeta)*(g(2, rs) - g(1, rs) + g(3, rs)/params_a_fz20)-f_zeta(zeta)*g(3, rs)/params_a_fz20
    # exc_density = eps_c_per_particle
    return eps_c_per_particle



def opz_pow_n(z, n, *, zeta_threshold=1e-12):
    """(1+z)^n with zeta screening (mirrors libxc maple/util.mpl style)."""
    return jnp.where(z + 1.0 <= zeta_threshold, zeta_threshold**n, (z + 1.0) ** n)


def PW_mod_c(rho, params, *, dens_threshold=1e-15):
    """PW92 (PW_mod) correlation energy per particle with libxc-like screening.

    Key behaviors matched to PySCF/libxc (empirically):
      1) If (rho_up + rho_down) < dens_threshold => exc = 0
      2) Else: each spin density is floored to dens_threshold before computing
         rs and zeta (this avoids zeta=±1 and stabilizes low-density behavior)

    Parameters
    ----------
    rho : array, shape (2,6,N)
        UKS-like rho; only rho[:,0,:] is used here.
    params : dict
        Must contain PW parameter arrays used by the original implementation.
    dens_threshold : float
        Density screening threshold; libxc uses ~1e-15 for PW_mod in PySCF builds.
    """
    params_a_pp = jnp.asarray(params["PW/params_a_pp"])
    params_a_a = jnp.asarray(params["PW/params_a_a"])
    params_a_alpha1 = jnp.asarray(params["PW/params_a_alpha1"])
    params_a_beta1 = jnp.asarray(params["PW/params_a_beta1"])
    params_a_beta2 = jnp.asarray(params["PW/params_a_beta2"])
    params_a_beta3 = jnp.asarray(params["PW/params_a_beta3"])
    params_a_beta4 = jnp.asarray(params["PW/params_a_beta4"])
    params_a_fz20 = params["PW/params_a_fz20"]

    rho_up_raw = rho[0, 0, :]
    rho_down_raw = rho[1, 0, :]
    rho_total_raw = rho_up_raw + rho_down_raw

    masked = rho_total_raw < dens_threshold

    # libxc-like screening: floor spin densities once total density is non-negligible
    rho_up = jnp.where(masked, 0.0, jnp.maximum(rho_up_raw, dens_threshold))
    rho_down = jnp.where(masked, 0.0, jnp.maximum(rho_down_raw, dens_threshold))
    rho_total = rho_up + rho_down

    rho_total_safe = jnp.where(masked, 1.0, rho_total)
    zeta = (rho_up - rho_down) / rho_total_safe
    rs = (3.0 / (4.0 * jnp.pi) / rho_total_safe) ** (1.0 / 3.0)

    def f_zeta(z):
        return (opz_pow_n(z, 4.0 / 3.0) + opz_pow_n(-z, 4.0 / 3.0) - 2.0) / (2.0 ** (4.0 / 3.0) - 2.0)

    def g_aux(k, rs_vals):
        return (
            params_a_beta1[k] * jnp.sqrt(rs_vals)
            + params_a_beta2[k] * rs_vals
            + params_a_beta3[k] * rs_vals ** 1.5
            + params_a_beta4[k] * rs_vals ** (params_a_pp[k] + 1.0)
        )

    def g(k, rs_vals):
        k -= 1
        return (
            -2.0
            * params_a_a[k]
            * (1.0 + params_a_alpha1[k] * rs_vals)
            * jnp.log(1.0 + 1.0 / (2.0 * params_a_a[k] * g_aux(k, rs_vals)))
        )

    fz = f_zeta(zeta)
    eps_c = (
        g(1, rs)
        + zeta**4 * fz * (g(2, rs) - g(1, rs) + g(3, rs) / params_a_fz20)
        - fz * g(3, rs) / params_a_fz20
    )
    return jnp.where(masked, 0.0, eps_c)


_PBE_GAMMA = float((1.0 - jnp.log(2.0)) / (jnp.pi**2))
_PBE_BETA = 0.06672455060314922
_PBE_CT = float(((1.0 / 12.0) * 3.0 ** (5.0 / 6.0) * jnp.pi ** (1.0 / 6.0)) ** 2)

def _phi(zeta, *, zeta_threshold=1e-12):
    return 0.5 * (
        opz_pow_n(zeta, 2.0 / 3.0, zeta_threshold=zeta_threshold)
        + opz_pow_n(-zeta, 2.0 / 3.0, zeta_threshold=zeta_threshold)
    )


def PBE_corr_c_new(rho,PW, params, *, dens_threshold=1e-12, sigma_floor=1e-32):
    """PBE correlation (GGA_C_PBE) energy per particle with libxc-like screening."""
    dtype = rho.dtype
    gamma = jnp.asarray(params["fH/params_a_gamma"], dtype=dtype)
    beta_gamma = jnp.asarray(params['fH/params_a_beta'] / params["fH/params_a_gamma"], dtype=dtype)

    rho_up_raw = rho[0, 0, :]
    rho_down_raw = rho[1, 0, :]
    rho_total_raw = rho_up_raw + rho_down_raw

    masked = rho_total_raw < dens_threshold

    # libxc-like density floor for PBE: once total density is non-negligible,
    # floor each spin-density to dens_threshold before computing zeta/phi.
    rho_up = jnp.where(masked, 0.0, jnp.maximum(rho_up_raw, dens_threshold))
    rho_down = jnp.where(masked, 0.0, jnp.maximum(rho_down_raw, dens_threshold))
    rho_total = rho_up + rho_down
    rho_total_safe = jnp.where(masked, 1.0, rho_total)

    zeta = (rho_up - rho_down) / rho_total_safe
    phi = _phi(zeta)
    phi2 = phi * phi
    phi3 = phi2 * phi

    # libxc-like sigma floors are applied per-spin channel.
    grad_u = rho[0, 1:4, :]
    grad_d = rho[1, 1:4, :]
    sigma_uu = jnp.sum(grad_u * grad_u, axis=0)
    sigma_dd = jnp.sum(grad_d * grad_d, axis=0)
    sigma_ud = jnp.sum(grad_u * grad_d, axis=0)

    sigma_uu = jnp.maximum(sigma_uu, sigma_floor)
    sigma_dd = jnp.maximum(sigma_dd, sigma_floor)
    sigma = sigma_uu + sigma_dd + 2.0 * sigma_ud
    sigma = jnp.maximum(sigma, 0.0)

    # Reduced-gradient parameter t^2 (called d2 here) used in the PBE H-term.
    ct = jnp.asarray(_PBE_CT, dtype=dtype)
    d2 = ct * sigma / (phi2 * rho_total_safe ** (7.0 / 3.0))

    eps_lda = PW_mod_c(rho, params, dens_threshold=dens_threshold)
    # eps_lda = PW

    # Avoid 0/0 in the masked branch: eps_lda==0 makes A() singular.
    eps_safe = jnp.where(masked, -1.0, eps_lda)
    d2_safe = jnp.where(masked, 0.0, d2)
    phi3_safe = jnp.where(masked, 1.0, phi3)

    def A(eps, u3):
        return beta_gamma / jnp.expm1(-eps / (gamma * u3))

    d2A = d2_safe * A(eps_safe, phi3_safe)
    h = gamma * phi3_safe * jnp.log(
        1.0 + beta_gamma * d2_safe * (1.0 + d2A) / (1.0 + d2A * (1.0 + d2A))
    )
    exc = eps_lda + h
    return jnp.where(masked, 0.0, exc)


def PBE_pw_c(rho,PW, params, *, pbe_dens_threshold=1e-12, sigma_floor=1e-32, pw_dens_threshold=1e-15):
    """Match PySCF's `eval_xc(',PBE') - eval_xc(',PW_mod')` pointwise."""
    pbe = PBE_corr_c_new(rho,PW, params, dens_threshold=pbe_dens_threshold, sigma_floor=sigma_floor)
    # pw = PW_mod_c(rho, params, dens_threshold=pw_dens_threshold)
    return pbe - PW