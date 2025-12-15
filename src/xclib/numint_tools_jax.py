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
params = {"PW/params_a_pp":[1,  1,  1],    
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
             "fH/params_a_tscale" : 1,}
def PW_mod_c(rho,params):
    params_a_pp     = params["PW/params_a_pp"]
    params_a_a      = params["PW/params_a_a"]
    params_a_alpha1 = params["PW/params_a_alpha1"]
    params_a_beta1  = params["PW/params_a_beta1"]
    params_a_beta2  = params["PW/params_a_beta2"]
    params_a_beta3  = params["PW/params_a_beta3"]
    params_a_beta4  = params["PW/params_a_beta4"]
    params_a_fz20   = params["PW/params_a_fz20"]
    rho_up = rho[0, 0, :]    # α自旋密度
    rho_down = rho[1, 0, :]  # β自旋密度
    rho_total = rho_up + rho_down
    # rho_total = jnp.where(rho_total < 1e-12, 1e-12, rho_total)
    zeta = (rho_up - rho_down) / rho_total
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
def PW_alpha_c(rho_up,params):
    params_a_pp     = params["PW/params_a_pp"]
    params_a_a      = params["PW/params_a_a"]
    params_a_alpha1 = params["PW/params_a_alpha1"]
    params_a_beta1  = params["PW/params_a_beta1"]
    params_a_beta2  = params["PW/params_a_beta2"]
    params_a_beta3  = params["PW/params_a_beta3"]
    params_a_beta4  = params["PW/params_a_beta4"]
    params_a_fz20   = params["PW/params_a_fz20"]
    rho_total = rho_up 
    # rho_total = jnp.where(rho_total < 1e-12, 1e-12, rho_total)
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
def PBE_pw_c(rho,xc_PW,params):
    PI2 = jnp.pi ** 2
    # # PBE 相关常数（与 XCFun 中一致）
    param_gamma = params["fH/params_a_gamma"]
    param_beta_accurate = params["fH/params_a_beta"]
    param_beta_gamma = param_beta_accurate / param_gamma
    rho_up = rho[0, 0, :]    # α自旋密度
    rho_down = rho[1, 0, :]  # β自旋密度
    rho_total = rho_up + rho_down
    # print(rho_total)
    zeta = (rho_up - rho_down) / rho_total
    rs = (3.0 / (4.0 * jnp.pi)/rho_total)**(1.0/3.0)
    gnn = jnp.sum(jnp.sum(rho[:,1:4],axis=0)**2,axis=0)
    def A(eps, u3):
        return param_beta_gamma / jnp.expm1(-eps / (param_gamma * u3))
    def H(d2, eps, u3):
        d2A = d2 * A(eps, u3)
        return param_gamma * u3 * jnp.log(
            1.0 + param_beta_gamma * d2 * (1.0 + d2A)
            / (1.0 + d2A * (1.0 + d2A))
        )
    def opz_pow_n(z,n):
        return jnp.where(z+1<=1e-12, 1e-12**n, (z+1)**n)
    def mphi(z) :
        return (opz_pow_n(z,2.0/3.0) + opz_pow_n(-z,2.0/3.0))/2.0
    eps_lda = PW_mod_c(rho,params)
    u = mphi(zeta)          # phi
    u3 = u**3
    C_t = ((1.0 / 12.0) * 3.0 ** (5.0 / 6.0) * jnp.pi ** (1.0 / 6.0)) ** 2
    d2 = C_t * gnn / (u * u * rho_total ** (7.0 / 3.0))
    return H(d2, eps_lda, u3)