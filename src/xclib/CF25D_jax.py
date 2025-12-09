from .numint_tools_jax import *
# import numpy as np
import jax
import jax.numpy as jnp
@jax.jit
def int_xc_jax(params,rho,weight,ee): 
    # rho = np.float128(rho)
    xa = xf(rho[0])
    xb = xf(rho[1])
    xab = jnp.sqrt((xa**2+xb**2))
    vx_a = vxf(rho[0],wx=2.50)
    ux_a = uxf(xa,gamma=0.004)
    w_a = wf(rho[0])
    ex_a = exf(rho[0])
    vx_b = vxf(rho[1],wx=2.50)
    ux_b = uxf(xb,gamma=0.004)
    w_b = wf(rho[1])
    ex_b = exf(rho[1])
    for i in range(4):
        for j in range(4-i):
            for k in range(6-i-j):
                ee.append(E_ijk(rho[0],ex_a,vx_a,ux_a,w_a,i,j,k,weight)+E_ijk(rho[1],ex_b,vx_b,ux_b,w_b,i,j,k,weight))
    LSDA = PW_mod_c(rho,params)
    for i in range(9):
        ee.append((rho[0,0]*LSDA*w_a**i+rho[1,0]*LSDA*w_b**i).dot(weight))
    PBE_LSDA = PBE_pw_c(rho,LSDA,params)
    for i in range(9):
        ee.append((rho[0,0]*PBE_LSDA*w_a**i+rho[1,0]*PBE_LSDA*w_b**i).dot(weight))
    rho_a,rho_b = rho[:,0]
    UEGab = LSDA
    UEGa = PW_alpha_c(rho_a,params)
    UEGb = PW_alpha_c(rho_b,params)
    zeta = zetf(rho[0])
    zetb = zetf(rho[1])
    zetab = zeta+zetb
    h_anti = h_function(xa**2+xb**2,zetab)
    ha = h_function(xa**2,zeta,alpha=0.00515088)
    hb = h_function(xb**2,zetb,alpha=0.00515088)
    Da = Dsigma(rho[0])
    Db = Dsigma(rho[1])
    for value in h_anti:
        ee.append(((UEGab*(rho_a+rho_b)-UEGa*rho_a-UEGb*rho_b)*value).dot(weight))
    for a,b in zip(ha,hb):
        ee.append((UEGa*rho_a*a*Da+UEGb*rho_b*b*Db).dot(weight))
    ucab = uxf(xab,0.0031)
    for i in range(5):
        ee.append(((UEGab*(rho_a+rho_b)-UEGa*rho_a-UEGb*rho_b)*ucab**i).dot(weight))
    ua = uxf(xa,0.06)
    ub = uxf(xb,0.06)
    for i in range(5):
        ee.append((UEGa*rho_a*ua**i*Da+UEGb*rho_b*ub**i*Db).dot(weight))
    return ee
def cache_xc_kernel1(mf_hf, spin=1, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc. Note dm the zeroth order density matrix must be a
    hermitian matrix.
    '''
    ni, mol, grids, dm = mf_hf._numint,mf_hf.mol,mf_hf.grids,mf_hf.make_rdm1()
    ao_deriv = 2
    xctype = 'MGGA'
    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, True, grids)
    if dm[0].ndim == 1:  # RKS
        rho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            # print(ao.dtype)
            rho.append(make_rho(0, ao, mask, xctype))
        rho = jnp.hstack(rho)
        if spin == 1:  # RKS with nr_rks_fxc_st
            rho *= .5
            rho = jnp.repeat(rho[jnp.newaxis], 2, axis=0)
    else:  # UKS
        assert dm[0].ndim == 2
        # assert spin == 1
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa.append(make_rho(0, ao, mask, xctype))
            rhob.append(make_rho(1, ao, mask, xctype))
        rho = jnp.array([jnp.hstack(rhoa), jnp.hstack(rhob)])
    # vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
    return rho