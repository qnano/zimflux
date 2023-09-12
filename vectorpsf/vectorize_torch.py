import torch
import numpy as np
import matplotlib.pyplot as plt

import math
import time
from typing import List, Tuple, Union, Optional
from torch import Tensor
from config import z_stack_params
import tifffile
import napari


#@torch.jit.script
def compute_crlb(mu: Tensor, jac: Tensor, *, skip_axes: List[int] = []):
    """
    Compute crlb from expected value and per pixel derivatives.
    mu: [N, H, W]
    jac: [N, H,W, coords]
    """
    if not isinstance(mu, torch.Tensor):
        mu = torch.Tensor(mu)

    if not isinstance(jac, torch.Tensor):
        jac = torch.Tensor(jac)

    naxes = jac.shape[-1]
    axes = [i for i in range(naxes) if not i in skip_axes]
    jac = jac[..., axes]

    sample_dims = tuple(np.arange(1, len(mu.shape)))

    fisher = torch.matmul(jac[..., None], jac[..., None, :])  # derivative contribution
    fisher = fisher / mu[..., None, None]  # px value contribution
    fisher = fisher.sum(sample_dims)

    crlb = torch.zeros((len(mu), naxes), device=mu.device)
    crlb[:, axes] = torch.sqrt(torch.diagonal(torch.inverse(fisher), dim1=1, dim2=2))
    return crlb


class Model_vectorial_psf_IBg(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, NA:float, zvals, refmed:float, refcov:float, refimm:float, refimmnom:float, Lambda:float, Npupil:int, abberations, zmin:float,
                zmax:float, K:int, N:int,
                M:int, L:int, Ax, Bx,
                Dx, Ay, pixelsize:float, By, Dy, Mx:int, My:int, numparams_fit:int, thetatry, dev:str, zstack:bool,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix):

        mu, dmudtheta = poissonrate(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                                    zmax, K, N,
                                    M, L, Ax, Bx,
                                    Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, thetatry, dev, zstack,
                                    wavevector, wavevectorzimm, all_zernikes, PupilMatrix, ibg_ony=True) # numparamsfit = 2

        return mu, dmudtheta




class Model_vectorial_psf(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, NA:float, zvals, refmed:float, refcov:float, refimm:float, refimmnom:float, Lambda:float, Npupil:int, abberations, zmin:float,
                zmax:float, K:int, N:int,
                M:int, L:int, Ax, Bx,
                Dx, Ay, pixelsize:float, By, Dy, Mx:int, My:int, numparams_fit:int, thetatry, dev:str, zstack:bool,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix):


        mu, dmudtheta = poissonrate(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                                    zmax, K, N,
                                    M, L, Ax, Bx,
                                    Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, thetatry, dev, zstack,
                                    wavevector, wavevectorzimm, all_zernikes, PupilMatrix)

        return mu, dmudtheta

def sf_modulation(xyz:Tensor, mod:Tensor, roipos):
    k = mod[:, :, :3]
    phase = mod[:, :,4]
    depth = mod[:,:,3]
    relint = mod[:,:,5]
    xyz_mod = xyz * 1
    xyz_mod[:,:2] = roipos + xyz[:,0:2]
    em_phase = ((xyz_mod[:,None] * k).sum(-1))%(2*np.pi) - phase

    deriv = depth[:,:,None] * k * torch.cos(em_phase)[:,:,None] * relint[:,:,None] # [spots,patterns,coords]
    intensity = (1+depth * torch.sin(em_phase))*relint

    return intensity,deriv
import napari
def show_napari(mov):
    import napari

    with napari.gui_qt():
        napari.view_image(mov)

def show_napari_tensor(fit,mu):
    import napari
    fit = fit.detach().cpu().numpy()
    mu = mu.detach().cpu().numpy()
    with napari.gui_qt():
        napari.view_image(np.concatenate((fit, mu),axis=-1))

class Model_vectorial_psf_simflux(torch.nn.Module):
    def __init__(self, roisize):
        super().__init__()
        self.roisize = roisize
    def forward(self, NA: float, zvals, refmed: float, refcov: float, refimm: float, refimmnom: float, Lambda: float,
                Npupil: int, abberations, zmin: float,
                zmax: float, K: int, N: int,
                M: int, L: int, Ax, Bx,
                Dx, Ay, pixelsize: float, By, Dy, Mx: int, My: int, numparams_fit: int, thetatry, dev: str,
                zstack: bool,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix, mod, roipos, smp):

        # normalize psf params
        photons = thetatry[:,3]*1
        bg = thetatry[:,4]*1/3


        xy = (thetatry[:, [1,0]] - self.roisize / 2) * pixelsize
        z = thetatry[:, 2] * 1000
        xyz = torch.cat((xy,z[:,None]), dim=-1)
        # careful treat x and y are different for psf and simflux
        normalized_psf_theta = torch.cat((xyz, torch.ones(z.size())[:,None].to(dev),torch.zeros(z.size())[...,None].to(dev)), dim=-1)
        # thetatry[:,3] = 1
        # thetatry[:, 4] = 0
        psf_ev, psf_deriv = poissonrate(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                                    zmax, K, N,
                                    M, L, Ax, Bx,
                                    Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, normalized_psf_theta, dev, zstack,
                                    wavevector, wavevectorzimm, all_zernikes, PupilMatrix)

        # thetatry[:, 0:2] = thetatry[:, 0:2]/pixelsize + self.roisize / 2
        # thetatry[:, 2] = thetatry[:, 2] / 1000
        # thetatry[:, [1, 0]] = thetatry[:, [0, 1]]
        psf_deriv[:,:,:,[1,0]]=psf_deriv[:,:,:,[0,1]]*65
        psf_deriv[:,:,:, 2] = psf_deriv[:,:,:, 2]*1000
        mod_intensity, mod_deriv = sf_modulation(thetatry[:,:3], mod,roipos)

        # expand mu and dmudtheta using tile

        # mu_expanded = mu[:,None,...].repeat([1,8,1,1])
        # jac_expanded = dmudtheta[:,None,...].repeat([1,8,1,1,1])
        #
        #
        # # add simflux illumation
        # mu_sim = bg[:,None, None, None] + mod_intensity[:, :, None, None] * photons[:,None,None,None] * mu_expanded[:, :, :, :]
        #
        # deriv_xyz = photons[:,None,None,None,None] * (mod_deriv[:, :,None, None, :] * mu_expanded[..., None] +
        #                                mod_intensity[:, :, None, None, None] * jac_expanded[..., :3])
        # #show_napari(deriv_xyz[...,2].detach().cpu().numpy())
        # deriv_I = mu_expanded * mod_intensity[:, :, None, None]
        #
        # deriv_bg = torch.ones(deriv_I.shape, device=deriv_I.device)

        phot = photons[:, None, None, None]
        mu = bg[:, None, None,None] + mod_intensity[:, :, None, None] * phot * psf_ev[:, None, :, :]

        # mu_adj = bg[:, None, None] + mod_intensityadj[:, :, None, None] * phot * psf_evadj[:, None, :, :]
        #
        # dmu = (mu_adj - mu) / 0.0000001
        # dmu = dmu.cpu().detach().numpy()
        deriv_causedbyillu = mod_deriv[:, :, None, None,:] * psf_ev[:, None, :,:, None]
        deriv_causedbypsf = mod_intensity[:, :, None, None, None] * psf_deriv[:, None,:,:, :3]
        deriv_xyz = phot[..., None] * (mod_deriv[:, :, None, None] * psf_ev[:, None, ..., None] +
                                       mod_intensity[:, :, None, None, None] * psf_deriv[:, None, ..., :3])

        deriv_I = psf_ev[:, None] * mod_intensity[:, :, None, None]

        deriv_bg = torch.ones(deriv_I.shape, device=deriv_I.device)/3 # change to not a hard 3 (numpat)

        # add weigths to dmudthata
        #weigth = (smp - mu_sim) / (mu_sim)
        full_jacobian= torch.cat((deriv_xyz,
                   deriv_I[..., None],
                   deriv_bg[..., None]), -1)

        #jac_temp = weigth[..., None] * full_jacobian
        #jac_summed = jac_temp.sum(1)


        # return summed jacobian

        return mu, full_jacobian
       # return mu_expanded.type(torch.double), jac_expanded.type(torch.double)
@torch.jit.script
def likelihood_v2(image, mu, dmudtheta, simflux:bool):
    if simflux:
        sample_dims = [-3, -2, -1]
        sample_dimsjac = [-4, -3, -2]
    else:
        sample_dims = [-2, -1]
        sample_dimsjac = [-3, -2]
    varfit = 0
    # calculation of weight factors
    keps = 1e3 * 2.220446049250313e-16

    mupos = (mu > 0) * mu + (mu < 0) * keps

    weight = (image - mupos) / (mupos + varfit)
    dweight = (image + varfit) / (mupos + varfit) ** 2
    num_params = 5

    # log-likelihood, gradient vector and Hessian matrix
    logL = torch.sum((image + varfit) * torch.log(mupos + varfit) - (mupos + varfit), sample_dims)
    gradlogL = torch.sum(weight[..., None] * dmudtheta, sample_dimsjac)
    HessianlogL = torch.zeros((gradlogL.size(0), num_params, num_params))
    for ii in range(num_params):
        for jj in range(num_params):
            HessianlogL[:, ii, jj] = torch.sum(-dweight * dmudtheta[..., ii] * dmudtheta[..., jj], sample_dims)

    return logL, gradlogL, HessianlogL

@torch.jit.script
def MLE_instead_lmupdate(cur, mu, jac, smp, lambda_: float, param_range_min_max, simflux:bool, scale_old=torch.Tensor(1).to('cuda')):
    """
    Separate some of the calculations to speed up with jit script
    """

    merit, grad, Hessian = likelihood_v2(smp, mu, jac, simflux)
    diag = torch.diagonal(Hessian, dim1=-2, dim2=-1)
    b = torch.eye(diag.size(1))

    c = diag.unsqueeze(2).expand(diag.size(0),diag.size(1),diag.size(1))

    diag_full = c * b
    # matty = Hessian + lambda_ * diag_full

    # thetaupdate
    # update of fit parameters via Levenberg-Marquardt
    Bmat = Hessian + lambda_ * diag_full
    Bmat = Bmat.to(device='cuda')

    dtheta = torch.linalg.solve(-Bmat, grad)

    dtheta[torch.isnan(dtheta)] = -0.1 * cur[torch.isnan(dtheta)]
    cur = cur + dtheta

    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))

    scale = 1
    return cur, scale


def MLE_instead_lmupdate_nonjit(cur, mu, jac, smp, lambda_: float, param_range_min_max, simflux: bool,
                         scale_old=torch.Tensor(1).to('cuda')):
    """
    Separate some of the calculations to speed up with jit script
    """

    merit, grad, Hessian = likelihood_v2(smp, mu, jac, simflux)
    diag = torch.diagonal(Hessian, dim1=-2, dim2=-1)
    b = torch.eye(diag.size(1))

    c = diag.unsqueeze(2).expand(diag.size(0), diag.size(1), diag.size(1))

    diag_full = c * b
    # matty = Hessian + lambda_ * diag_full

    # thetaupdate
    # update of fit parameters via Levenberg-Marquardt
    Bmat = Hessian + lambda_ * diag_full
    Bmat = Bmat.to(device='cuda')
    try:
        dtheta = torch.linalg.solve(-Bmat, grad)
    except:
        dtheta = -0.1 * cur
    dtheta[torch.isnan(dtheta)] = -0.1 * cur[torch.isnan(dtheta)]
    cur = cur + dtheta

    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))

    scale = 1
    return cur, scale


class LM_MLE_simflux(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations: int, lambda_: float, tol):
        """
        model:
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()
        self.tol = tol
        self.model = model
        self.iterations = int(iterations)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)

    def forward(self, smp, NA: float, zvals, refmed: float, refcov: float, refimm: float, refimmnom: float,
                Lambda: float, Npupil: int, abberations, zmin: float,
                zmax: float, K: int, N: int,
                M: int, L: int, Ax, Bx,
                Dx, Ay, pixelsize: float, By, Dy, Mx: int, My: int, numparams_fit: int, thetatry, dev: str,
                zstack: bool,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix, mod, roipos, ibg_only: bool = False):
        cur = (thetatry * 1).type(torch.float)
        smp = smp.type(torch.float)
        mod = mod.type(torch.float)
        roipos = roipos.type(torch.float)
        mu = torch.zeros(smp.size()).to(dev).type(torch.float)
        jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2], smp.size()[3], cur.size()[1])).to(dev).type(torch.float)
        scale = torch.zeros(cur.size(), device=dev).type(torch.float)
        if ibg_only:
            jac = torch.zeros((smp.size()[0], smp.size()[1], smp.size()[2], 2)).to(dev)
            scale = torch.zeros((cur.size()[0], 2), device=dev)

        traces = torch.zeros((self.iterations + 1, cur.size()[0], cur.size()[1]), device=dev).type(torch.float)
        traces[0, :, :] = cur

        tol = torch.ones((cur.size()[0], cur.size()[1])).to(dev) * self.tol[None,...].repeat([cur.size()[0],1])
        good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
        delta = torch.ones(cur.size()).to(dev).type(torch.float)
        bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)

        i = 0
        flag_tolerance = 0

        while (i < self.iterations) and (flag_tolerance == 0):

            mu[good_array, :, :,:], jac[good_array, :, :, :,:] = self.model(NA, zvals, refmed, refcov, refimm, refimmnom,
                                                                        Lambda, Npupil, abberations, zmin,
                                                                        zmax, K, N,
                                                                        M, L, Ax, Bx,
                                                                        Dx, Ay, pixelsize, By, Dy, Mx, My,
                                                                        numparams_fit, cur[good_array, :], dev, zstack,
                                                                        wavevector, wavevectorzimm, all_zernikes,
                                                                        PupilMatrix, mod[good_array,:], roipos[good_array,:], smp[good_array,...])



            if ibg_only:
                cur[good_array, 3:5], scale[good_array, :] = lm_update(cur[good_array, 3:5], mu[good_array, :, :],
                                                                       jac[good_array, :, :, :], smp[good_array, :, :],
                                                                       self.lambda_, self.param_range_min_max[3:5, :],
                                                                       scale[good_array, :])
            else:

                cur[good_array, :], scale[good_array, :] = MLE_instead_lmupdate_nonjit(cur[good_array, :], mu[good_array,...],
                                                                       jac[good_array, :, :,: :],  smp[good_array,...],
                                                                     self.lambda_, self.param_range_min_max,True,
                                                                     scale[good_array, :])
                # cur[good_array, :], scale[good_array, :] = lm_update(cur[good_array, :], mu[good_array, ...],
                #                                                                 jac[good_array, :, :, ::],
                #                                                                 smp[good_array, ...],
                #                                                                 self.lambda_, self.param_range_min_max,
                #                                                                  scale[good_array, :])


            traces[i + 1, good_array, :] = cur[good_array, :]
            delta[good_array, :] = torch.absolute(traces[i - 1, good_array, :] - traces[i, good_array, :])

            bool_array[good_array] = (delta[good_array, :] < tol[good_array, :]).type(torch.bool)
            test = torch.sum(bool_array, dim=1)
            good_array = test != 5

            # loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))
            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        return cur, traces, (smp - mu) ** 2 / mu



class LM_MLE(torch.nn.Module):
    def __init__(self, model, param_range_min_max, iterations: int, lambda_: float, tol: float):
        """
        model:
            module that takes parameter array [batchsize, num_parameters]
            and returns (expected_value, jacobian).
            Expected value has shape [batchsize, sample dims...]
            Jacobian has shape [batchsize, sample dims...,num_parameters]
        """
        super().__init__()
        self.tol = tol
        self.model = model
        self.iterations = int(iterations)

        # if not isinstance(param_range_min_max, torch.Tensor):
        #    param_range_min_max = torch.Tensor(param_range_min_max).to(smp.device)

        self.param_range_min_max = param_range_min_max
        self.lambda_ = float(lambda_)
    def forward(self, smp, NA:float, zvals, refmed:float, refcov:float, refimm:float, refimmnom:float, Lambda:float, Npupil:int, abberations, zmin:float,
                zmax:float, K:int, N:int,
                M:int, L:int, Ax, Bx,
                Dx, Ay, pixelsize:float, By, Dy, Mx:int, My:int, numparams_fit:int, thetatry, dev:str, zstack:bool,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix, ibg_only:bool = False):
        cur = thetatry * 1
        mu = torch.zeros(smp.size()).to(dev)
        jac = torch.zeros((smp.size()[0], smp.size()[1],smp.size()[2], cur.size()[1])).to(dev)
        scale = torch.zeros(cur.size(), device=dev)
        if ibg_only:
            jac = torch.zeros((smp.size()[0], smp.size()[1],smp.size()[2], 2)).to(dev)
            scale = torch.zeros((cur.size()[0],2), device=dev)

        traces = torch.zeros((self.iterations+1, cur.size()[0], cur.size()[1]),device = dev).type(torch.float)
        traces[0,:,:] = cur

        tol = torch.ones((cur.size()[0],cur.size()[1])).to(dev) * self.tol
        good_array = torch.ones(cur.size()[0]).to(dev).type(torch.bool)
        delta = torch.ones(cur.size()).to(dev).type(torch.float)
        bool_array = torch.ones(cur.size()).to(dev).type(torch.bool)

        i=0
        flag_tolerance = 0
        while (i < self.iterations) and (flag_tolerance== 0):

            mu[good_array,:,:], jac[good_array,:,:,:] = self.model( NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                zmax, K, N,
                M, L, Ax, Bx,
                Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, cur[good_array,:], dev, zstack,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
            if ibg_only:
                cur[good_array,3:5], scale[good_array,:] = lm_update(cur[good_array,3:5], mu[good_array,:,:], jac[good_array,:,:,:], smp[good_array,:,:], self.lambda_, self.param_range_min_max[3:5,:], scale[good_array,:])
            else:
                cur[good_array, :], scale[good_array, :] = MLE_instead_lmupdate(cur[good_array, :], mu[good_array, :, :],
                                                                     jac[good_array, :, :, :], smp[good_array, :, :],
                                                                     self.lambda_, self.param_range_min_max, False,
                                                                     scale[good_array, :])

            traces[i+1,good_array,:] = cur[good_array,:]
            delta[good_array,:] =   torch.absolute(traces[i-1,good_array,:] - traces[i,good_array,:])

            bool_array[good_array] = (delta[good_array,:]<tol[good_array,:]).type(torch.bool)
            test = torch.sum(bool_array,dim=1)
            good_array = test !=5



            #loglik = torch.sum(smp * torch.log(mu / smp), dim=(1, 2)) - torch.sum(mu - smp, dim=(1, 2))
            if torch.sum(good_array) == 0:
                flag_tolerance = 1
            i = i + 1
        # if chi:
        #     return cur, loglik, (smp - mu)**2/mu
        # else:

        return cur, traces, (smp - mu) ** 2 / mu


@torch.jit.script
def lm_alphabeta(mu, jac, smp):
    """
    mu: [batchsize, numsamples]
    jac: [batchsize, numsamples, numparams]
    smp: [batchsize, numsamples]
    """
    # assert np.array_equal(smp.shape, mu.shape)


    sampledims = [i for i in range(1, len(smp.shape))]

    invmu = 1.0 / torch.clip(mu, min=1e-9)
    af = smp * invmu ** 2

    jacm = torch.matmul(jac[..., None], jac[..., None, :])
    alpha = jacm * af[..., None, None]
    alpha = alpha.sum(sampledims)

    beta = (jac * (smp * invmu - 1)[..., None]).sum(sampledims)
    return alpha, beta


#@torch.jit.script
def lm_update(cur, mu, jac, smp, lambda_: float, param_range_min_max, scale_old=torch.Tensor(1).to('cuda')):
    """
    Separate some of the calculations to speed up with jit script
    """
    alpha, beta = lm_alphabeta(mu, jac, smp)
    scale_old = scale_old.to(device=cur.device)
    K = cur.shape[-1]

    if True:  # scale invariant. Helps when parameter scales are quite different
        # For a matrix A, (element wise A*A).sum(0) is the same as diag(A^T * A)

        scale = (alpha * alpha).sum(1)
        # scale /= scale.mean(1, keepdim=True) # normalize so lambda scale is not model dependent

        if (scale_old.size() != torch.Size([1])) and K !=1 :
            scale = torch.maximum(scale, scale_old)

        # assert torch.isnan(scale).sum()==0


        alpha += lambda_ * scale[:, :, None] * torch.eye(K, device=smp.device)[None]
    else:
        # regular LM, non scale invariant
        alpha += lambda_ * torch.eye(K, device=smp.device)[None]

    steps = torch.linalg.solve(alpha, beta)

    cur[torch.isnan(cur)] = 0
    steps[torch.isnan(steps)] = -0.1
    # assert torch.isnan(cur).sum() == 0
    # assert torch.isnan(steps).sum() == 0

    cur = cur + steps
    # if Tensor.dim(param_range_min_max) == 2:
    cur = torch.maximum(cur, param_range_min_max[None, :, 0].to(cur.device))
    cur = torch.minimum(cur, param_range_min_max[None, :, 1].to(cur.device))
    if K ==1:
        cur = cur[:,0]
    # elif Tensor.dim(param_range_min_max) == 3:
    #
    # cur = torch.maximum(cur, param_range_min_max[:, :, 0])
    # cur = torch.minimum(cur, param_range_min_max[:, :, 1])
    # else:
    #     raise 'check bounds'
    if scale_old.size() != torch.Size([1]):
        return cur, scale
    else:
        return cur, scale





def thetalimits(abberations,Lambda, Mx, My,pixelsize,zspread, dev, zstack= False):


    zernikecoefsmax = 0.25*Lambda*torch.ones((abberations.size()[0])).to(dev)

    roisizex = Mx *pixelsize
    roisizey = My *pixelsize
    xmin = -roisizex / 2.4
    xmax = roisizex / 2.4
    ymin = -roisizey / 2.4
    ymax = roisizey / 2.4
    zmin = zspread[0]
    zmax = zspread[1]

    if zstack:
        thetamin = torch.concat((torch.tensor([xmin, ymin, zmin, 1, 0]).to(dev),-zernikecoefsmax), dim=0)
        thetamax = torch.concat((torch.tensor([xmax, ymax, zmax, 1e6, 1e5]).to(dev) ,zernikecoefsmax), dim = 0)
    else:
        thetamin = torch.tensor([xmin, ymin, zmin, 1, 0]).to(dev)
        thetamax = torch.tensor([xmax, ymax, zmax, 1e6, 1e5]).to(dev)

    return thetamin, thetamax


def initial_guess(smp, dev, zstack=False, Lambda=1e-6):
    from glrt.torch_stuff import Gaussian2DAstigmaticPSF, LM_MLE
    torch.cuda.empty_cache()
    calib = [
        [1.0573989152908325,
         -0.14864186942577362,
         0.1914146989583969,
         0.10000000149011612],
        [1.0528310537338257,
         0.14878079295158386,
         0.18713828921318054,
         9.999999974752427e-07]]



    initial = torch.zeros((smp.size()[0], 5)).to(dev)
    initial[:, 0:2] = smp.size()[2] / 2
    initial[:, 2] = 0
    initial[:, 3] = 2000
    initial[:, 4] = 1

    roisize = smp.size()[2]
    param_range = torch.Tensor([
        [0, roisize - 1],
        [0, roisize - 1],
        [-1, 1],
        [1, 1e9],
        [1e-6, 1e6],
    ]).to(dev)
    multiple_gpu = True
    model = Gaussian2DAstigmaticPSF(roisize, calib)
    mle = LM_MLE(model, lambda_=Lambda, iterations=200,
                 param_range_min_max=param_range)

    with torch.no_grad():

        smp_ = smp




        if multiple_gpu:
            mle = torch.nn.DataParallel(mle)  # select if multiple gpus
        else:
            mle = torch.jit.script(mle)  # select if single gpus
        params_, loglik_, _ = mle.forward(smp_, initial)

        params = params_.cpu().detach().numpy()





    return params_


@torch.jit.script
def get_zernikefunctions(orders, XPupil, YPupil, dev: str):
    x = XPupil * 1
    y = YPupil * 1

    zersize = orders.size()
    Nzer = zersize[0]
    radormax = torch.max(orders[:, 0])
    azormax = torch.max(abs(orders[:, 1]))
    Nx, Ny = x.size()[0], x.size()[0]

    # Evaluation of the radial Zernike polynomials using the recursion relation for
    # the Jacobi polynomials.

    zerpol = torch.zeros((Nx, Ny, int(radormax * azormax - 1), int(azormax) + 1)).to(dev)
    rhosq = x ** 2 + y ** 2
    rho = torch.sqrt(rhosq)
    zerpol[:, :, 0, 0] = torch.ones(x.size())

    for jm in range(int(azormax) + 1):
        m = jm * 1
        if m > 0:
            zerpol[:, :, jm, jm] = rho * zerpol[:, :, jm - 1, jm - 1]

        zerpol[:, :, jm + 2, jm] = ((m + 2) * rhosq - m - 1) * torch.squeeze(zerpol[:, :, jm, jm])
        itervalue = int(radormax) - 1 - m + 2

        for p in range(itervalue):
            piter = p + 2
            n = m + 2 * piter
            jn = n * 1

            zerpol[:, :, jn, jm] = (2 * (n - 1) * (n * (n - 2) * (2 * rhosq - 1) - m ** 2) * zerpol[:, :, jn - 2, jm] -
                                    n * (n + m - 2) * (n - m - 2) * zerpol[:, :, jn - 4, jm]) / (
                                           (n - 2) * (n + m) * (n - m))

    phi = torch.arctan2(y, x)

    allzernikes = torch.zeros((Nx, Ny, Nzer)).to(dev)

    for j in range(Nzer):
        n = int(orders[j, 0])
        m = int(orders[j, 1])
        if m >= 0:
            allzernikes[:, :, j] = zerpol[:, :, n, m] * torch.cos(m * phi)
        else:
            allzernikes[:, :, j] = zerpol[:, :, n, -m] * torch.sin(-m * phi)

    return allzernikes


#@torch.jit.script
def get_pupil_matrix(NA: float, zvals, refmed: float, refcov: float, refimm: float, refimmnom: float, Lambda: float,
                     Npupil: int, abberations, dev: str):


    PupilSize = 1.0
    DxyPupil = 2 * PupilSize / Npupil
    XYPupil = torch.arange(-PupilSize + DxyPupil / 2, PupilSize, DxyPupil).to(dev)
    YPupil, XPupil = torch.meshgrid(XYPupil, XYPupil, indexing='xy')

    # % calculation of relevant Fresnel-coefficients for the interfaces
    # % between the medium and the cover slip and between the cover slip
    # % and the immersion fluid
    # % The Fresnel-coefficients should be divided by the wavevector z-component
    # % of the incident medium, this factor originates from the
    # % Weyl-representation of the emitted vector spherical wave of the dipole.
    argMed = 1 - (XPupil ** 2 + YPupil ** 2) * NA ** 2 / refmed ** 2
    phiMed = torch.arctan2(torch.tensor(0, dtype=torch.float).to(dev), argMed)
    complex1 = torch.tensor(1j, dtype=torch.complex64).to(dev)

    # CosThetaMed = test4 * (test2 - test3 - 0j)
    CosThetaMed = torch.sqrt(torch.abs(argMed)) * (torch.cos(phiMed / 2) - complex1 * torch.sin(phiMed / 2) - 0j)
    CosThetaCov = torch.sqrt(1 - (XPupil ** 2 + YPupil ** 2) * NA ** 2 / refcov ** 2 - 0j)
    CosThetaImm = torch.sqrt(1 - (XPupil ** 2 + YPupil ** 2) * NA ** 2 / refimm ** 2 - 0j)
    CosThetaImmnom = torch.sqrt(1 - (XPupil ** 2 + YPupil ** 2) * NA ** 2 / refimmnom ** 2 - 0j)

    FresnelPmedcov = 2 * refmed * CosThetaMed / (refmed * CosThetaCov + refcov * CosThetaMed)
    FresnelSmedcov = 2 * refmed * CosThetaMed / (refmed * CosThetaMed + refcov * CosThetaCov)
    FresnelPcovimm = 2 * refcov * CosThetaCov / (refcov * CosThetaImm + refimm * CosThetaCov)
    FresnelScovimm = 2 * refcov * CosThetaCov / (refcov * CosThetaCov + refimm * CosThetaImm)
    FresnelP = FresnelPmedcov * FresnelPcovimm
    FresnelS = FresnelSmedcov * FresnelScovimm
    #
    # # setting of vectorial functions
    Phi = torch.arctan2(YPupil, XPupil)
    CosPhi = torch.cos(Phi)
    SinPhi = torch.sin(Phi)
    CosTheta = CosThetaMed  # sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);
    SinTheta = torch.sqrt(1 - CosTheta ** 2)

    pvec = torch.zeros((Npupil, Npupil, 3), dtype=torch.complex64).to(dev)
    svec = torch.zeros((Npupil, Npupil, 3), dtype=torch.complex64).to(dev)

    pvec[:, :, 0] = (FresnelP + 0j) * (CosTheta + 0j) * (CosPhi + 0j)
    pvec[:, :, 1] = FresnelP * CosTheta * (SinPhi + 0j)
    pvec[:, :, 2] = -FresnelP * (SinTheta + 0j)
    svec[:, :, 0] = -FresnelS * (SinPhi + 0j)
    svec[:, :, 1] = FresnelS * (CosPhi + 0j)
    svec[:, :, 2] = 0

    PolarizationVector = torch.zeros((Npupil, Npupil, 2, 3), dtype=torch.complex64).to(dev)
    PolarizationVector[:, :, 0, :] = CosPhi[:, :, None] * pvec - SinPhi[:, :, None] * svec
    PolarizationVector[:, :, 1, :] = SinPhi[:, :, None] * pvec + CosPhi[:, :, None] * svec

    # definition aperture
    ApertureMask = (XPupil ** 2 + YPupil ** 2) < 1.0

    # aplanatic amplitude factor
    Amplitude = ApertureMask * torch.sqrt(CosThetaImm) / (refmed * CosThetaMed)
    Amplitude[~ApertureMask] = 0 + 0j
    #
    # calculation aberration function
    size_pup = XPupil.size()
    Waberration = torch.zeros(size_pup, dtype=torch.complex64).to(dev)
    orders = abberations[:, 0:2]
    zernikecoefs = abberations[:, 2]
    normfac = torch.sqrt(2 * (orders[:, 0] + 1) / (1 + (orders[:, 1] == 0)))
    zernikecoefs = normfac * zernikecoefs
    all_zernikes = get_zernikefunctions(orders, XPupil, YPupil, dev)

    for j in range(len(zernikecoefs)):
        Waberration = Waberration + zernikecoefs[j] * all_zernikes[:, :, j]

    Waberration = Waberration + zvals[0] * refimm * CosThetaImm - zvals[1] * refimmnom * CosThetaImmnom - zvals[
        2] * refmed * CosThetaMed
    PhaseFactor = torch.exp(2 * torch.pi * complex1 * torch.real(Waberration / Lambda))
    Waberration = Waberration * ApertureMask
    #
    # test = torch.tensor(PhaseFactor).to(dev)
    # compute pupil matrix
    PupilMatrix = Amplitude[..., None, None] * PhaseFactor[..., None, None] * PolarizationVector

    PupilMatrix[~ApertureMask] = 0 + 0j
    # calculate wavevector inside immersion fluid and z-component inside medium
    wavevector = torch.zeros((XPupil.size()[0], XPupil.size()[0], 3), dtype=torch.complex64).to(dev)
    wavevector[:, :, 0] = (2 * torch.pi * NA / Lambda) * XPupil
    wavevector[:, :, 1] = (2 * torch.pi * NA / Lambda) * YPupil
    wavevector[:, :, 2] = (2 * torch.pi * refmed / Lambda) * CosThetaMed
    wavevectorzimm = (2 * torch.pi * refimm / Lambda) * CosThetaImm
    #
    return wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix
    #return PupilSize

@torch.jit.script
def cztfunc2D(N: int, M: int, L: int, datain, Amt, Bmt, Dmt, dev: str, zstack:bool):
    # cztfunc2D(datain, Amt, Bmt, Dmt, params)
    if zstack:
        cztin = torch.zeros((datain.size()[0], L, 2, 3), dtype=torch.complex64).to(dev)

        cztin[:, 0:N, :, :] = Amt[..., None, None] * datain
        temp = Dmt[..., None, None] * torch.fft.fft(cztin, dim=1)
        cztout = torch.fft.ifft(temp, dim=1)
        dataout = Bmt[..., None, None] * cztout[:, 0:M, :, :]
        dataout = torch.transpose(dataout, 0, 1)
    else:
        cztin = torch.zeros((datain.size()[0],datain.size()[1], L, 2, 3), dtype=torch.complex64).to(dev)

        cztin[:,:, 0:N, :, :] = Amt[None,..., None, None] * datain
        temp = Dmt[None,..., None, None] * torch.fft.fft(cztin, dim=2)
        cztout = torch.fft.ifft(temp, dim=2)
        dataout = Bmt[None,..., None, None] * cztout[:,:, 0:M, :, :]
        dataout = torch.transpose(dataout, 1, 2)
    return dataout


@torch.jit.script
def cztfunc3D(N: int, M: int, L: int, datain, Amt, Bmt, Dmt, dev: str, zstack:bool):
    # cztfunc2D(datain, Amt, Bmt, Dmt, params)
    if zstack:
        dim = 3  # 3 dimensions
        cztin = torch.zeros((datain.size()[0], L, dim, 2, 3), dtype=torch.complex64).to(dev)

        cztin[:, 0:N, :, :, :] = Amt[..., None, None, None] * datain
        temp = Dmt[..., None, None, None] * torch.fft.fft(cztin, dim=1)
        cztout = torch.fft.ifft(temp, dim=1)
        dataout = Bmt[..., None, None, None] * cztout[:, 0:M, :, :, :]
        dataout = torch.transpose(dataout, 0, 1)

    else:
        dim = 3  # 3 dimensions
        cztin = torch.zeros((datain.size()[0], datain.size()[1], L, dim, 2, 3), dtype=torch.complex64).to(dev)

        cztin[:, :, 0:N, :, :, :] = Amt[None, ..., None, None, None] * datain
        temp = Dmt[None, ..., None, None, None] * torch.fft.fft(cztin, dim=2)
        cztout = torch.fft.ifft(temp, dim=2)
        dataout = Bmt[None, ..., None, None, None] * cztout[:,: , 0:M, :, :, :]
        dataout = torch.transpose(dataout, 1, 2)
    return dataout

@torch.jit.script
def get_field_matrix_derivatives(zmin: float, zmax: float, K: int, N: int, M: int, L: int, Ax, Bx,
                                 Dx, Ay, By, Dy, dev: str, Mx: int, My: int, Npupil: int, Lambda: float,
                                 abberations, PupilMatrix, all_zernikes, wavevector, wavevectorzimm, theta,
                                 zstack: bool):
    complex1 = torch.tensor(1j, dtype=torch.complex64).to(dev)

    dz = (zmax - zmin) / K
    Zimage = torch.linspace(zmin, zmax + dz, K).to(dev)
    numders = 3  # 3D

    orders = abberations[:, 0:2]
    normfac = torch.sqrt(2 * (orders[:, 0] + 1) / (1 + (orders[:, 1] == 0)))

    # find theta at z = 0 and index


    num_beads = theta.size()[0]
    if zstack == True:
        xemit = theta[ 0]  # distance from center
        yemit = theta[ 1]
        zemit = theta[ 2]

        FieldMatrix = torch.zeros((Mx, My, K, 2, 3), dtype=torch.complex64).to(dev)
        FieldMatrixDerivatives = torch.zeros((Mx, My, K, numders + abberations.size()[0], 2, 3),
                                             dtype=torch.complex64).to(dev)
        PupilFunctionDerivatives = torch.zeros((Npupil, Npupil, 3, 2, 3), dtype=torch.complex64).to(
            dev)  # pupil, pupil, dim, 2,3

        for jz in range(K):
            zemitrun = Zimage[jz]

            # phase contribution due to position of the emitter
            Wlateral = xemit * wavevector[:, :, 0] + yemit * wavevector[:, :, 1]
            Wpos = Wlateral + (zemit - zemitrun) * wavevectorzimm
            PositionPhaseMask = torch.exp(-complex1 * Wpos)
            # Pupil function
            PupilFunction = PositionPhaseMask[..., None, None] * PupilMatrix

            # Pupil function for xy - derivatives
            PupilFunctionDerivatives[:, :, 0, :, :] = -complex1 * wavevector[:, :, 0][..., None, None] * PupilFunction
            PupilFunctionDerivatives[:, :, 1, :, :] = -complex1 * wavevector[:, :, 1][..., None, None] * PupilFunction

            # pupil functions for z-derivatives (only for xyz, stage)
            PupilFunctionDerivatives[:, :, 2, :, :] = -complex1 * wavevectorzimm[..., None, None] * PupilFunction

            # Field matrix and derivatives
            # IntermediateImage = cztfunc2D(PupilFunction,Ay,By,Dy,params)
            IntermediateImage = cztfunc2D(N, M, L, PupilFunction, Ay, By, Dy, dev, zstack)
            FieldMatrix[:, :, jz, :, :] = cztfunc2D(N, M, L, IntermediateImage, Ax, Bx, Dx, dev, zstack)
            IntermediateImage = cztfunc3D(N, M, L, PupilFunctionDerivatives[:, :, 0:3, :, :], Ay, By, Dy, dev, zstack)
            FieldMatrixDerivatives[:, :, jz, 0:numders, :, :] = cztfunc3D(N, M, L, IntermediateImage, Ax, Bx, Dx, dev, zstack)

            # for abberations

            # add rows along dimension (not needed for matlab)

            for jzer in range(abberations.size()[0]):
                jder = numders + jzer

                PupilFunction = (2 * torch.pi * complex1 * normfac[jzer] * all_zernikes[:, :, jzer] / Lambda)[
                                    ..., None, None] \
                                * (PositionPhaseMask[..., None, None] * PupilMatrix)
                IntermediateImage = cztfunc2D(N, M, L, PupilFunction, Ay, By, Dy, dev, zstack)

                FieldMatrixDerivatives[:, :, jz, jder, :, :] = cztfunc2D(N, M, L, IntermediateImage, Ax, Bx, Dx, dev, zstack)


    # # plotting intermediate results
    # if params.debugmode:
    #     jz = int(torch.ceil(Mz / 2))
    #     numb = 0
    #     plt.figure()
    #     for itel in range(2):
    #         for jtel in range(3):
    #             tempim = FieldMatrix[:,:,jz,itel,jtel]
    #             numb +=1
    #             plt.subplot(2,3,numb)
    #             plt.imshow(abs(tempim))
    #             plt.title('Mag')
    #     plt.show()
    #
    #     numb = 0
    #     plt.figure()
    #     for itel in range(2):
    #         for jtel in range(3):
    #             tempim = FieldMatrix[:,:,jz,itel,jtel]
    #             numb +=1
    #             plt.subplot(2,3,numb)
    #             plt.imshow(torch.angle(tempim)*180/torch.pi)
    #             plt.title('Phase')
    #     plt.show()
    else:
        xemit = theta[:, 0]  # distance from center
        yemit = theta[:, 1]
        zemit = theta[:, 2]
        wavevector = torch.tile(wavevector[None,...], (num_beads,1,1,1))
        Wlateral = xemit[..., None, None] * wavevector[:,:, :, 0] + yemit[..., None,None]  * wavevector[:, :,:, 1]
        Wpos = Wlateral + (zemit[..., None,None]) * wavevector[:,:, :, 2]  # only for medium!
        #Wpos = Wlateral + (zemit[..., None, None]) * wavevectorzimm
        PositionPhaseMask = torch.exp(-complex1 * Wpos)
        # Pupil function
        PupilFunction = PositionPhaseMask[..., None, None] * PupilMatrix

        PupilFunctionDerivatives = torch.zeros((num_beads,Npupil, Npupil, 3, 2, 3), dtype=torch.complex64).to(
            dev)  # pupil, pupil, dim, 2,3
        # Pupil function for xy - derivatives
        PupilFunctionDerivatives[:,:, :, 0, :, :] = -complex1 * wavevector[:,:, :, 0][..., None, None] * PupilFunction
        PupilFunctionDerivatives[:,:, :, 1, :, :] = -complex1 * wavevector[:,:, :, 1][..., None, None] * PupilFunction

        # pupil functions for z-derivatives (only for xyz, medium!!)
        PupilFunctionDerivatives[:,:, :, 2, :, :] = -complex1 * wavevector[:,:, :, 2][..., None, None] * PupilFunction
        #PupilFunctionDerivatives[:, :, :, 2, :, :] = -complex1 * wavevectorzimm[..., None, None] * PupilFunction

        IntermediateImage = cztfunc2D(N, M, L, PupilFunction, Ay, By, Dy, dev, zstack)
        FieldMatrix = cztfunc2D(N, M, L, IntermediateImage, Ax, Bx, Dx, dev, zstack)
        IntermediateImage = cztfunc3D(N, M, L, PupilFunctionDerivatives, Ay, By, Dy, dev, zstack)
        FieldMatrixDerivatives = cztfunc3D(N, M, L, IntermediateImage, Ax, Bx, Dx, dev, zstack)

    return FieldMatrix, FieldMatrixDerivatives


@torch.jit.script
def get_normalization(NA: float, Lambda: float, Npupil: int, pixelsize: float, PupilMatrix):
    # Intensity matrix
    IntensityMatrix = torch.zeros((3, 3))
    for itel in range(3):
        for jtel in range(3):
            pupmat1 = PupilMatrix[:, :, :, itel]
            pupmat2 = PupilMatrix[:, :, :, jtel]
            IntensityMatrix[itel, jtel] = torch.sum(torch.sum(torch.real(pupmat1 * torch.conj(pupmat2))))

    # normalization to take into account discretization correctly
    DxyPupil = 2 * NA / Lambda / Npupil
    normfac = DxyPupil ** 2 / pixelsize ** 2
    IntensityMatrix = normfac * IntensityMatrix

    # evaluation normalization factors
    normint_free = torch.sum(torch.diag(IntensityMatrix)) / 3

    return normint_free


@torch.jit.script
def get_psfs_derivatives(PupilMatrix, FieldMatrix, FieldMatrixDerivatives, NA: float, Lambda: float, Npupil: int,
                         pixelsize: float, zstack: bool):
    # function [PSF,PSFder] = get_psfs_derivatives(params,PupilMatrix,FieldMatrix,FieldMatrixDerivatives)
    # % This function calculates the free or fixed dipole PSFs given the field
    # % matrix, the dipole orientation, and the pupil polarization, as well as
    # % the derivatives w.r.t. the xyz coordinates of the emitter and w.r.t. the
    # % emission wavelength lambda.
    # %
    # % parameters: emitter/absorber dipole orientation (characterized by angles
    # % pola and azim).
    # %

    # PSF normalization (only freely diffusive)
    normint_free = get_normalization(NA, Lambda, Npupil, pixelsize, PupilMatrix)

    # if free and ZSTACK!
    if zstack:
        FreePSF = 1 / 3 * torch.sum(torch.sum(torch.abs(FieldMatrix) ** 2, dim=-1), dim=-1)
        tmp = torch.transpose(FieldMatrixDerivatives, 3, 4)
        tmpFieldMatrixDerivatives = torch.transpose(tmp, 4, 5)
        # tmpFieldMatrixDerivatives = torch.transpose(FieldMatrixDerivatives, [0,1,2,4,5,3])
        FreePSFder = (2 / 3) * torch.sum(torch.sum(torch.real(torch.conj(FieldMatrix[..., None]) *
                                                              tmpFieldMatrixDerivatives), dim=-2), dim=-2)
    else:
        FreePSF = 1 / 3 * torch.sum(torch.sum(torch.abs(FieldMatrix) ** 2, dim=-1), dim=-1)
        tmp = torch.transpose(FieldMatrixDerivatives, 3, 4)
        tmpFieldMatrixDerivatives = torch.transpose(tmp, 4, 5)

        FreePSFder = (2 / 3) * torch.sum(torch.sum(torch.real(torch.conj(FieldMatrix[..., None]) *
                                                              tmpFieldMatrixDerivatives), dim=-2), dim=-2)

    # TODO : no zstack

    FreePSF = FreePSF / normint_free
    FreePSFder = FreePSFder / normint_free

    # free
    PSF = FreePSF * 1
    PSFder = FreePSFder * 1

    return PSF, PSFder

@torch.jit.script
def poissonrate(NA:float, zvals, refmed:float, refcov:float, refimm:float, refimmnom:float, Lambda:float, Npupil:int,
                abberations, zmin:float, zmax:float, K:int, N:int, M:int, L:int, Ax, Bx,
                Dx, Ay, pixelsize:float, By, Dy, Mx:int, My:int, numparams:int, theta_, dev:str, zstack:bool,  wavevector, wavevectorzimm, all_zernikes, PupilMatrix, ibg_ony:bool=False):

    if zstack:
        wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals, refmed, refcov,
                                                                                              refimm, refimmnom, Lambda,
                                                                                              Npupil, abberations, dev)


    FieldMatrix, FieldMatrixDerivatives = get_field_matrix_derivatives(zmin, zmax, K, N, M, L, Ax, Bx, Dx, Ay, By, Dy,
                                                                       dev,
                                                                       Mx, My, Npupil, Lambda, abberations, PupilMatrix,
                                                                       all_zernikes, wavevector, wavevectorzimm, theta_,zstack)

    PSF, PSFder = get_psfs_derivatives(PupilMatrix, FieldMatrix, FieldMatrixDerivatives, NA, Lambda, Npupil, pixelsize,
                                       zstack)


    if zstack:
        mu = torch.zeros((Mx, My, K)).to(dev)
        dmudtheta = torch.zeros(( Mx, My, K, numparams)).to(dev)
        mu[:, :, :] = theta_[3] * PSF[:, :, :] + theta_[4]
        dmudtheta[:, :, :, 0:3] = theta_[3] * PSFder[:, :, :, 0:3]
        dmudtheta[:, :, :, 3] = PSF
        dmudtheta[:, :, :, 4] = 1
        dmudtheta[:, :, :, 5::] = theta_[3] * PSFder[:, :, :, 3::]

    elif ibg_ony == True:
        numbeads = theta_.size()[0]
        mu = torch.zeros((numbeads, Mx, My)).to(dev)
        dmudtheta = torch.zeros((numbeads, Mx, My, numparams)).to(dev)
        mu[:, :, :] = theta_[:, 3, None, None] * PSF[:, :, :] + theta_[:, 4, None, None]
        dmudtheta[:, :, :, 0:3] = theta_[:, 3, None, None, None] * PSFder[:, :, :, 0:3]
        dmudtheta[:, :, :, 3] = PSF
        dmudtheta[:, :, :, 4] = 1
        dmudtheta = dmudtheta[:, :, :, 3:5]

    else:
            numbeads = theta_.size()[0]
            mu = torch.zeros((numbeads, Mx, My)).to(dev)

            dmudtheta = torch.zeros((numbeads, Mx, My, numparams)).to(dev)
            mu[:,:, :] = theta_[:,3,None, None] * PSF[:,:, :] + theta_[:,4,None, None]
            dmudtheta[:,:, :, 0:3] = theta_[:,3,None, None,None] * PSFder[:,:, :, 0:3]
            dmudtheta[:,:, :, 3] = PSF
            dmudtheta[:,:, :, 4] = 1
            # dmudtheta[:, :, :, 5::] = theta_[3] * PSFder[:, :, 3::]

    return mu, dmudtheta

@torch.jit.script
def likelihood(numparams:int,image,mu,dmudtheta,dev:str):

    varfit = 0
    # calculation of weight factors
    keps = 1e3*2.220446049250313e-16
    mupos = (mu>0)*mu + (mu<0)*keps

    weight = (image-mupos)/(mupos+varfit)
    dweight = (image+varfit)/(mupos+varfit)**2

    num_beads = image.size()[0]

    #log-likelihood, gradient vector and Hessian matrix
    logL = torch.sum(torch.sum((image + varfit) * torch.log(mupos + varfit) - (mupos + varfit), dim=-1), dim=-1)
    gradlogL = torch.sum(torch.sum(weight[..., None]*dmudtheta, dim=1), dim=1)

    HessianlogL = (-dweight[...,None, None]*dmudtheta[:,:,:,None, :] * dmudtheta[:, :,:, :,None]).sum((1,2))
    #
    # # TODO: try to optimize following:
    # HessianlogL = torch.zeros((num_beads, numparams, numparams)).to(dev)
    # for qq in range(num_beads):
    #     for ii in range(numparams):
    #         for jj in range(numparams):
    #             HessianlogL[qq,ii,jj] = torch.sum(-weight[qq,:,:]*dmudtheta[qq,:,:,ii]*dmudtheta[qq,:,:,jj])

    return logL,gradlogL,HessianlogL
