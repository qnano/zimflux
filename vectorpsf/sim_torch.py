# this script is for single images - and we assume the abberations are known. not for zstack


import torch
import numpy as np
import matplotlib.pyplot as plt

import math
from time import time
from typing import List, Tuple, Union, Optional
from torch import Tensor
from config import sim_params
import tifffile
import napari
from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, Model_vectorial_psf, LM_MLE

def show_napari(img):
    viewer = napari.Viewer()
    viewer.add_image(img)

with torch.no_grad():
    params = sim_params()
    dev = 'cuda'
    zstack = False
    #region get values from params
    NA = params.NA
    refmed = params.refmed
    refcov = params.refcov
    refimm = params.refimm
    refimmnom = params.refimmnom
    Lambda = params.Lambda
    Npupil = params.Npupil
    abberations = torch.from_numpy(params.abberations).to(dev)
    zvals = torch.from_numpy(params.zvals).to(dev)
    zspread = torch.tensor(params.zspread).to(dev)
    numparams = params.numparams
    numparams_fit = 5
    K = params.K
    Mx = params.Mx
    My = params.My
    Mz = params.Mz

    pixelsize = params.pixelsize
    Ax = torch.tensor(params.Axmt).to(dev)
    Bx = torch.tensor(params.Bxmt).to(dev)
    Dx = torch.tensor(params.Dxmt).to(dev)
    Ay = torch.tensor(params.Aymt).to(dev)
    By = torch.tensor(params.Bymt).to(dev)
    Dy = torch.tensor(params.Dymt).to(dev)

    N = params.cztN
    M = params.cztM
    L = params.cztL

    Nitermax = params.Nitermax
    # endregion
    numbeads = 200
    wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals, refmed,
                                                                      refcov, refimm, refimmnom, Lambda,Npupil, abberations, dev)
    #
    # test = PupilMatrix.detach().cpu().numpy()
    # show_napari(np.transpose(np.absolute(test),[2,3,0,1]))

    # region simulation
    dx = (1-2*torch.rand((numbeads,1)))*params.pixelsize
    dy = (1-2*torch.rand((numbeads,1)))*params.pixelsize
    dz = (1-2*torch.rand((numbeads,1)))*200
    Nphotons = torch.ones((numbeads,1))*4000 + (1-2*torch.rand((numbeads,1)))*300
    Nbackground = torch.ones((numbeads,1))*100 + (1-2*torch.rand((numbeads,1)))*10

    ground_truth = torch.concat((dx, dy, dz, Nphotons, Nbackground),axis=1).to(dev)
    zmin = 0
    zmax = 0
    mu, dmudtheta = poissonrate(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin, zmax, K, N,
                                M, L, Ax, Bx,
                                Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, ground_truth, dev, zstack,
                                wavevector, wavevectorzimm, all_zernikes, PupilMatrix)

    spots_ = torch.poisson(mu)
    spots = spots_.detach().cpu().numpy()
    # show_napari(spots)
    # endregion

    theta = initial_guess(spots_, dev)
    theta[:,0:2] = (theta[:,0:2] - Mx/2)*pixelsize
    theta[:, 2] = 0#(theta[:, 2])*1000
    theta[:, 3] = 4000
    theta[:, 4] = 100
    thetamin, thetamax = thetalimits(abberations,Lambda, Mx, My,pixelsize,zspread, dev, zstack= False)
    thetaretry = theta * 1

    param_range = torch.concat((thetamin[...,None], thetamax[...,None]),dim=1)
    model = Model_vectorial_psf()
    mle = LM_MLE(model, lambda_=1e-1, iterations=150,
                 param_range_min_max=param_range)
    mle = torch.jit.script(mle)

    params_, loglik_, _ = mle.forward(spots_, NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                zmax, K, N,
                M, L, Ax, Bx,
                Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, theta, dev, zstack,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix)

    # with params
    mu_est, dmu = model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                zmax, K, N,
                M, L, Ax, Bx,
                Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, params_, dev, zstack,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix)


    # with ground truth
    mu_mod, dmu = model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                zmax, K, N,
                M, L, Ax, Bx,
                Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, ground_truth, dev, zstack,
                wavevector, wavevectorzimm, all_zernikes, PupilMatrix)




    crlb_ = compute_crlb(mu,dmu)
    errors = ground_truth - params_


prec = errors.std(0).cpu().detach().numpy()
rmsd = torch.sqrt((errors ** 2).mean(0)).cpu().detach().numpy()
crlb = (torch.mean(crlb_, dim=0)).cpu().detach().numpy()

print('rmsd = ',rmsd )
print('preciscion = ',  prec)
print('crlb = ', crlb)

mu_num = mu_mod.cpu().detach().numpy()
mu_est = mu_est.cpu().detach().numpy()
show_napari(np.concatenate((mu_num, mu_est), axis=1))
