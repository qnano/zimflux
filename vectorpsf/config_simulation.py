import pandas as pd
import ypstruct as yp
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib

def sim_params(depth=0, astig=0, roisize=16):

    params = yp.struct()
    params.debugmode = False
    # fitting parameters
    params.Nitermax = 20
    params.tollim = 1e-6
    params.varfit = 0


    # camera offset and gain
    params.offset = 0
    params.gain = 1



    # PSF/optical parameters
    params.NA = 1.49
    params.refmed = 1.33
    params.refcov = 1.52
    params.refimm = 1.52
    params.refimmnom = params.refcov
    params.fwd = 120e3
    params.depth = depth
    params.zrange = [0,0] # actual range - only relevant for zstack
    if astig !=0:
        params.zspread = [-2000,2000]
    else:
        params.zspread = [-300,300]# spread - limits of estimator
    params.ztype =  'medium'
    params.Lambda = 715
    params.lambdacentral = 715
    params.lambdaspread = [715,715]
    params.xemit = 0.0
    params.yemit = 0.0
    params.zemit = 0.0
    params.Npupil = 64
    params.pixelsize = 65
    params.samplingdistance = params.pixelsize
    params.Mx = roisize # roisize
    params.My = params.Mx
    params.Mz = 1

    params.xrange = params.pixelsize*params.Mx/2
    params.yrange = params.pixelsize*params.My/2


    zvals, _ = get_rimismatch(params)


    params.zvals = zvals

    # model parameters
    params.alpha = 0
    params.beta = 0
    params.K = 1 # zstack number
    params.m = 1

    params.abberations = np.array([
    [2 , -2, 0.0],
    [2, 2, 60/params.lambdacentral],
    [3, -1, 0],
    [3, 1, 0],
    [3, -3, 0],
    [3, 3, 0],
    [4, 0, 0],
    [4, -2, 0],
    [4, 2, 0],
    [5, -1, 0],
    [5, 1, 0],
    [6, 0, 0]

    ])

    params.abberations[:,2] =  params.abberations[:,2]*params.lambdacentral
    # params.abberations[:,2] = np.array([ 1.67204599e+00,  6.23799172e+01,  9.77935574e-01,
    #     1.40272625e+00, -2.14605383e+00, -1.24214515e+00, -2.16069084e-01,
    #     1.52329518e+00,  8.13260374e+00, -5.16444508e+00, -7.38316159e-01,
    #     5.88651617e+00])
    #params.abberations[:, 2] = (np.load('C:/Development/simflux3d/projects/vectorpsf/delft_beads_jan16_20nm.npy'))

    if astig!=0:
        params.abberations[1, 2] = astig
    else:
        params.abberations[:, 2] = (np.load('/vectorpsf/delft_beads_feb23_15nm.npy'))
        params.abberations[-6, 2] = -0.2
    #params.abberations[8, 2] = params.abberations[8, 2]*0.2

    PupilSize = params.NA / params.Lambda
    params.numparams = 5 + np.size(params.abberations[:, 2])
    # calculate auxiliary vectors for chirpz

    Ax,Bx,Dx = prechirpz(PupilSize, params.xrange, params.Npupil, params.Mx)
    Ay, By, Dy = prechirpz(PupilSize, params.yrange, params.Npupil, params.My)

    params.cztN = params.Npupil
    params.cztM = params.Mx
    params.cztL = params.Npupil + params.Mx - 1

    params.Axmt = np.matlib.repmat(Ax, params.Mx, 1)
    params.Bxmt = np.matlib.repmat(Bx, params.Mx, 1)
    params.Dxmt = np.matlib.repmat(Dx, params.Mx, 1)
    params.Aymt = np.matlib.repmat(Ay, params.Npupil, 1)
    params.Bymt = np.matlib.repmat(By, params.Npupil, 1)
    params.Dymt = np.matlib.repmat(Dy, params.Npupil, 1)


    return params

def z_stack_params():

    params = yp.struct()
    params.debugmode = False
    # fitting parameters
    params.Nitermax = 50
    params.tollim = 1e-6
    params.varfit = 0


    # camera offset and gain
    params.offset = 100
    params.gain = 0.44



    # PSF/optical parameters
    params.NA = 1.49
    params.refmed = 1.35
    params.refcov = 1.52
    params.refimm = 1.52
    params.refimmnom = params.refcov
    params.fwd = 120e3
    params.depth = 0
    params.zrange = [-400,600] # actual range
    params.zspread = [-1000,1000] # spread
    params.ztype = 'stage' # 'medium'
    params.Lambda = 640
    params.lambdacentral = 640
    params.lambdaspread = [640,640]
    params.xemit = 0.0
    params.yemit = 0.0
    params.zemit = 0.0
    params.Npupil = 64
    params.pixelsize = 65
    params.samplingdistance = params.pixelsize
    params.Mx = 16 # roisize
    params.My = params.Mx
    params.Mz = 0

    params.xrange = params.pixelsize*params.Mx/2
    params.yrange = params.pixelsize*params.My/2

    # SAF check

    if params.NA>params.refmed and params.depth<4*params.Lambda:
        zvals, _ = get_rimismatch(params)
    else:
        zvals, _ = get_rimismatch(params)

    params.zvals = zvals

    # model parameters
    params.alpha = 0
    params.beta = 0
    params.K = 50 # zstack number
    params.m = 1

    params.abberations = np.array([
    [2.0, -2, 0.0],
    [2, 2, 100/params.lambdacentral],
    [3, -1, 0],
    [3, 1, 0],
    [3, -3, 0],
    [3, 3, 0],
    [4, 0, 0],
    [4, -2, 0],
    [4, 2, 0],
    [5, -1, 0],
    [5, 1, 0],
    [6, 0, 0]

    ])

    params.abberations[:,2] =  params.abberations[:,2]*params.lambdacentral

    PupilSize = params.NA / params.Lambda
    params.numparams = 5 + np.size(params.abberations[:, 2])
    # calculate auxiliary vectors for chirpz

    Ax,Bx,Dx = prechirpz(PupilSize, params.xrange, params.Npupil, params.Mx)
    Ay, By, Dy = prechirpz(PupilSize, params.yrange, params.Npupil, params.My)

    params.cztN = params.Npupil
    params.cztM = params.Mx
    params.cztL = params.Npupil + params.Mx - 1

    params.Axmt = np.matlib.repmat(Ax, params.Mx, 1)
    params.Bxmt = np.matlib.repmat(Bx, params.Mx, 1)
    params.Dxmt = np.matlib.repmat(Dx, params.Mx, 1)
    params.Aymt = np.matlib.repmat(Ay, params.Npupil, 1)
    params.Bymt = np.matlib.repmat(By, params.Npupil, 1)
    params.Dymt = np.matlib.repmat(Dy, params.Npupil, 1)


    return params

def get_rimismatch(params):

    refins = np.array([params.refimm, params.refimmnom, params.refmed])
    zvals = np.array([0, params.fwd ,-params.depth])
    NA = params.NA
    K = len(refins)
    if (NA>params.refmed):
        NA = params.refmed

    fsqav = np.zeros((K,1))
    fav =  np.zeros((K,1))
    Amat =  np.zeros((K,K))


    for jj in range(K):
        fsqav[jj] = refins[jj]**2 - 0.5 * NA**2
        fav[jj] = ((2/3)/NA**2) * (refins[jj]**3 - (refins[jj]**2-NA**2)**(3/2))
        Amat[jj, jj] = fsqav[jj] - fav[jj]**2

        for kk in range(jj):
            Amat[jj,kk] = (1/4/NA**2) * (refins[jj] * refins[kk] * (refins[jj]**2 + refins[kk]**2) -
            (refins[jj]**2 + refins[kk]**2 - 2*NA**2) * np.sqrt(refins[jj]**2-NA**2) * np.sqrt(refins[kk]**2-NA**2) +
            (refins[jj]**2 - refins[kk]**2)**2*np.log((np.sqrt(refins[jj]**2 - NA**2)+np.sqrt(refins[kk]**2-NA**2))/(refins[jj]+refins[kk])))
            Amat[jj, kk] = Amat[jj,kk] - fav[jj] * fav[kk]
            Amat[kk,jj] = Amat[jj,kk]*1
    zvalsratio = np.zeros((K, 1))
    Wrmsratio = np.zeros((K, K))
    for jvpr in range(K-1):
        jv = jvpr + 1
        zvalsratio[jv] = Amat[0,jv]/Amat[0,0]
        for kvpr in range(K-1):
            kv = kvpr + 1

            Wrmsratio[jv,kv] = Amat[jv,kv] - Amat[0, jv]*Amat[0,kv]/Amat[0,0]

    zvals[0] = zvalsratio[1] * zvals[1] + zvalsratio[2] * zvals[2]
    Wrms = Wrmsratio[1,1] * zvals[1] ** 2 + Wrmsratio[2,2] * zvals[2] ** 2 + 2 * Wrmsratio[1,2]* zvals[1] * zvals[2]
    Wrms = np.sqrt(Wrms)

    return zvals, Wrms

def set_saffocus(params):
    NA = params.NA
    refmed = params.refmed
    refcov = params.refcov
    refimm = params.refimm
    refimmnom = params.refimmnom
    Lambda = params.Lambda
    Npupil = params.Npupil

    zvals = np.array([0, params.fwd, -params.depth])

    # pupil radius (in diffraction units) and pupil coordinate sampling
    PupilSize = 1.0
    DxyPupil = 2*PupilSize/Npupil
    XYPupil = np.arange(-PupilSize+DxyPupil/2,PupilSize,DxyPupil)
    YPupil,XPupil = np.meshgrid(XYPupil,XYPupil)

    # % calculation of relevant Fresnel-coefficients for the interfaces
    # % between the medium and the cover slip and between the cover slip
    # % and the immersion fluid
    # % The Fresnel-coefficients should be divided by the wavevector z-component
    # % of the incident medium, this factor originates from the
    # % Weyl-representation of the emitted vector spherical wave of the dipole.

    argMed = 1-(XPupil**2+YPupil**2)*NA**2/refmed**2
    phiMed = np.arctan2(0,argMed)
    CosThetaMed = np.sqrt(abs(argMed))*(np.cos(phiMed/2)-1j*np.sin(phiMed/2) - 0j)
    CosThetaCov = np.sqrt(1-(XPupil**2+YPupil**2)*NA**2/refcov**2 - 0j)
    CosThetaImm = np.sqrt(1-(XPupil**2+YPupil**2)*NA**2/refimm**2- 0j)
    CosThetaImmnom = np.sqrt(1-(XPupil**2+YPupil**2)*NA**2/refimmnom**2- 0j)

    FresnelPmedcov = 2*refmed*CosThetaMed/(refmed*CosThetaCov+refcov*CosThetaMed)
    FresnelSmedcov = 2*refmed*CosThetaMed/(refmed*CosThetaMed+refcov*CosThetaCov)
    FresnelPcovimm = 2*refcov*CosThetaCov/(refcov*CosThetaImm+refimm*CosThetaCov)
    FresnelScovimm = 2*refcov*CosThetaCov/(refcov*CosThetaCov+refimm*CosThetaImm)
    FresnelP = FresnelPmedcov*FresnelPcovimm
    FresnelS = FresnelSmedcov*FresnelScovimm

    # setting of vectorial functions
    Phi = np.arctan2(YPupil,XPupil)
    CosPhi = np.cos(Phi)
    SinPhi = np.sin(Phi)
    CosTheta = CosThetaMed #sqrt(1-(XPupil.^2+YPupil.^2)*NA^2/refmed^2);
    SinTheta = np.sqrt(1-CosTheta**2)

    pvec = np.zeros((Npupil, Npupil,3), dtype=complex)
    svec = np.zeros((Npupil, Npupil,3), dtype=complex)

    pvec[:,:,0] = (FresnelP+0j)*(CosTheta+0j)*(CosPhi+0j)
    pvec[:,:,1] = FresnelP*CosTheta*(SinPhi+0j)
    pvec[:,:,2] = -FresnelP*(SinTheta+0j)
    svec[:,:,0]  = -FresnelS*(SinPhi+0j)
    svec[:,:,1]  = FresnelS*(CosPhi +0j)
    svec[:,:,2]  = 0

    PolarizationVector = np.zeros((Npupil,Npupil,2,3), dtype=complex)
    PolarizationVector[:,:,0,:] = CosPhi[:,:,None]*pvec-SinPhi[:,:,None]*svec
    PolarizationVector[:,:,1,:] = SinPhi[:,:,None]*pvec+CosPhi[:,:,None]*svec

    # definition aperture
    ApertureMask = (XPupil**2+YPupil**2)<1.0

    # aplanatic amplitude factor
    Amplitude = ApertureMask*np.sqrt(CosThetaImm)/(refmed*CosThetaMed)
    Strehlnorm = 0
    for itel in range(2):
        for jtel in range(3):
            Strehlnorm = Strehlnorm + abs(np.sum(np.sum(Amplitude*np.exp(2*np.pi*1j)*PolarizationVector[:,:,itel,jtel])))**2

    Nz = 101
    zpos = np.linspace(3/2*params.zspread[0],3/2*params.zspread[1],Nz)
    Strehl = np.zeros((Nz))
    for jz in range(Nz):
        zvals[0] = params.fwd - 1.25*refimm/refmed*params.depth + zpos[jz]
        Wzpos = zvals[0]*refimm*CosThetaImm-zvals[1]*refimmnom*CosThetaImmnom-zvals[2]*refmed*CosThetaMed
        Wzpos = Wzpos*ApertureMask
        for itel in range(2):
            for jtel in range(3):
                Strehl[jz] = Strehl[jz] + abs(np.sum(np.sum(Amplitude*np.exp(2*np.pi*1j*Wzpos/Lambda)*PolarizationVector[:,:,itel,jtel])))**2/Strehlnorm

    indz = np.argmax(Strehl)

    if indz <= 2:
        indz = 2
    elif indz > (Nz-3):
        indz = Nz-3

    zfit = np.polyfit(zpos[indz-2:indz+2],Strehl[indz-2:indz+2],2)
    zvals[0] = params.fwd - 1.25 * refimm / refmed * params.depth - zfit[1] / (2 * zfit[0])
    MaxStrehl = np.polyval(zfit,- zfit[1]/(2*zfit[0]))
    Wrms = Lambda/(2*np.pi)*np.log(1/MaxStrehl)

    #
    if params.debugmode:
        print('image plane depth from cover slip = %f nm\n' % -zvals[2])
        print('free working distance = %f mu\n' % (1e-3*zvals[1]))
        print('nominal z-stage position = %f mu\n' % (1e-3 * zvals[0]))
        print('rms aberration due to RI mismatch = %f mlambda\n' % (1e3 * Wrms / params.Lambda))
    if params.debugmode:
        plt.figure()
        plt.plot(zpos - 1.25 * refimm / refmed * params.depth, Strehl)
        plt.xlabel('Stage position')
        plt.ylabel('Strehl')
        plt.show()

    return zvals, Wrms


def prechirpz(xsize,qsize,N,M):


    L = N+M-1
    sigma = 2*np.pi*xsize*qsize/N/M
    Afac = np.exp(2*1j*sigma*(1-M))
    Bfac = np.exp(2*1j*sigma*(1-N))
    sqW = np.exp(2*1j*sigma)
    W = sqW**2
    Gfac = (2*xsize/N)*np.exp(1j*sigma*(1-N)*(1-M))

    Utmp = np.zeros((N),dtype=complex)
    A = np.zeros((N),dtype=complex)
    Utmp[0] = sqW*Afac
    A[0]= 1.0

    START = 1
    for index, item in enumerate(Utmp[START:], START):
        A[index] = Utmp[index-1]*A[index-1]
        Utmp[index] = Utmp[index - 1] * W


    Utmp = np.zeros(M,dtype=complex)
    B = np.ones(M,dtype=complex)
    Utmp[0] = sqW*Bfac
    B[0] = Gfac
    for index, item in enumerate(Utmp[START:], START):
        B[index] = Utmp[index-1]*B[index-1]
        Utmp[index] = Utmp[index - 1] * W


    Utmp = np.zeros(max(N,M)+1,dtype=complex)
    Vtmp = np.zeros(max(N,M)+1,dtype=complex)
    Utmp[0] = sqW
    Vtmp[0] = 1.0

    for index, item in enumerate(Utmp[START:], START):
        Vtmp[index] = Utmp[index-1]*Vtmp[index-1]
        Utmp[index] = Utmp[index - 1] * W

    D = np.ones(L, dtype=complex)
    for i in range(M):
        D[i] = np.conj(Vtmp[i])

    for i in range(N):
        D[L-1-i] = np.conj(Vtmp[i+1])


    D = np.fft.fft(D)

    return A,B,D

# params = z_stack_params()

# calculate auxiliary vectors for chirpz
