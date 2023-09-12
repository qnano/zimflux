# this script is for single images - and we assume the abberations are known. not for zstack
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../../vectorpsf')

import torch
from vectorize_torch_simulation import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
    Model_vectorial_psf_simflux, Model_vectorial_psf, LM_MLE_simflux, LM_MLE
from config_simulation import sim_params


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial


    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def show_napari(img):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)


astig_array = np.array([-30,-60,-90])/1000*715
angle_beam =  np.linspace(40,61,60)
save_path = './'


improv_matrix = np.zeros((len(angle_beam),len(astig_array)))
improv_matrixx= np.zeros((len(angle_beam),len(astig_array)))
improv_matrixy= np.zeros((len(angle_beam),len(astig_array)))
clrb_matrix_smlmz = np.zeros((len(angle_beam),len(astig_array)))
clrb_matrix_smlmy = np.zeros((len(angle_beam),len(astig_array)))
clrb_matrix_smlmx = np.zeros((len(angle_beam),len(astig_array)))
clrb_matrix_sfz = np.zeros((len(angle_beam),len(astig_array)))
clrb_matrix_sfx = np.zeros((len(angle_beam),len(astig_array)))
clrb_matrix_sfy = np.zeros((len(angle_beam),len(astig_array)))

check_esti = False

for angle_i in tqdm(range(len(angle_beam))):
    for astig_i in range(len(astig_array)):

        with torch.no_grad():
            angle = angle_beam[angle_i]
            pitch = 640 / np.absolute(1.33 * (1 - np.cos(np.arcsin(1.52 / 1.33 * np.sin(angle / 180 * np.pi)))))
            p_lat = 640 / np.absolute(1.33 * (np.sin(np.arcsin(1.52 / 1.33 * np.sin(angle / 180 * np.pi)))))

            depth = 300
            mod=[]
            params = sim_params(depth=depth,astig=astig_array[astig_i] )
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
            numbeads = 400
            wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals, refmed,
                                                                              refcov, refimm, refimmnom, Lambda,Npupil, abberations, dev)
            test = 1






            test = PupilMatrix.detach().cpu().numpy()

            # region simulation
            dx = torch.ones((numbeads,1))*0#(1-2*torch.rand((numbeads,1)))*2*params.pixelsize
            dy = torch.ones((numbeads,1))*0
            dz = (1-2*torch.rand((numbeads,1))) * 300
            Nphotons = torch.ones((numbeads,1))*2800#+ (1-2*torch.rand((numbeads,1)))*300
            Nbackground = torch.ones((numbeads,1))*7.6# + (1-2*torch.rand((numbeads,1)))*10

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

            theta =ground_truth*1
            theta[:,0:2] = (theta[:,[1,0]] - Mx/2)*pixelsize
            theta[:, 2] = 0#(theta[:, 2])*1000
            theta[:, 3] = Nphotons[0]
            theta[:, 4] = Nbackground[0]


            num_patternstot = 3
            mod_contrast = 0.85
            kx1 =  2*np.pi/p_lat * 65 * 0.5*np.sqrt(2)
            ky1 = 2*np.pi/p_lat * 65 * 0.5*np.sqrt(2)

            kz1 = 2 * np.pi / pitch* 1000
            phase = 0

            for i in range(num_patternstot):
                modtemp = [kx1, ky1, kz1, mod_contrast, phase, 1 / num_patternstot]
                if i != 0:
                    phase = phase + 2 * np.pi / (num_patternstot)


                modtemp = [kx1, ky1, kz1, mod_contrast, phase, 1 / num_patternstot]
                if i == 0:
                    mod = modtemp
                else:
                    mod = np.vstack((mod, modtemp))
            dev = 'cuda'
            mod = np.tile(mod[None, ...], [numbeads, 1, 1])
            random_phase = np.random.randint(0,100,np.size(mod,0))/100*0
            random_phase = np.tile(random_phase[:,None],[1,3])
            mod[:, : ,4] = (mod[:, : ,4] + random_phase)%(2*np.pi)
            mod_ = torch.from_numpy(np.asarray(mod)).to(dev)
            thetamin, thetamax = thetalimits(abberations,Lambda, Mx, My,pixelsize,zspread, dev, zstack= False)
            thetaretry = theta * 1
            param_range = torch.concat((thetamin[...,None], thetamax[...,None]),dim=1)
            model = Model_vectorial_psf()
            # with ground truth
            mu_mod, dmu  = model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                        zmax, K, N,
                        M, L, Ax, Bx,
                        Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, ground_truth, dev, zstack,
                        wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
            crlb_ = compute_crlb(mu_mod,dmu)
            crlb_smlm = (torch.mean(crlb_, dim=0)).cpu().detach().numpy()
            simflux_model = Model_vectorial_psf_simflux(Mx)
            roi_pos = np.zeros(np.shape(theta[:,[0,1]]))
            roi_pos_ = torch.tensor(roi_pos).to(dev)
            ground_truth[:, 0:2] = ground_truth[:, 0:2]/ pixelsize + Mx / 2
            ground_truth[:, 2] = ground_truth[:, 2] /1000
            ground_truth[:, [1, 0]] = ground_truth[:, [0, 1]]
            mu_simflux, dmudtheta_simflux = simflux_model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                            zmax, K, N,
                            M, L, Ax, Bx,
                            Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, ground_truth, dev, zstack,
                            wavevector, wavevectorzimm, all_zernikes, PupilMatrix, mod_, roi_pos_,torch.tensor(0).to(dev))
            crlb_sf = compute_crlb(mu_simflux.type(torch.float),dmudtheta_simflux.type(torch.float))
            crlbsf = (torch.mean(crlb_sf, dim=0)).cpu().detach().numpy()
            improv_matrix[angle_i, astig_i] = crlb_smlm[2]/(crlbsf[2])
            improv_matrixx[angle_i, astig_i] = crlb_smlm[1] / (crlbsf[0])
            improv_matrixy[angle_i, astig_i] = crlb_smlm[0] / (crlbsf[1])
            clrb_matrix_sfz[angle_i, astig_i] = crlbsf[2]
            clrb_matrix_sfx[angle_i, astig_i] = crlbsf[0]
            clrb_matrix_sfy[angle_i, astig_i] = crlbsf[1]

            clrb_matrix_smlmz[angle_i, astig_i] = crlb_smlm[2]
            clrb_matrix_smlmx[angle_i, astig_i] = crlb_smlm[1]
            clrb_matrix_smlmy[angle_i, astig_i] = crlb_smlm[0]

cm = 1/2.54

from matplotlib import rc, font_manager
fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
    'weight' : 'normal', 'size' : 12}
rc('text', usetex=True)
rc('font',**fontProperties)
from scipy.interpolate import UnivariateSpline
from matplotlib.ticker import FormatStrFormatter
fig, ax = plt.subplots(figsize=(9*cm, 6*cm))

ax.tick_params(axis='both', which='major', labelsize=10)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

for ii in range(np.size(improv_matrix,1)):

    x=np.flip(640 / np.absolute(1.33 * (1 - np.cos(np.arcsin(1.52 / 1.33 * np.sin(angle_beam / 180 * np.pi))))))
    y= np.flip(clrb_matrix_sfz[:,ii])
    B_spline_coeff = UnivariateSpline(x, y,s=30)
    X_Final = np.linspace(x.min(), x.max(), 500)
    Y_Final = B_spline_coeff(X_Final)

    ax.plot(X_Final,
             Y_Final*1000,label=r'$Z_2^2$ = '+str(int(astig_array[ii]/715*1000)) + r'm$\lambda$' )

ax.set_xlabel(r'Axial pitch [nm]')
ax.set_ylabel('ZIMFLUX CRLB in z [nm] ')

ax.set_ylim(6,12.5)


fig.tight_layout(pad=0.1)
plt.savefig(save_path + 'CRLBZIMFLUX'+str(depth)+'.png', dpi=600)
plt.show()
