import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.insert(1, '../../vectorpsf')
import torch
from vectorize_torch_simulation import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
    Model_vectorial_psf_simflux, Model_vectorial_psf, LM_MLE_simflux, LM_MLE
from config_simulation import sim_params
torch.cuda.empty_cache()

def show_napari(img):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)


# Init variables
save_path = './'
pitch = 494
p_lat = 452
astigmatism = np.array([30,60,90])/1000*715 # this is atigmatism actually
z_array =  np.linspace(-300,300,60)
depth = 300

# empty arrays
improv_matrix = np.zeros((len(z_array),len(astigmatism)))
improv_matrixx= np.zeros((len(z_array),len(astigmatism)))
improv_matrixy= np.zeros((len(z_array),len(astigmatism)))
clrb_matrix_smlmz = np.zeros((len(z_array),len(astigmatism)))
clrb_matrix_smlmy = np.zeros((len(z_array),len(astigmatism)))
clrb_matrix_smlmx = np.zeros((len(z_array),len(astigmatism)))
clrb_matrix_sfz = np.zeros((len(z_array),len(astigmatism)))
clrb_matrix_sfx = np.zeros((len(z_array),len(astigmatism)))
clrb_matrix_sfy = np.zeros((len(z_array),len(astigmatism)))

check_esti = False
roisize_arr=[30]

for roi in tqdm(range(len(roisize_arr))):
    for z_i in tqdm(range(len(z_array))):
        for modi in range(len(astigmatism)):
            with torch.no_grad():

                mod=[]
                params = sim_params(depth=depth,astig=astigmatism[modi],roisize=roisize_arr[roi])
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
                numbeads = 150
                wavevector, wavevectorzimm, Waberration, all_zernikes, PupilMatrix = get_pupil_matrix(NA, zvals, refmed,
                                                                                  refcov, refimm, refimmnom, Lambda,Npupil, abberations, dev)


                # region simulation
                dx = torch.ones((numbeads,1))*0
                dy = torch.ones((numbeads,1))*0
                dz = torch.ones((numbeads,1))*z_array[z_i]
                Nphotons = torch.ones((numbeads,1))*2800
                Nbackground = torch.ones((numbeads,1))*8
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

                theta = ground_truth*1
                theta[:,0:2] = (theta[:,[1,0]] - Mx/2)*pixelsize
                theta[:, 2] = z_array[z_i]
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
                random_phase = np.random.randint(0,100,np.size(mod,0))/100*2*np.pi
                random_phase = np.tile(random_phase[:,None],[1,3])
                mod[:, : ,4] = (mod[:, : ,4] + random_phase)%(2*np.pi)
                mod_ = torch.from_numpy(np.asarray(mod)).to(dev)
                thetamin, thetamax = thetalimits(abberations,Lambda, Mx, My,pixelsize,zspread, dev, zstack= False)
                thetaretry = theta * 1
                param_range = torch.concat((thetamin[...,None], thetamax[...,None]),dim=1)
                model = Model_vectorial_psf()
                mle = LM_MLE(model, lambda_=1e-3, iterations=40,
                             param_range_min_max=param_range, tol=torch.tensor([1e-2,1e-2,1e-6,1e-2,1e-2]).to(dev))

                params_smlm = theta*1
                params_ = theta
                mu_est, dmu = model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                                            zmax, K, N,
                                            M, L, Ax, Bx,
                                            Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, params_, dev, zstack,
                                            wavevector, wavevectorzimm, all_zernikes, PupilMatrix)
                # with ground truth
                mu_mod, dmu  = model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                            zmax, K, N,
                            M, L, Ax, Bx,
                            Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, ground_truth, dev, zstack,
                            wavevector, wavevectorzimm, all_zernikes, PupilMatrix)

                crlb_ = compute_crlb(mu_mod,dmu)
                errors = ground_truth - params_smlm
                prec_smlm = errors.std(0).cpu().detach().numpy()
                rmsd_smlm = torch.sqrt((errors ** 2).mean(0)).cpu().detach().numpy()
                crlb_smlm = (torch.mean(crlb_, dim=0)).cpu().detach().numpy()
                mu_num = mu_mod.cpu().detach().numpy()
                mu_est = mu_est.cpu().detach().numpy()
                simflux_model = Model_vectorial_psf_simflux(Mx)

                roi_pos = np.zeros(np.shape(theta[:,[0,1]]))
                roi_pos_ = torch.tensor(roi_pos).to(dev)

                ground_truth[:, 0:2] = ground_truth[:, 0:2]/ pixelsize + Mx / 2
                ground_truth[:, 2] = ground_truth[:, 2] /1000
                ground_truth[:, [1, 0]] = ground_truth[:, [0, 1]]

                params_[:, 0:2] = params_[:, 0:2]/ pixelsize + Mx / 2
                params_[:, 2] = params_[:, 2] /1000
                params_[:, [1, 0]] = params_[:, [0, 1]]


                mu_simflux, dmudtheta_simflux = simflux_model.forward(NA, zvals, refmed, refcov, refimm, refimmnom, Lambda, Npupil, abberations, zmin,
                                zmax, K, N,
                                M, L, Ax, Bx,
                                Dx, Ay, pixelsize, By, Dy, Mx, My, numparams_fit, ground_truth, dev, zstack,
                                wavevector, wavevectorzimm, all_zernikes, PupilMatrix, mod_, roi_pos_,torch.tensor(0).to(dev))

                mle = LM_MLE_simflux(simflux_model, lambda_=0.01, iterations=40,
                                     param_range_min_max=param_range, tol=torch.tensor(([1e-3,1e-3,1e-7,1e-3,1e-3])).to(dev))



                e = theta*1
                params_sf = theta.detach().cpu().numpy()

                filter = abs(params_sf[:,:])<np.repeat(np.array([12,12,0.5,5000,500])[None,...], numbeads,axis=0)
                filter = np.sum(filter, axis=1)
                filter = filter==5
                errors_sf = ground_truth[filter,:] - e[filter,:]
                prec_sf = errors_sf.std(0).cpu().detach().numpy()
                rmsd_sf = torch.sqrt((errors_sf ** 2).mean(0)).cpu().detach().numpy()

                crlb_sf = compute_crlb(mu_simflux.type(torch.float),dmudtheta_simflux.type(torch.float))
                crlbsf = (torch.mean(crlb_sf, dim=0)).cpu().detach().numpy()

                prec_sf = prec_sf *np.array([65,65,1000,1,1])
                rmsd_sf = rmsd_sf *np.array([65,65,1000,1,1])
                crlbsf = crlbsf * np.array([65, 65, 1000, 1, 1])

                improv_matrix[z_i, modi] = crlb_smlm[2]/(crlbsf[2])
                improv_matrixx[z_i, modi] = crlb_smlm[1] / (crlbsf[0])
                improv_matrixy[z_i, modi] = crlb_smlm[0] / (crlbsf[1])
                clrb_matrix_sfz[z_i, modi] = crlbsf[2]
                clrb_matrix_sfx[z_i, modi] = crlbsf[0]
                clrb_matrix_sfy[z_i, modi] = crlbsf[1]

                clrb_matrix_smlmz[z_i, modi] = crlb_smlm[2]
                clrb_matrix_smlmx[z_i, modi] = crlb_smlm[1]
                clrb_matrix_smlmy[z_i, modi] = crlb_smlm[0]

    # plot grahps
    cm = 1/2.54
    import matplotlib
    from matplotlib import rc, font_manager
    from matplotlib.ticker import FormatStrFormatter
    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal', 'size' : 12}
    rc('text', usetex=True)
    rc('font',**fontProperties)
    from scipy.interpolate import UnivariateSpline
    fig, ax = plt.subplots(figsize=(9*cm, 6*cm))
    plt.rcParams['text.usetex'] = True
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    for ii in range(np.size(improv_matrix,1)):
        x = z_array
        y = clrb_matrix_smlmz[:,ii]
        B_spline_coeff = UnivariateSpline(x, y, k=5, s=1)
        X_Final = np.linspace(x.min(), x.max(), 500)
        Y_Final = B_spline_coeff(X_Final)


        ax.plot(X_Final,Y_Final,label=r'$Z_2^2$ = '+str(int(astigmatism[ii]/715*1000)) + r'm$\lambda$' )
    ax.set_xticks([-300, -200, -100, 0, 100, 200, 300])
    ax.set_xlabel(r' z from focal plane [nm]')
    ax.set_ylabel(r'Astig. PSF CRLB in z [nm] ')

    fig.tight_layout(pad=0.3)
    plt.savefig(save_path+ 'no_scale_astig'+str(roisize_arr[roi])+'_smlm'+str(depth)+'.png', dpi=600)
    plt.show()
    plt.close()



    from matplotlib.ticker import FormatStrFormatter
    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal', 'size' : 12}


    fig, ax = plt.subplots(figsize=(9*cm, 6*cm))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))


    ax.tick_params(axis='both', which='major', labelsize=10)
    #ax.set_ylim(2,35)



    for ii in range(np.size(improv_matrix,1)):
        x = z_array
        y = clrb_matrix_sfz[:,ii]
        B_spline_coeff = UnivariateSpline(x, y,k=5, s=1)
        X_Final = np.linspace(x.min(), x.max(), 500)
        Y_Final = B_spline_coeff(X_Final)

        ax.plot(X_Final,Y_Final,label=r'$Z_2^2$ = '+str(int(astigmatism[ii]/715*1000)) + r'm$\lambda$' )
        ax.set_xticks([-300, -200, -100, 0, 100, 200, 300])
    ax.set_xlabel(r"z from focal plane [nm]")
    ax.set_ylabel(r'ZIMLFUX CRLB in z [nm]')

    fig.tight_layout(pad=0.3)
    plt.savefig(save_path + 'no_scale_astig'+str(roisize_arr[roi])+'_sf'+str(depth)+'.png', dpi=600)
    plt.show()

