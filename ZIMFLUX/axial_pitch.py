import sys
sys.path.insert(1, '/projects/vectorpsf')
import pandas as pd
import yaml
from photonpy.simflux import SimfluxProcessor
import photonpy.utils.multipart_tiff as read_tiff
from photonpy import Context,Gauss3D_Calibration
import numpy as np
import os
import pandas as pd
#import trackpy
import tifffile
import photonpy.smlm.process_movie as process_movie

sys.path.insert(1, '../vectorpsf')
import matplotlib.pyplot as plt
import torch
from vectorize_torch import poissonrate, get_pupil_matrix, initial_guess, thetalimits, compute_crlb, \
    Model_vectorial_psf_simflux, Model_vectorial_psf, LM_MLE_simflux, LM_MLE

from config import sim_params


def show_napari(img):
    import napari
    #viewer = napari.gui_qt()
    viewer = napari.Viewer()
    viewer.add_image(img)

# for 2311 2 :    'startframe': 15000,
#    'maxframes': 8000,

# Configuration for spot detection and 2D Gaussian fitting
cfg = {
   'roisize':16,
   'spotDetectSigma': 3,
   'maxSpotsPerFrame': 100,
   'detectionThreshold': 3, #10 for datadec28
    #'detectionThreshold': 800,
    'patternFrames': [[0]],
    #'patternFrames': [[0,2,4,6],[1,3,5,7]],
    #'patternFrames': [[0,2,4],[1,3,5]],
   'gain': 0.45,
   'offset': 100,
   'pixelsize' : 65, # nm/pixel
   'startframe': 0,
   'maxframes': 0,
    'use3D': True,
   'chisq_threshold': 70,
    'ref_index': 1.33,
    'wavelength': 640}


calib_fn = 'E:/SIMFLUX Z/astigmatism_cal_100mlam_1/astigmatist_cal_100mlam_1_MMStack_Default.ome_gausscalib20nm.yaml'
calib = Gauss3D_Calibration.from_file(calib_fn)
cfg['psf_calib'] = calib_fn
_, cal = Gauss3D_Calibration.from_file_py(calib_fn)

direction = 'z'
path = "C:/data/SIMFLUX Z/"+direction+"shift_32nm_newsample_jan11/"
img_range = [5,20]
stepsize = 64

if direction == 'z':
    path = "E:\SIMFLUX Z\shifting_in_z_up/" # down is postive to negative
    img_range = [5, 35]
    stepsize = 30

dirnames = next(os.walk(path))[1]

concatfrequency  = np.array([])
colorcode  = np.array([])
concat_zpos = np.array([])
iter=0
iterarray = np.array([])

for dir in dirnames:
    for pattern in [0]:
        path_iter = path+dir+'/'+dir + '_MMStack_Default.ome.tif'
        # if direction=='x':
        #     pattern = pattern_arrayforx[iter]
        # if direction=='y':
        #     pattern = pattern_arrayfory[iter]


        sp = SimfluxProcessor(path_iter, cfg)
        sp.levmarparams_astig = np.array([1e3, 1e3, 1e3, 1e3, 1e3])

        if direction == 'z':
            image = tifffile.imread(path_iter)
            image_1pattern = image[np.arange(pattern, np.size(image, 0), 2), :, :]
            image_shiftcorrected = image_1pattern*1


        if direction == 'z':
            image_shiftcorrected = image_shiftcorrected[img_range[0]:img_range[1],:,:]
        else:
            image_shiftcorrected = image_shiftcorrected[img_range[0]:img_range[1], 0:230, 0:230]
        summed_image = np.mean(image_shiftcorrected,axis=0)
        tifffile.imwrite('E:/temp.tif',summed_image)
        sp.src_fn = 'E:/temp.tif'

        sp.detect_rois(ignore_cache=True, roi_batch_size=1)
        roi_ori, pixels = process_movie.load_rois(sp.rois_fn)

        tifffile.imwrite('E:/temp.tif',image_shiftcorrected)
        sp.pattern_frames = [[np.arange(0,int(img_range[1]-img_range[0]))]]
        static_roi, pxstack, roi_info, pxstack_summed1_pat1 = sp.fix_rois_laeteral(roi_ori)
        pxstack = np.clip((pxstack-np.min(pxstack))*0.33,0,np.inf)
        sum_fit_vector, traces_all, ibg_vec, ibg_traces_vec, iterations_vector_tot, \
        iterations_ibg_tot_1, checkpx1 = sp.gaussian_fitting_zstack(pxstack, vectorfit=True, depth = 0,check_pixels=True)

        sp.fit_sine(lateral = True,display=False, direction=direction, stepsize=stepsize)
        #         sp2.fit_sine()
        sp.filter_onR(min_mod=0.7, filter = 0.05,display=False) #0.5 for x direction
        if np.median(stepsize / abs(sp.sum_ds.params_sine_filtered[:, 0])) >0:

            concatfrequency = np.concatenate((concatfrequency, stepsize/sp.sum_ds.params_sine_filtered[:,0]))
            concat_zpos = np.concatenate((concat_zpos, sp.sum_ds.pos[:,2]))
            colorcode = np.concatenate((colorcode, np.ones(np.size(sp.sum_ds.params_sine_filtered[:,0]))*iter))
        if np.median(stepsize/sp.sum_ds.params_sine_filtered[:,0])<np.inf:
            iterarray = np.append(iterarray, iter)

        iter=iter+1


colorcode = colorcode[np.logical_and(concatfrequency<np.inf,concatfrequency>-np.inf)]
concat_zpos = concat_zpos
concatfrequency = concatfrequency[np.logical_and(concatfrequency<np.inf,concatfrequency>-np.inf)]

for i in range(iter):

    plt.hist(concatfrequency[colorcode==i], bins=15, label=str(i),  alpha=0.4)

plt.xlim(500,900)
plt.legend()
plt.show()

# new figure
cm = 1/2.54
import matplotlib
from matplotlib import rc, font_manager
from helper_plots import histo_gaussian

fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
    'weight' : 'normal', 'size' : 12}
rc('text', usetex=True)
rc('font',**fontProperties)


xlabel = 'Pitch in ' +direction + ' [nm]'
ylabel = 'Counts'
fig, ax = plt.subplots(figsize=(8*cm, 4.5*cm))
ax.tick_params(axis='both', which='major', labelsize=10)
#ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
fig.tight_layout()
xlim = [400,1200]
ylim = 0
_,mean, amp, std= histo_gaussian(ax, concatfrequency[np.logical_and(concatfrequency, concatfrequency)],
                            xlabel, ylabel, xlim, ylim, init_mean = 600,  binwidth = 50)

print('mean = ', mean)
#plt.savefig('C:/Users/Gebruiker/Desktop/inkscape/fig_lateral_pitch/lateralpitch' +direction + '.png', bbox_inches='tight', pad_inches=0.1, dpi=900)
plt.show()
