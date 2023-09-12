from photonpy.simflux import ZimfluxProcessor
from photonpy import Gauss3D_Calibration
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './ZIMFLUX')
from plot_figure_nanorulers import plot_nanorulerclusters
import os
# -------------------------------- User inputs --------------------------------------
cfg = {
   'roisize':16,
   'spotDetectSigma': 3,
   'maxSpotsPerFrame': 2000,
   'detectionThreshold': 6,  # 6!
    'patternFrames': [[0,1,2]],
   'gain': 0.36,
   'offset': 100,
   'pixelsize' : 65, # nm/pixel
   'startframe':0,
   'maxframes': 0,
    'use3D': True,
    'chisq_threshold': 3,
    'ref_index': 1.33,
    'wavelength': 640,
    'debugMode' :False,
    'usecuda': True,
    'depth': 60
}

# absolute path to data
path = 'E:/Zimflux/3Dnanorulers/nanorulers_simflux_10ms_jan13_MMStack_Default.ome.tif'

# absolute path to gain and offset data
cfg['gain'] = 'E:/Zimflux/calibration/bright_2_MMStack_Default.ome.tif'
cfg['offset'] = 'E:/Zimflux/calibration/dark_1_MMStack_Default.ome.tif'


# calibration for astigmatic PSF to make initial guess
calib_fn = os.path.dirname(os.path.realpath('__file__'))+ '/astigmatist_gausscalib20nm.yaml'
calib = Gauss3D_Calibration.from_file(calib_fn)
cfg['psf_calib'] = calib_fn

# intial wavevector in axial direction and number of iterations refinement
init_kz = 13.2
num_iterations = 20
# -------------------------------- end user inputs --------------------------------------

# Detect spots
sp = ZimfluxProcessor(path, cfg)
sp.detect_rois(ignore_cache=False, roi_batch_size=200)

# fit vectorial PSF and find the drift
sumfit = sp.spot_fitting( vectorfit=True, check_pixels=False)
sp.drift_correct(framesPerBin=100, display=True,
                 outputfn=sp.resultsdir + 'drift')
plt.show()

# find pitch and direction pattern
sp.estimate_angles(1, pitch_minmax_nm=[400,500], dft_peak_search_range=0.004)

# refine axial pitch
sp.refine_kz(init_kz,num_iterations)
#
# estimate phases
sp.estimate_phases(10, iterations=10)

# zimflux estimation
e, traces, iterss = sp.gaussian_vector_2_simflux(lamda=0.1, iter=40, pick_percentage=1, depth=sp.depth)

# apply drift
sp.zf_ds_undrift = sp.zf_ds[:]
sp.zf_ds_undrift.applyDrift(sp.drift)
sp.zf_ds_undrift.save(sp.resultsdir + "zimflux_undrift" +".hdf5")
sp.sum_ds_filtered_undrift = sp.sum_ds_filtered[:]
sp.sum_ds_filtered_undrift.applyDrift(sp.drift)
sp.sum_ds_filtered_undrift.save(sp.resultsdir + "smlm_undrift" +".hdf5")

# find cluster data
clusterdata= sp.cluster_picassopicksv2(drift_corrected=True)

# plot clusterdata
plot_nanorulerclusters(clusterdata,sp.resultsdir)

