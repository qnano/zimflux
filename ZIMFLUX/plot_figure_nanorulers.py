
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



def plot_nanorulerclusters(plotdata, result_folder):

    std_smlm = plotdata[0][0]
    std_sf = plotdata[0][1]
    stdx_smlm = plotdata[0][2]
    stdx_sf = plotdata[0][3]
    stdy_smlm = plotdata[0][4]
    stdy_sf = plotdata[0][5]
    mean_array_cluster0_smlm = plotdata[2][0]
    mean_array_cluster1_smlm = plotdata[2][0]
    mean_array_cluster0_sf = plotdata[3][0]
    mean_array_cluster1_sf = plotdata[3][0]

    angle_array_smlm = plotdata[1][0]
    distance_array_smlm = plotdata[1][1]
    angle_array_sf = plotdata[1][2]
    distance_array_sf = plotdata[1][3]

    cm = 1/2.54
    import matplotlib
    from matplotlib import rc, font_manager
    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal', 'size' : 12}
    rc('text', usetex=True)
    rc('font',**fontProperties)



    fig, ax = plt.subplots(figsize=(9*cm, 6*cm))

    ax.tick_params(axis='both', which='major', labelsize=10)
    # plt.rcParams['text.usetex'] = True



    std_data_smlm = {'Precision [nm]': np.array(std_smlm)[np.array(std_sf) < 30],
                     'Mode': ['SMLM' for i in range(len(np.array(std_sf)[np.array(std_sf) < 30]))],
                     'Direction': 'z'}
    std_data_smlmx = {'Precision [nm]': np.array(stdx_smlm)[np.array(std_sf) < 30],
                     'Mode': ['SMLM' for i in range(len(np.array(std_sf)[np.array(std_sf) < 30]))],
                     'Direction': 'x'}
    std_data_smlmy = {'Precision [nm]': np.array(stdy_smlm)[np.array(std_sf) < 30],
                     'Mode': ['SMLM' for i in range(len(np.array(std_sf)[np.array(std_sf) < 30]))],
                     'Direction': 'y'}


    std_data_sf = {'Precision [nm]': np.array(std_sf)[np.array(std_sf) < 30],
                     'Mode': ['ZIMFLUX' for i in range(len(np.array(std_sf)[np.array(std_sf) < 30]))],
                     'Direction': 'z'}
    std_data_sfx = {'Precision [nm]': np.array(stdx_sf)[np.array(std_sf) < 30],
                     'Mode': ['ZIMFLUX' for i in range(len(np.array(std_sf)[np.array(std_sf) < 30]))],
                     'Direction': 'x'}
    std_data_sfy = {'Precision [nm]': np.array(stdy_sf)[np.array(std_sf) < 30],
                     'Mode': ['ZIMFLUX' for i in range(len(np.array(std_sf)[np.array(std_sf) < 30]))],
                     'Direction': 'y'}

    df = pd.concat([pd.DataFrame(std_data_smlm),pd.DataFrame(std_data_smlmx),pd.DataFrame(std_data_smlmy),
                     pd.DataFrame(std_data_sf),pd.DataFrame(std_data_sfx),pd.DataFrame(std_data_sfy)])
    x = 'Direction'
    y = 'Precision [nm]'
    p = sns.swarmplot(x=x, y=y, data=df, hue='Mode',  dodge=True,size=1.5)

    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(pad=0.1)
    plt.show()

    temp = np.resize([0,1], np.size(std_sf)).astype(int)
    y_sf = np.concatenate((np.array(std_sf)[temp==0],np.array(std_sf)[temp==1]))
    y_smlm = np.concatenate((np.array(std_smlm)[temp==0],np.array(std_smlm)[temp==1]))

    cm = 1/2.54
    import matplotlib
    from matplotlib import rc, font_manager
    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal', 'size' : 12}
    rc('text', usetex=True)
    rc('font',**fontProperties)
    plt.figure(figsize=(8*cm,6*cm))
    plt.scatter(angle_array_smlm/np.pi*180, distance_array_smlm, label='Astig. PSF', facecolor='#1f77b4', alpha=0.5)

    slope_angle_smlm = np.polyfit(angle_array_smlm, distance_array_smlm, 1)
    plt.plot(np.array([0, 1.6])/np.pi*180, slope_angle_smlm[0] * np.array([0, 1.6]) + slope_angle_smlm[1],
             linestyle=':', markersize=0, color='#1f77b4', linewidth=4)

    plt.scatter(angle_array_sf/np.pi*180, np.array(distance_array_sf), label='ZIMFLUX', alpha=0.5, facecolor='#ff7f0e')
    slope_angle_sf = np.polyfit(angle_array_sf, distance_array_sf, 1)
    plt.plot(np.array([0, 1.6])/np.pi*180, slope_angle_sf[0] * np.array([0, 1.6]) + slope_angle_sf[1], linestyle=':',
             markersize=0, color='#ff7f0e', linewidth=4)

    plt.xlabel(r'Angle $\theta$ [$^\circ$]')
    plt.ylabel('Length $l$ [nm]')
    plt.legend(fontsize=10)

    #plt.grid()

    plt.tight_layout(pad=0.2)
    plt.savefig(result_folder+ 'Angle_length' + str(slope_angle_smlm[0]) +
                str(slope_angle_smlm[1])+ 'sf_' + str(slope_angle_sf[0]) +
                str(slope_angle_sf[1])+'.png', dpi=600)

    plt.show()


    def gauss(x, a, x0, sigma, c):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) + c)
    from scipy.optimize import curve_fit
    cm = 1/2.54
    import matplotlib
    from matplotlib import rc, font_manager
    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal', 'size' : 12}
    rc('text', usetex=True)
    rc('font',**fontProperties)
    plt.figure(figsize=(10*cm,10*cm))
    plt.hist(np.concatenate((mean_array_cluster0_smlm[:,2], mean_array_cluster1_smlm[:,2])), label='SMLM', facecolor='#1f77b4', alpha=0.5)


    plt.xlabel(r'pos')
    plt.ylabel('pos')
    plt.legend(fontsize=10, frameon=False)

    #plt.grid()

    # plt.legend()
    # plt.title('mean smlm pos = ' + str(np.round(popt_smlm[1], 2)) + ' sf = ' + str(np.round(popt_sf[1], 2)))
    plt.tight_layout(pad=0)
    plt.show()

    cm = 1/2.54
    import matplotlib
    from matplotlib import rc, font_manager
    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal', 'size' : 12}
    rc('text', usetex=True)
    rc('font',**fontProperties)
    plt.figure(figsize=(8*cm,6*cm))
    binwidth_cluster = 5



    n, bins, _ = plt.hist(np.array(distance_array_smlm), label='Astig. PSF',  facecolor='#1f77b4', alpha=0.5,
              bins=np.arange(40, 110, binwidth_cluster))


    popt_smlm, pcov_smlm = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n, p0=[max(n),80, 10, 0],
                                     maxfev=2000)
    plt.plot(np.linspace(50, 110, 300),
             gauss(np.linspace(50, 110, 300), *popt_smlm), c='#1f77b4',)
    n, bins, _ = plt.hist(distance_array_sf, label='ZIMFLUX', alpha=0.5, facecolor='#ff7f0e',
                bins= np.arange(40, 110, binwidth_cluster)
    )


    popt_sf, pcov_sf = curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n, p0=[max(n), 80, 10, 0],
                             maxfev=2000, method='lm')
    plt.plot(np.linspace(50, 110, 300),
             gauss(np.linspace(50, 110, 300), *popt_sf),
              c='#ff7f0e')

    plt.xlabel(r'Length $l$ [nm]')

    plt.ylabel('Counts')
    plt.legend(fontsize=10)

    plt.tight_layout(pad=0.2)
    plt.savefig(result_folder+'clusterdata_std' +
                str(popt_sf[1].round(1)) + ' ' + str(popt_sf[2].round(1)) + ' ' + str(popt_smlm[1].round(1)) + ' ' + str(
        popt_smlm[2].round(1))
                + '.png', dpi=600)

    plt.show()


