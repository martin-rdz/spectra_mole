import datetime, math
import os
import numpy as np
#from Scientific.IO import NetCDF
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spectra_mole.VIS_Colormaps as VIS_Colormaps
import spectra_mole.viridis as viridis


class pltRange():
    def __init__(self, time=[0, -1], height=[0, -1]):
        self.t_bg = time[0]
        self.t_ed = time[-1]
        self.h_bg = height[0]
        self.h_ed = height[-1]


# In[4]:


filelist = os.listdir("../output")
#print(filelist)
filelist = [f for f in filelist if "mole_output" in f]
filelist = sorted(filelist)
filename = '../output/20150617_1459_mole_output.nc'
#filename = '../output/20150617_1700_mole_output.nc'
#filename = '../output/20150611_1830_mole_output.nc'
#filename = '../output/20150617_2014_mole_output.nc'
print(len(filelist))
print("filename ", filename)


def run_filename(filename):
    savepath = '../plots/region'
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    f = netCDF4.Dataset(filename, 'r')
    time_list = f.variables["timestamp"][:]
    range_list = f.variables["range"][:]

    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in time_list]

    # this is the last valid index
    jumps = np.where(np.diff(time_list)>15)[0]
    for ind in jumps[::-1].tolist():
        print(ind)
        # and modify the dt_list
        dt_list.insert(ind+1, dt_list[ind]+datetime.timedelta(seconds=10))

    rect = pltRange(time=[100, 500], height=[10, 40])
    rect = pltRange(time=[0, -1], height=[0, -1])
    #rect = pltRange(time=[0, 676], height=[0, -1])
    #rect = pltRange(time=[170, -169], height=[0, -1])

    #rect = pltRange(time=[0, 1183], height=[0, -1])
    # case 0611
    # rect = pltRange(time=[0, 341], height=[0, -1])
    # case 0625
    #rect = pltRange(time=[300, 1190], height=[0, -1])
    # second cloud 0130-0400
    #rect = pltRange(time=[170, 680], height=[0, 65])
    # second cloud 0530-0800
    # rect = pltRange(time=[851, 1361], height=[0, 65])
    # case 0801
    #rect = pltRange(time=[2571, 3086], height=[0, -1])
    # case 0612
    #rect = pltRange(time=[0, 170], height=[0, 60])
    #print(time_list[:-1] - time_list[1:])

    quality_flag = f.variables["quality_flag"][:]
    wipro_vel = f.variables["v"][:].copy()
    wipro_vel_fit = f.variables['v_fit'][:].copy()
    print(f.variables.keys())
    wipro_ucorr_vel = f.variables["v_raw"][:]
        
    tg_v_term = False
    if 'mira_v_term' in f.variables.keys():
        tg_v_term = True
        print('v_term', tg_v_term)
        v_term = f.variables["mira_v_term"][:]
        for ind in jumps[::-1].tolist():
            v_term = np.insert(v_term, ind+1, np.full(height_list.shape, -99.), axis=0)
        v_term = np.ma.masked_less(v_term, -90., copy=True)

    wpZ_Bragg = f.variables["Z"][:]
    wpZ_raw = f.variables["Z_raw"][:]
    mira_Z = f.variables["Z_cr"][:]
    mira_Z = np.ma.masked_invalid(mira_Z)
    cal_const = f.variables["est_cal_const"][:]
    cal_corr = f.variables["cal_corr"][:]
    sigma_b = f.variables["sigma_broadening"][:]

    wipro_width = f.variables["width"][:]
    width_raw = f.variables["width_raw"][:]
    width_cr = f.variables["width_cr"][:]

    error_diff = f.variables["error_diff"][:]
    error_fit = f.variables["error_fit"][:]

    for ind in jumps[::-1].tolist():
        print(ind)
        # add the fill array
        quality_flag = np.insert(quality_flag, ind+1, np.full(range_list.shape, -1), axis=0)
        wipro_vel = np.insert(wipro_vel, ind+1, np.full(range_list.shape, -99.), axis=0)
        wipro_vel_fit = np.insert(wipro_vel_fit, ind+1, np.full(range_list.shape, -99.), axis=0)
        wipro_ucorr_vel = np.insert(wipro_ucorr_vel, ind+1, np.full(range_list.shape, -99.), axis=0)
        wpZ_Bragg = np.insert(wpZ_Bragg, ind+1, np.full(range_list.shape, -200), axis=0)
        wpZ_raw = np.insert(wpZ_raw, ind+1, np.full(range_list.shape, -200), axis=0)
        mira_Z = np.insert(mira_Z, ind+1, np.full(range_list.shape, -200), axis=0)
        cal_const = np.insert(cal_const, ind+1, np.full(range_list.shape, 1e-200), axis=0)
        sigma_b = np.insert(sigma_b, ind+1, np.full(range_list.shape, -1), axis=0)
        
        wipro_width = np.insert(wipro_width, ind+1, np.full(range_list.shape, -99.), axis=0)
        width_raw = np.insert(width_raw, ind+1, np.full(range_list.shape, -99.), axis=0)
        width_cr = np.insert(width_cr, ind+1, np.full(range_list.shape, -99.), axis=0)
        error_diff = np.insert(error_diff, ind+1, np.full(range_list.shape, -99.), axis=0)
        error_fit = np.insert(error_fit, ind+1, np.full(range_list.shape, -99.), axis=0)

    cal_const = np.ma.masked_less_equal(cal_const, 1e-150, copy=True)
    quality_flag = np.ma.masked_less(quality_flag, 0., copy=True)
    wipro_vel = np.ma.masked_less(wipro_vel, -90., copy=True)
    wipro_ucorr_vel = np.ma.masked_less(wipro_ucorr_vel, -90., copy=True)
    wpZ_Bragg = np.ma.masked_less_equal(wpZ_Bragg, -200, copy=True)
    wpZ_raw = np.ma.masked_less_equal(wpZ_raw, -200, copy=True)
    mira_Z = np.ma.masked_less_equal(mira_Z, -200, copy=True)
    cal_const = np.ma.masked_less_equal(cal_const, 1e-200, copy=True)
    sigma_b = np.ma.masked_less_equal(sigma_b, -1, copy=True)

    wipro_width = np.ma.masked_less(wipro_width, -90., copy=True)
    width_raw = np.ma.masked_less(width_raw, -90., copy=True)
    width_cr = np.ma.masked_less(width_cr, -90., copy=True)
    error_diff = np.ma.masked_less(error_diff, -90., copy=True)
    error_fit = np.ma.masked_less(error_fit, -90., copy=True)

    wipro_vel = np.ma.masked_where(quality_flag > 3.0, wipro_vel)
    wipro_vel_fit = np.ma.masked_where(quality_flag > 3.0, wipro_vel_fit)
    #quality_flag = np.ma.masked_where(quality_flag >= 2.0, quality_flag)
    np.set_printoptions(threshold='nan')
    wipro_ucorr_vel = np.ma.masked_invalid(wipro_ucorr_vel)


    # In[5]:


    #print(f.variables)
    print(f.variables.keys())

    #print(f.variables['v'][:])

    print(f.variables['v'].units)

    print(f.variables['quality_flag'].comment)

    print('creation time', f.creation_time)
    #print('settings ', f.settings)


    # In[6]:


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(wipro_vel[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=VIS_Colormaps.carbonne_map, vmin=-1.5, vmax=1.5)
    cbar = fig.colorbar(pcmesh)
    #ax.set_xlim([dt_list[0], dt_list[-1]])
    #ax.set_ylim([height_list[0], height_list[-1]])
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Velocity [m s$\mathregular{^{-1}}$]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))


    # ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval=5))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_vel_corr.png"
    fig.savefig(savename, dpi=250)


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(wipro_ucorr_vel[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=VIS_Colormaps.carbonne_map, vmin=-1.5, vmax=1.5)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Velocity [m s$\mathregular{^{-1}}$]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    # ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval=5))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_vel_wp.png"
    fig.savefig(savename, dpi=250)


    # In[7]:


    quality_flag[quality_flag == 5] = 4

    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(quality_flag[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=VIS_Colormaps.cloudnet_map,
                        vmin=-0.5, vmax=10.5)
    cbar = fig.colorbar(pcmesh, ticks=[0, 1, 2, 3, 4, 5, 6])
    cbar.ax.set_yticklabels(["not influenced", "correction reliable",
                            "plankton", "low SNR", 
                            "noisy spectrum\nmelting layer",
                            "",
                            ""])

    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    #cbar.ax.set_ylabel("Flag", fontweight='semibold', fontsize=15)

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=13,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_quality_flag.png"
    plt.subplots_adjust(right=0.9)
    #plt.tight_layout()
    fig.savefig(savename, dpi=250)


    # In[8]:


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(np.log10(cal_const[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed])),
                        cmap='gist_rainbow', vmin=-16.5, vmax=-13.5)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("RWP Calibration Constant [log10]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_system_parameter.png"
    fig.savefig(savename, dpi=250)


    # In[9]:


    zmax = 10
    #zmax = 40
    cmap = viridis.viridis
    cmap = 'jet'
    print('maximum wind profiler ', np.max(wpZ_raw[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]))
    am = np.argmax(wpZ_raw[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed])
    am = np.unravel_index(am, wpZ_raw[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed].shape)
    print(dt_list[rect.t_bg:rect.t_ed][am[0]], range_list[rect.h_bg:rect.h_ed][am[1]])
    print('cloud radar ', np.nanmax(10 * np.log10(mira_Z[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed])))
    am = np.nanargmax(mira_Z[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed])
    am = np.unravel_index(am, mira_Z[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed].shape)
    print(dt_list[rect.t_bg:rect.t_ed][am[0]], range_list[rect.h_bg:rect.h_ed][am[1]])

    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(wpZ_raw[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=-35, vmax=zmax)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Reflectivity [dBZ]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_wp_total_reflectivity_jet.png"
    fig.savefig(savename, dpi=250)


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(wpZ_Bragg[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=-35, vmax=zmax)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Reflectivity [dBZ]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_wp_corr_reflectivity_jet.png"
    fig.savefig(savename, dpi=250)

    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(mira_Z[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=-35, vmax=zmax)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Reflectivity [dBZ]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_mira_reflectivity.png"
    fig.savefig(savename, dpi=250)


    # In[10]:


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(sigma_b[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        # normally the range is 1.5 to 4
                        cmap='gist_rainbow', vmin=1.5, vmax=7)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("sigma_blure [px]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_sigma_blure.png"
    fig.savefig(savename, dpi=250)


    # In[11]:


    if tg_v_term:
        
        fig, ax = plt.subplots(1, figsize=(10, 5.7))
        pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                            height_list[rect.h_bg:rect.h_ed],
                            np.transpose(v_term[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                            cmap=VIS_Colormaps.carbonne_map, vmin=-2, vmax=2)
        cbar = fig.colorbar(pcmesh)
        ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
        ax.set_ylim([height_list[rect.h_bg], height_list[rect.h_ed-1]])
        ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
        ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
        cbar.ax.set_ylabel("Terminal velocity [m s$\mathregular{^{-1}}$]", fontweight='semibold', fontsize=15)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
        #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
        #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval=2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

        ax.tick_params(axis='both', which='major', labelsize=14, 
                    right=True, top=True, width=2, length=5)
        ax.tick_params(axis='both', which='minor', width=1.5, 
                    length=3.5, right=True, top=True)
        cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                            width=2, length=4)

        savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")                + "_terminal_vel.png"
        fig.savefig(savename, dpi=250)

        np.max(v_term)


    # In[12]:


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(wipro_vel_fit[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=VIS_Colormaps.carbonne_map, vmin=-1.5, vmax=1.5)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Velocity [m s$\mathregular{^{-1}}$]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    # ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval=5))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_vel_wp_fit.png"
    fig.savefig(savename, dpi=250)


    diff_estimates = wipro_vel - wipro_vel_fit
    diff_estimates = np.ma.masked_where(quality_flag == 0, diff_estimates)
    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(diff_estimates[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=VIS_Colormaps.carbonne_map, vmin=-1., vmax=1.0)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Differenece between the estimates [m s$\mathregular{^{-1}}$]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval=2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_vel_fit1.png"
    #fig.savefig(savename, dpi=250)


    # In[13]:


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(wipro_width[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=0.01, vmax=1)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Spectral width [dBZ]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_wp_corr_width.png"
    fig.savefig(savename, dpi=250)


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(width_raw[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=0.01, vmax=1)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Spectral width [dBZ]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_wp_width.png"
    fig.savefig(savename, dpi=250)


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(width_cr[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=0.01, vmax=1)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Spectral width [dBZ]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_mira_width.png"
    fig.savefig(savename, dpi=250)


    # In[14]:


    print(error_diff.max())

    cmap=VIS_Colormaps.carbonne_map
    cmap='RdBu'

    error_diff = np.ma.masked_greater(error_diff, 2)
    error_diff = np.ma.masked_where(np.logical_or(quality_flag == 0, wipro_vel.mask, error_diff > 1), error_diff)
    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    ax.patch.set_facecolor('darkgrey')
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(error_diff[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=-0.2, vmax=0.2)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Vertical air velocity bias [m s$\mathregular{^{-1}}$]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval=2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_error_diff.png"
    fig.savefig(savename, dpi=250)


    error_fit = np.ma.masked_greater(error_fit, 2)
    error_fit = np.ma.masked_where(np.logical_or(quality_flag == 0, wipro_vel_fit.mask), error_fit)
    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    ax.patch.set_facecolor('darkgrey')
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(error_fit[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
                        cmap=cmap, vmin=-0.2, vmax=0.2)
    cbar = fig.colorbar(pcmesh)
    ax.set_xlim([dt_list[rect.t_bg], dt_list[rect.t_ed-1]])
    ax.set_ylim([range_list[rect.h_bg], range_list[rect.h_ed-1]])
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Vertical air velocity bias [m s$\mathregular{^{-1}}$]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    #ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,10)))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0,3,6,9,12,15,18,21]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,30]))
    #ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(interval=2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))

    ax.tick_params(axis='both', which='major', labelsize=14, 
                right=True, top=True, width=2, length=5)
    ax.tick_params(axis='both', which='minor', width=1.5, 
                length=3.5, right=True, top=True)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_error_fir.png"
    fig.savefig(savename, dpi=250)

for filename in filelist:
    run_filename("../output/" + filename)