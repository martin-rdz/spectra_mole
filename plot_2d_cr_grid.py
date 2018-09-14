
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


# In[14]:


filelist = os.listdir("../output")
print(filelist)
filelist = [f for f in filelist if "mole_terminal_output" in f]
filelist = sorted(filelist)
filename = '../output/20150617_1459_mole_output.nc'
#filename = '../output/20150617_1700_mole_output.nc'
#filename = '../output/20150611_1830_mole_output.nc'
#filename = '../output/20150617_2014_mole_output.nc'
filename = "../output/" + filelist[2]
print(len(filelist))


def run_filename(filename):
    print("filename ", filename)
    savepath = '../plots/region'
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    f = netCDF4.Dataset(filename, 'r')
    time_list = f.variables["timestamp"][:]
    range_list = f.variables["range"][:]
    print(f)

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
    v_term = f.variables["v_term"][:].copy()
    v_air = f.variables['v_air'][:].copy()


    for ind in jumps[::-1].tolist():
        print(ind)
        # add the fill array
        quality_flag = np.insert(quality_flag, ind+1, np.full(range_list.shape, -1), axis=0)
        v_term = np.insert(v_term, ind+1, np.full(range_list.shape, -99.), axis=0)
        v_air = np.insert(v_air, ind+1, np.full(range_list.shape, -99.), axis=0)


    quality_flag = np.ma.masked_less(quality_flag, 0., copy=True)
    v_term = np.ma.masked_less(v_term, -90., copy=True)
    v_air = np.ma.masked_less(v_air, -90., copy=True)


    v_term = np.ma.masked_where(quality_flag > 3.0, v_term)
    v_air = np.ma.masked_where(quality_flag > 3.0, v_air)
    #quality_flag = np.ma.masked_where(quality_flag >= 2.0, quality_flag)
    np.set_printoptions(threshold='nan')


    # In[15]:


    #print(f.variables)
    print(f.variables.keys())

    #print(f.variables['v'][:])

    print(f.variables['v_term'].units)

    print(f.variables['quality_flag'].comment)

    print('creation time', f.creation_time)
    #print('settings ', f.settings)


    # In[16]:


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(v_term[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
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

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_cr_v_term.png"
    fig.savefig(savename, dpi=250)
    plt.close()


    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list[rect.t_bg:rect.t_ed]),
                        range_list[rect.h_bg:rect.h_ed],
                        np.transpose(v_air[rect.t_bg:rect.t_ed, rect.h_bg:rect.h_ed]),
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

    savename = savepath + "/" + dt_list[0].strftime("%Y%m%d_%H%M")            + "_cr_v_air.png"
    fig.savefig(savename, dpi=250)
    plt.close()


error_list = []
for filename in filelist[40:]:
    try:
        run_filename("../output/" + filename)
    except:
        error_list.append(filename)
        
print(error_list)