#! /usr/bin/env python3
"""
Author: radenz@tropos.de

provides the horizontal wind velocity from different sources
"""


#from __future__ import print_function
import os, re
import datetime
import netCDF4   
#from Scientific.IO import NetCDF
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np


def calc_hvel(data):
    """calculate absolute horizontal velocity from u,v"""
    hvel = np.sqrt(np.power(data[:,4],2) + np.power(data[:,5], 2))
    return hvel

    
def dt_to_timestamp(dt):
    """calculate timestamp from dt
    only needed for python2.7, copied from spectra mole"""
    #timestamp_midnight = int((datetime.datetime(self.dt[1].year, self.dt[1].month, self.dt[1].day) - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)) #python3
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def ts_to_dt(ts):
    return datetime.datetime.utcfromtimestamp(ts)

    
def nearest(point, array, delta):
    """ searches nearest point in given array and returns (i, value[i]) 
    taken from BA programm an improved with index calculation, copied from spectra mole"""
    #i = bisect.bisect_left(array, point)
    i = int( (point - array[0])/delta )
    #print("search nearest ", i, point, " | ", array[max(0,i-5):i+6])
    #print("array shape ", array.shape)
    nearest = min( array[max(0,i-10):i+10], key=lambda t: abs(point - t) )
    #print('nearest ', nearest)
    #print('np.where ', np.where(array==nearest))
    i = np.where(array==nearest)[0][0]
    return (i, nearest)


class gdas():
    """
    loads advection speed profile from gdas data, interpolates it to a (given) grid
    and provides it to spectra mole;
    there it is used for the correction of beam width broadening
    """
    
    def __init__(self, dt, height_grid):
        """define suitable files, load them and interpolate
        dt: day of the measurement"""
        self.dt = dt
        self.height_grid = height_grid
        self.delta_h = np.mean(np.diff(self.height_grid))
        filepath="../radiosondes/"
        data00 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_00.txt", skiprows=1)
        data03 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_03.txt", skiprows=1)
        data06 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_06.txt", skiprows=1)
        data09 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_09.txt", skiprows=1)
        data12 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_12.txt", skiprows=1)
        data15 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_15.txt", skiprows=1)
        data18 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_18.txt", skiprows=1)
        data21 = np.loadtxt(filepath+"gdas_lindenberg_"+dt.strftime("%Y%m%d")+"_21.txt", skiprows=1)
        data24 = np.loadtxt(filepath+"gdas_lindenberg_"
                            +(dt+datetime.timedelta(days=1)).strftime("%Y%m%d")+"_00.txt", skiprows=1)
        #print(data24[:,2])
        #self.profiles_height: height interpolated horizontal winds
        self.profiles_height = np.zeros((9, height_grid.shape[0]))
        height_interp = interp.interp1d(data00[:,2], calc_hvel(data00),bounds_error=False, fill_value=0.0)
        self.profiles_height[0,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data03[:,2], calc_hvel(data03),bounds_error=False, fill_value=0.0)
        self.profiles_height[1,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data06[:,2], calc_hvel(data06),bounds_error=False, fill_value=0.0)
        self.profiles_height[2,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data09[:,2], calc_hvel(data09),bounds_error=False, fill_value=0.0)
        self.profiles_height[3,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data12[:,2], calc_hvel(data12),bounds_error=False, fill_value=0.0)
        self.profiles_height[4,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data15[:,2], calc_hvel(data15),bounds_error=False, fill_value=0.0)
        self.profiles_height[5,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data18[:,2], calc_hvel(data18),bounds_error=False, fill_value=0.0)
        self.profiles_height[6,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data21[:,2], calc_hvel(data21),bounds_error=False, fill_value=0.0)
        self.profiles_height[7,:] = height_interp(height_grid)
        height_interp = interp.interp1d(data24[:,2], calc_hvel(data24),bounds_error=False, fill_value=0.0)
        self.profiles_height[8,:] = height_interp(height_grid)
        
        time_list = np.arange(0,86401,10800)
        time_list += dt_to_timestamp(dt)
        self.delta_t = 1800   #30min used for grid generation and searching
        self.time_grid = np.arange(0,86401, self.delta_t)
        self.time_grid += dt_to_timestamp(dt)
        self.delta_h = np.mean(np.diff(self.height_grid))

        self.profiles = np.zeros((self.time_grid.shape[0], height_grid.shape[0]))
        for i in range(height_grid.shape[0]):
            time_interp = interp.interp1d(time_list, self.profiles_height[:,i], bounds_error=False, fill_value=0.0)
            self.profiles[:,i] = time_interp(self.time_grid)

        #plot for test reason
        #fig, ax = plt.subplots(1, figsize=(10, 8))
        #pcmesh = ax.pcolormesh(time_list,
        #              self.height_grid, np.transpose(self.profiles_height),
        #              cmap='gist_rainbow_r')
        #cbar = fig.colorbar(pcmesh)
        
        #fig, ax = plt.subplots(1, figsize=(10, 8))
        #pcmesh = ax.pcolormesh(self.time_grid, 
        #              self.height_grid, np.transpose(self.profiles),
        #              cmap='gist_rainbow_r')
        #cbar = fig.colorbar(pcmesh)

        #fig, ax = plt.subplots(1, figsize=(10, 8))
        #ax.plot(self.profiles_height[0,:], self.height_grid, "o", color="red")
        #ax.plot(self.profiles[0,:], self.height_grid, "-", color="red")
        
        #ax.plot(self.profiles_height[4,:], self.height_grid, "o", color="green")
        #ax.plot(self.profiles[24,:], self.height_grid, "-", color="green")
        
        #ax.plot(self.profiles_height[-1,:], self.height_grid, "o", color="blue")
        #ax.plot(self.profiles[-1,:], self.height_grid, "-", color="blue")

    def get_pixel(self, ts, height):
        """search specified timestamp/height and return advection speed [m/s]
        NEW height: search for height and interpolate
        timestamp: timestamp for which the pixel shall be searched 
        
        shear not working yet
        """
        
        nheight = nearest(height, self.height_grid, self.delta_h)      
        ntime = nearest(ts, self.time_grid, self.delta_t)      
        return self.profiles[ntime[0], nheight[0]], 0.0


class wp_advect():
    """
    loads advection velocity profile form the windprofiler off-zenith measurement
    the output should be consistent with the gdas class
    """

    def __init__(self, filename):
        """ """
        self.filename = filename
        self.f = netCDF4.Dataset(filename, 'r')
        #self.f = NetCDF.NetCDFFile(filename, 'r') 
        self.f.set_auto_maskandscale(False)

        self.time_list = self.f.variables["Timestamp"][:]
        self.delta_t = 60.*60.
        self.height = self.f.variables["WP_Height"][:]
        self.height = np.mean(self.height, 0)
        self.height = np.ma.masked_invalid(self.height)
        # quick hack to get 1-d height array
        self.delta_h = np.mean(np.diff(self.height))
        
        self.u_vel = self.f.variables["WP_U"][:]
        self.u_vel = np.ma.masked_greater_equal(self.u_vel, 1e20)
        self.v_vel = self.f.variables["WP_V"][:]
        self.v_vel = np.ma.masked_greater_equal(self.v_vel, 1e20)

        #print('shape u_vel', self.u_vel.shape)
        #print('available datetimes')
        #for i in range(self.time_list.shape[0]):
        #    print(i, datetime.datetime.utcfromtimestamp(self.time_list[i]))

    def get_pixel(self, ts, height):
        """search specified timestamp/height and return advection speed [m/s]
        2D interpolation included
        NEW height: search for height and interpolate
        timestamp: timestamp for which the pixel shall be searched 
        
        shear not working yet
        """
        
        #print('selected timestamp ', ts)
        ntime = nearest(ts, self.time_list, self.delta_t)      
        #print('self.ntime ', ntime)
        #print('selected height ', height)
        nheight = nearest(height, self.height.data, self.delta_h)
        #print('self.nheight ', nheight)

        if ts < ntime[1]:
            # found right boundary
            n_right = ntime[0]
            n_left = max(ntime[0] - 1, 0)
        elif ts > ntime[1]:
            # found left boundary
            n_right = min(ntime[0] + 1, self.time_list.shape[0]-1)
            n_left = ntime[0]

        if height < nheight[1]:
            # found upper boundary
            n_upper = nheight[0]
            n_lower = max(nheight[0] - 1, 0)
        elif height > nheight[1]:
            # found lower boundary
            n_upper = min(nheight[0] + 1, self.height.shape[0]-1)
            n_lower = nheight[0]

        u_left = self.u_vel[n_left, n_lower] + \
                (self.u_vel[n_left, n_upper] - self.u_vel[n_left, n_lower])/ \
                (self.height[n_upper] - self.height[n_lower])* \
                (height - self.height[n_lower])
        # print('u_left ', self.u_vel[n_left, n_lower], u_left,
        #        self.u_vel[n_left, n_upper])
        v_left = self.v_vel[n_left, n_lower] + \
                (self.v_vel[n_left, n_upper] - self.v_vel[n_left, n_lower])/ \
                (self.height[n_upper] - self.height[n_lower])* \
                (height - self.height[n_lower])
        # print('v_left ', self.v_vel[n_left, n_lower], v_left,
        #        self.v_vel[n_left, n_upper])
        u_right = self.u_vel[n_right, n_lower] + \
                 (self.u_vel[n_right, n_upper] - self.u_vel[n_right, n_lower])/ \
                 (self.height[n_upper] - self.height[n_lower])* \
                 (height - self.height[n_lower])
        # print('u_right ', self.u_vel[n_right, n_lower], u_right,
        #        self.u_vel[n_right, n_upper])
        v_right = self.v_vel[n_right, n_lower] + \
                 (self.v_vel[n_right, n_upper] - self.v_vel[n_right, n_lower])/ \
                 (self.height[n_upper] - self.height[n_lower])* \
                 (height - self.height[n_lower])
        # print('v_right ', self.v_vel[n_right, n_lower], v_right,
        #        self.v_vel[n_right, n_upper])
        # print('timestamps left, ts, right ',
        #        datetime.datetime.utcfromtimestamp(self.time_list[n_left]),
        #        datetime.datetime.utcfromtimestamp(ts),
        #        datetime.datetime.utcfromtimestamp(self.time_list[n_right]))

        self.height.mask = self.u_vel.mask[n_left]
        u_left = np.interp(height, self.height.compressed(), self.u_vel[n_left].compressed())
        self.height.mask = self.v_vel.mask[n_left]
        v_left = np.interp(height, self.height.compressed(), self.v_vel[n_left].compressed())
        self.height.mask = self.u_vel.mask[n_right]
        u_right = np.interp(height, self.height.compressed(), self.u_vel[n_right].compressed())
        self.height.mask = self.v_vel.mask[n_right]
        v_right = np.interp(height, self.height.compressed(), self.v_vel[n_right].compressed())


        u_ts = u_left + (u_right - u_left)/(self.time_list[n_right] -
                self.time_list[n_left])*(ts-self.time_list[n_left])
        v_ts = v_left + (v_right - v_left)/(self.time_list[n_right] -
                self.time_list[n_left])*(ts-self.time_list[n_left])

        advect_vel = np.sqrt(u_ts**2 + v_ts**2)
        return advect_vel, 0.0, (u_ts, v_ts)


class cloudnet_advect():
    """
    load the advection velocity from the cloudnet categorize file
    output should be consistent with the gdas class
    """
    
    def __init__(self, filename):
        """ 
        load the cloudnet categorize nc file
        """
        print("---- cloudnet advect -------------------------------------------")
        self.filename = filename
        self.f = netCDF4.Dataset(filename, 'r')
        self.f.set_auto_maskandscale(False)

        self.time_list = self.f.variables["time"][:]
        #hours since 2015-06-02 00:00:00 +0:00
        dt_zero = datetime.datetime.strptime(\
                re.search("[0-9]{8}", self.filename).group(), 
                '%Y%m%d')
        time_list = [dt_to_timestamp(dt_zero\
                +datetime.timedelta(hours=float(i)))\
                for i in self.time_list]
        self.time_list = np.array([int(t) for t in time_list])
        self.delta_t = np.mean(np.diff(self.time_list))
        # height above sea level (given in the nc file) has to be corrected 
        # for the height of lindenberg
        self.height = self.f.variables["model_height"][:] - 104.
        self.delta_h = np.mean(np.diff(self.height))
        self.u_vel = self.f.variables["uwind"][:]
        self.u_vel = np.ma.masked_less_equal(self.u_vel, -100.)
        self.v_vel = self.f.variables["vwind"][:]
        self.v_vel = np.ma.masked_less_equal(self.v_vel, -100.)
        #print('heights ', self.height)
        #print('times ', self.time_list)
        print("cloudnet advect time range ", self.time_list[:2].astype(int),
              self.time_list[-2:].astype(int),
              ts_to_dt(self.time_list[0].astype(int)),
              ts_to_dt(self.time_list[-1].astype(int)))
        print("delta t ", self.delta_t)

    def get_pixel(self, ts, height):
        """ 
        get the horizontal wind 
        and the wind shear (du/dz as the average of 3 range gates)

        adjusted for the more complex formula (Nastrom 1997)
        """
        #print('selected timestamp ', ts)
        ntime = nearest(ts, self.time_list, self.delta_t)      
        #print('self.ntime ', ntime)
        #print('selected height ', height)
        nheight = nearest(height, self.height, self.delta_h)      
        #print('self.nheight ', nheight)

        #running mean to prevent steps
        #u_vel = self.u_vel[ntime[0], nheight[0]]
        u_vel = np.interp(height, self.height, self.u_vel[ntime[0]])
        #v_vel = self.v_vel[ntime[0], nheight[0]]
        v_vel = np.interp(height, self.height, self.v_vel[ntime[0]])
        advect_vel = np.sqrt(u_vel**2+v_vel**2)
        shear_u = 0.5*((self.u_vel[ntime[0], nheight[0]+1] - self.u_vel[ntime[0], nheight[0]])/(self.height[nheight[0]+1]-self.height[nheight[0]]) \
                     + (self.u_vel[ntime[0], nheight[0]] - self.u_vel[ntime[0], nheight[0]-1])/(self.height[nheight[0]]-self.height[nheight[0]-1])) 
        shear_v = 0.5*((self.v_vel[ntime[0], nheight[0]+1] - self.v_vel[ntime[0], nheight[0]])/(self.height[nheight[0]+1]-self.height[nheight[0]]) \
                     + (self.v_vel[ntime[0], nheight[0]] - self.v_vel[ntime[0], nheight[0]-1])/(self.height[nheight[0]]-self.height[nheight[0]-1])) 
        return advect_vel, shear_u+shear_v, (u_vel, v_vel)

if __name__ == '__main__':
    #profile = gdas(datetime.datetime(2013, 9, 25), np.arange(300,7000,100))
    #print(profile.get_pixel(1380090931, 7))
    
    print("--- wind profiler -----------------------------------------")
    #profile = wp_advect("../colrawi2/wpl_20150602_002408.nc")
    #print('advect vel ', profile.get_pixel(1433238641, 700.))
    #print('advect vel ', profile.get_pixel(1433239841, 700.))
    print("--- cloudnet  -----------------------------------------")
    profile = cloudnet_advect("/home/radenz/colrawi2/cloudnet/20150602_lindenberg_categorize.nc")
    print('advect vel ', profile.get_pixel(1433238641, 700.))
    print('advect vel ', profile.get_pixel(1433239841, 700.))
    print('advect vel ', profile.get_pixel(1433239841, 4300.))
    print('advect vel ', profile.get_pixel(1433239841, 4500.))
    print('advect vel ', profile.get_pixel(1433239841, 4700.))

