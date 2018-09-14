#! /usr/bin/env python3
# coding=utf-8

import datetime
import re
import numpy as np
import netCDF4
from . import helpers


class cloudnet_attenuation():
    """
    load attenuation profile from cloudnet
    """

    def __init__(self, filename):
        self.filename = filename
        self.f = netCDF4.Dataset(filename, 'r')
        self.f.set_auto_maskandscale(False)
        #print(self.f)

        self.time_list = self.f.variables["time"][:]
        #hours since 2015-06-02 00:00:00 +0:00
        dt_zero = datetime.datetime.strptime(\
                re.search("[0-9]{8}", self.filename).group(), 
                '%Y%m%d')
        time_list = [helpers.dt_to_ts(dt_zero\
                +datetime.timedelta(hours=float(i)))\
                for i in self.time_list]
        self.time_list = np.array([int(t) for t in time_list])
        self.delta_t = np.mean(np.diff(self.time_list))

        self.height = self.f.variables["height"][:] - 104.
        self.delta_h = np.mean(np.diff(self.height))

        self.gas_atten = self.f.variables["radar_gas_atten"][:]
        self.liq_atten = self.f.variables["radar_liquid_atten"][:]
        #print(self.f.variables["radar_gas_atten"])

    def get_pixel(self, ts, height):
        """ 
        
        """
        ntime = helpers.argnearest(self.time_list, ts)
        #print('selected timestamp ', ts, ' self.ntime ', ntime)
        nheight = helpers.argnearest(self.height, height)  
        #print('selected height ', height, ' self.nheight ', nheight)
        #print(self.height.shape, self.gas_atten.shape)
        interp_gas_atten = np.interp(height, self.height, self.gas_atten[ntime])
        print(helpers.z2lin(interp_gas_atten))
        return interp_gas_atten


if __name__ == '__main__':
    atten = cloudnet_attenuation('../../colrawi/cloudnet/20150617_lindenberg_categorize.nc')
    print('advect vel ', atten.get_pixel(helpers.dt_to_ts(datetime.datetime(2015,6,17,21)), 5000.))