#! /usr/bin/env python3
# coding=utf-8
"""
Author: radenz@tropos.de

collection of tiny helper functions

"""

import datetime
import numpy as np
from numba import jit

def list_of_elem(elem, length):
    return [elem for i in range(length)]

def epoch_to_timestamp(time_raw):
    """
    converts rata die (days since year 0) to unix timestamp
    """
    offset = 719529  # offset between 1970-1-1 und 0000-1-1
    time = (time_raw - offset) * 86400
    return time

def dt_to_ts(dt):
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def ts_to_dt(ts):
    return datetime.datetime.utcfromtimestamp(ts)


def lin2z(array):
    return 10*np.log10(array)


def z2lin(array):
    return 10**(array/10.)


def fill_with(array, mask, fill):
    """fill an array where mask is true with fill value"""
    filled = array.copy()
    filled[mask] = fill
    return filled


#@profile
def detect_peak_simple(array, lthres=z2lin(-35.)):
    """
    detect noise separated peaks
    """
    ind = np.where(array > lthres)[0].tolist()
    jumps = [ind.index(x) for x, y in zip(ind, ind[1:]) if y - x != 1]
    runs = np.split(ind, [i + 1 for i in jumps])
    if runs[0].shape[0] > 0:
        peakindices = [(elem[0], elem[-1]) for elem in runs]
    else:
        peakindices = []
    return peakindices

def argnearest(array, value):
    i = np.where(array == min(array, key=lambda t: abs(value - t)))[0][0] 
    return i

def argnearest2(array, value):
    i = np.searchsorted(array, value)-1
    if not i == array.shape[0]-1 and np.abs(array[i]-value) > np.abs(array[i+1]-value):
        i = i+1
    return i

@jit
def estimate_noise(spec, mov_avg=1):
    """
    Noise estimate based on Hildebrand and Sekhon (1974)
    """
    i_noise = len(spec)
    spec_sort = np.sort(spec)
    for i in range(spec_sort.shape[0]):
        partial = spec_sort[:i+1]
        mean = partial.mean()
        var = partial.var()
        if var * mov_avg * 2 < mean**2.:
            i_noise = i
        else:
            # remaining part of spectrum has no noise characteristic
            break
    noise_part = spec_sort[:i_noise+1]

    return {'noise_mean': np.mean(noise_part), 'noise_sep': spec_sort[i_noise], 
            'noise_var': np.var(noise_part), 'no_noise_bins': i_noise}