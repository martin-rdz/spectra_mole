#! /usr/bin/env python3
# coding=utf-8

from collections import namedtuple

import logging
import itertools
import matplotlib
#matplotlib.use('Agg')
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage
from scipy.optimize import curve_fit

#from numba import jit
from . import helpers as h
from . import recPeakFinder
from . import vis
from . import advection
from . import writer
#import viridis  # fancy new colormap
#import VIS_Colormaps  # vertical velocity color map

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(filename='../test.log', mode='w')
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s', datefmt='%H:%M:%S')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# make a named tupel that contains the peak moments
mom = namedtuple('mom', ['Z', 'v', 'width', 'ileft', 'iright', 'snr'])
mom2str = lambda l: ' '.join(['Z:{0:.2f} v:{1:.2f} w:{2:.2f} snr:{5:.2f} | '.format(*mom) for mom in l])

flatten = lambda l: sum(map(flatten, l), []) if isinstance(l,list) else [l]
peaks2vel = lambda vel, peaks: [(vel[peak[0]], vel[peak[1]]) for peak in peaks]
vel2str = lambda vel: ['({:4.2f} {:4.2f})'.format(*pair) for pair in vel]

def peaks2snr(noise, z, peaks):
    """calculate the snr for noise, z and peaks"""
    if not peaks:
        raise ValueError('empty peak list')
    elif type(peaks) is tuple:
        return h.lin2z(np.sum(z[peaks[0]:peaks[1] + 1]/noise))
    else:
        return [h.lin2z(np.sum(z[peak[0]:peak[1] + 1]/noise)) for peak in peaks]

def gauss_func(x, m, sd):
    """
    calculate the gaussian function on a given grid

    x: grid
    m: mean
    sd: standard deviation
    """
    a = 1. / (sd * np.sqrt(2. * np.pi))
    return a * np.exp(-(x - m) ** 2 / (2. * sd ** 2))

def check_consistency(lst):
    """
    :lst: list of spectra dictionaries
    """ 
    #timestamps = ['{} {:.1f}'.format(elem['system'], elem['ts']) for elem in lst]
    timestamps = ['{} {}'.format(elem['system'], h.ts_to_dt(elem['ts']).strftime('%m-%d %H:%M:%S')) for elem in lst]
    heights = ['{} {:.1f}'.format(elem['system'], elem['range']) for elem in lst]
    logger.info('---- consistency check ---------------------------------------------------')
    logger.info('timestamps %s', timestamps)
    logger.info('heights    %s', heights)
    for pair in itertools.combinations(lst, 2):
        timediff = pair[0]['ts'] - pair[1]['ts']
        heightdiff = pair[0]['range'] - pair[1]['range']
        # 5.5sec is not enough as the cloud radar sometimes jumps more than 10s
        assert abs(timediff) < 7, 'times differ too much: {} - {}'.format(pair[0]['system'],
                                                                           pair[1]['system'])
        assert abs(heightdiff) < 40., 'heights differ too much: {} - {}'.format(pair[0]['system'],
                                                                                pair[1]['system'])

#@profile
def broaden_spectrum(adv_tupel, spectrum, cut_thres=None):
    """ """
    adv_vel, shear, _ = adv_tupel

    half_bw = 0.0262
    sigma_sq = 0.5 * (half_bw * adv_vel) ** 2  # \
    # + 3. / 24. * (half_bw * shear * self.wipro.nheight[1]) ** 2 \
    # + 1. / 36. * (half_bw * shear * self.wipro.delta_h) ** 2

    logger.debug('delta v %s', np.mean(np.diff(spectrum['vel'])))
    sigma_b = np.sqrt(sigma_sq) / np.mean(np.diff(spectrum['vel']))
    logger.info('sigma_b %s', sigma_b)



    specZ_filtered = spectrum['specZ'].copy()
    specSNRco_filtered = spectrum['specSNRco'].copy()
    plankton_mask = np.logical_and(
        h.lin2z(spectrum['specLDRmasked']) > -13, ~spectrum['specLDRmasked_mask'])
    if spectrum['range'] < 2100: 
        plankton_mask = np.logical_and( 
            h.lin2z(spectrum['specLDRmasked']) > -13, ~spectrum['specLDRmasked_mask']) 
    else: 
        plankton_mask = np.full(spectrum['specZ'].shape, False).astype(bool)

    specZ_filtered[plankton_mask] = np.nanmin(spectrum['specZ'])/3.
    specSNRco_filtered[plankton_mask] = np.nanmin(spectrum['specSNRco'])/3.
    specZbroad = scipy.ndimage.filters.gaussian_filter1d(specZ_filtered, sigma_b)
    specSNRbroad = scipy.ndimage.filters.gaussian_filter1d(specSNRco_filtered, sigma_b)

    spectrum_b = {'ts': spectrum['ts'], 'range': spectrum['range'], 'vel': spectrum['vel'],
                  'noise_thres': spectrum['noise_thres'], 'system': spectrum['system']+'_broad'}
    spectrum_b['specZ'] = specZbroad
    if cut_thres is None:
        spectrum_b['specZ_mask'] = specZbroad < spectrum['noise_thres']
    else: 
        spectrum_b['noise_thres'] = cut_thres
        spectrum_b['specZ_mask'] = specZbroad < cut_thres
        spectrum_b['specZ'][spectrum_b['specZ_mask']] = cut_thres/3.
    spectrum_b['specSNRco'] = specSNRbroad
    spectrum_b['sigma_b'] = sigma_b
    return spectrum_b


def check_particle_influence(spectrum, noise_thres):
    """
    check if there is signal in the spectrum above the threshold
    :return: bflag <dict>
    """
    if np.any(spectrum['specZ'] > noise_thres/3.):
        bflag = {"particle_influence": 1}
    else:
        bflag = {"particle_influence": 0}
    return bflag

#@profile
def calc_moments(spectrum):
    """
    replaces the calc_pnm_classic function in original mole

    :return : [mom <namedtuple>]
    """

    peak_list = h.detect_peak_simple(spectrum['specZ'], lthres=spectrum['noise_thres'])
    moments = []
    for peak in peak_list:
        refl = h.lin2z(
            np.sum(spectrum['specZ'][peak[0]:peak[1] + 1]))
        snr = h.lin2z(
            np.sum(spectrum['specZ'][peak[0]:peak[1] + 1]/spectrum['noise_thres']/3.))
        vel = np.average(spectrum['vel'][peak[0]:peak[1] + 1],
                         weights=spectrum['specZ'][peak[0]:peak[1] + 1])
        std = np.sqrt(np.average(
            (spectrum['vel'][peak[0]:peak[1] + 1] - vel) ** 2, 
            weights=spectrum['specZ'][peak[0]:peak[1] + 1]
            ))
        moments.append(mom(refl, vel, std, peak[0], peak[1], snr))

    # sort the moments list in descending order
    moments.sort(key=lambda x: x.Z, reverse=True)
    if not moments:
        # no peaks were found
        moments.append(mom(-200, -99., -99., -1, -1, -200))
    return moments


def filter_moments(moments, v, sigma, cr_iright):
    """remove all moments that are left of v+sigma and width < 0.08"""

    new_moments = []
    bflag = {'removed_moments': 0}
    if not np.isnan(v) and not np.isnan(sigma):
        logger.debug('at filter moments %s %s %s', mom2str(moments), v, sigma)
        
        for moment in moments:
            if moment[1] > v + sigma and moment[2] > 0.08 and moment.iright > cr_iright:
                new_moments.append(moment)

        if not new_moments:
            # no peaks were found
            new_moments.append(mom(-200, -99., -99., -1, -1, -200))
        new_moments.sort(key=lambda x: x.Z, reverse=True)
        bflag['removed_moments'] = 1

    return new_moments, bflag

#@profile
def check_rwp_calibration(spectrum_rwp, spectrum_broad):
    """
    check the rwp calibration by comparing with a broadened cloud radar spectrum

    :return: bflag, -cal_corr
    """
    # TODO take care with the sign
    bflag = {}

    deltaZ = h.lin2z(spectrum_rwp['specZ']) - h.lin2z(spectrum_broad['specZ'])
    deltaZ_mask = np.logical_or(spectrum_rwp['specZ_mask'], spectrum_broad['specZ_mask'])
    deltaZ = h.fill_with(deltaZ, deltaZ_mask, np.nan)

    # find the bounds of the particle peak
    ind_particle_peak = np.argwhere(spectrum_broad['specZ'] > spectrum_rwp['noise_thres']).ravel()
    if ind_particle_peak.shape[0] > 2:
        bounds_particle_peak = (ind_particle_peak[0]-1, ind_particle_peak[-1])
    else:
        bounds_particle_peak = (0, 0)

    logger.debug('bounds particle peak %s', bounds_particle_peak)
    logger.debug('shape of valid deltaZ %s', deltaZ[~deltaZ_mask].shape)
    # print('deltaZ ', deltaZ[bounds_particle_peak[0]:bounds_particle_peak[1]+1])
    # print('deltaZ_mask', deltaZ_mask[bounds_particle_peak[0]:bounds_particle_peak[1]+1])
    # any valid measurements in this range?
    if not np.all(deltaZ_mask[bounds_particle_peak[0]:bounds_particle_peak[1]]):
        # check the calibration at the minimum of deltaSNR
        mdeltaZ = np.nanargmin(deltaZ[bounds_particle_peak[0]:bounds_particle_peak[1]+1]) + bounds_particle_peak[0]
        # print('mdeltaZ ', mdeltaZ, deltaZ[mdeltaZ-2:mdeltaZ+3])
        # check the calibration at the maximum of the cloud radar peak
        maxmiraZ = np.nanargmax(spectrum_broad['specZ'][bounds_particle_peak[0]:bounds_particle_peak[1]+1]) + bounds_particle_peak[0]
        # print('maxmiraZ ', maxmiraZ, deltaZ[maxmiraZ-2:maxmiraZ+3])
    else:
        mdeltaZ = 0
        maxmiraZ = 0

    if mdeltaZ == 0 and maxmiraZ == 0:
        logger.info('check calibration: no particle signal')
        bflag['mod_calibration'] = 0
        bflag['unsecure_calibration'] = 0
        cal_corr = 1
    elif np.all(np.abs(deltaZ[mdeltaZ-1:mdeltaZ+2]) < 2) \
        and np.all(np.abs(deltaZ[maxmiraZ - 1:maxmiraZ + 2]) < 2):
        logger.info('check calibration: calibration ok')
        bflag['mod_calibration'] = 0
        bflag['unsecure_calibration'] = 0
        cal_corr = 1
    else:
        logger.info('check calibration: needs correction')
        # weight differently
        cal_corr = 2*np.mean(h.z2lin(deltaZ[mdeltaZ-1:mdeltaZ+2]))
        cal_corr += 3*np.mean(h.z2lin(deltaZ[maxmiraZ-1:maxmiraZ+2]))
        cal_corr = h.lin2z(cal_corr/5.)

        deltaZ_part_filled = h.fill_with(
            deltaZ, deltaZ_mask, 0.0)[bounds_particle_peak[0]:bounds_particle_peak[1]+1]
        stat_full_deltaZ = h.lin2z(np.std(h.z2lin(deltaZ_part_filled)))
        if cal_corr < 20 and stat_full_deltaZ < 24:
            logger.debug('correct calibration by: %s statistics %s', cal_corr, stat_full_deltaZ)
            bflag['mod_calibration'] = 1
            bflag['unsecure_calibration'] = 0
        else:
            logger.debug('calibration too much off: %s statistics %s', cal_corr, stat_full_deltaZ)
            cal_corr = np.nan
            bflag['mod_calibration'] = 0
            bflag['unsecure_calibration'] = 1

    return bflag, -cal_corr


def check_noise_level(spectrum_rwp, extern=None):
    """
    check if the Hildebrand-Sekhon noise is more than 4dB above the noise 
    provided by the standard signal processing

    if yes replace the noise information in the spectrum_rwp

    :returns : spectrum_rwp, bflag
    """
    
    bflag = {}
    if h.lin2z(spectrum_rwp['noise_lvl_hs']) > h.lin2z(spectrum_rwp['noise_lvl']) + 4:
        logger.debug('noise estimation wrong %s %s', h.lin2z(spectrum_rwp['noise_lvl_hs']), h.lin2z(spectrum_rwp['noise_lvl']))
        bflag['hs_higher_noise'] = 1
        spectrum_rwp['noise_lvl'] = spectrum_rwp['noise_lvl_hs']
        spectrum_rwp['noise_thres'] = spectrum_rwp['noise_lvl']*2
    else:
        bflag['hs_higher_noise'] = 0

    if extern != None and  h.lin2z(extern) > h.lin2z(spectrum_rwp['noise_lvl']) + 4:
        logger.debug('external noise higher %s %s', h.lin2z(extern), h.lin2z(spectrum_rwp['noise_lvl']))
        bflag['cr_higher_noise'] = 1
        spectrum_rwp['noise_lvl'] = extern/3.
        spectrum_rwp['noise_thres'] = extern
    else:
        bflag['cr_higher_noise'] = 0
        
    return spectrum_rwp, bflag

#@profile
def estimate_calibration(spectrum_rwp, spectrum_broad):
    """
    estimate the calibration constant in the interval -3.00 -0.94

    :return : cal_const
    """
    ivl, ivr = 91, 118
    assert abs(spectrum_rwp['vel'][ivl] - -3.0) < 0.1
    assert abs(spectrum_rwp['vel'][ivr] - -0.7) < 0.1

    if np.sum(~spectrum_broad['specZ_mask'][ivl:ivr]) > 0:
        spec_calibration = spectrum_broad['specZ'][ivl:ivr] / \
                ((spectrum_rwp['range'] ** 2) * spectrum_rwp['specSNRco'][ivl:ivr])
        cal_const = spec_calibration.mean()
    else:
        cal_const = np.nan

    return cal_const


def modify_calibration(spectrum, correction_factor):
    """ modify the calibration of a given spectrum (only 'specZ') by a correction factor"""
    assert not np.isnan(correction_factor), 'correction factor is not valid'
    spectrum_mod = spectrum.copy()
    logger.debug('cal_const %s %s', spectrum['cal_const'], correction_factor)
    cal_const = h.z2lin(h.lin2z(spectrum['cal_const']) + correction_factor)
    logger.debug('cal_const new %s', cal_const)

    logger.debug('old lvl %s thres %s', h.lin2z(spectrum['noise_lvl']), h.lin2z(spectrum['noise_thres']))
    noise_thres = h.z2lin(correction_factor) * spectrum['noise_thres']
    noise_lvl = h.z2lin(correction_factor) * spectrum['noise_lvl']
    noise_lvl_hs = h.z2lin(correction_factor) * spectrum['noise_lvl_hs']
    logger.debug('new lvl %s thres %s', h.lin2z(noise_lvl), h.lin2z(noise_thres))
    zspectrum = cal_const * (spectrum['range'] ** 2) * spectrum['specSNRco']

    spectrum_mod['specZ_mask'] = zspectrum < noise_thres
    spectrum_mod['specZ'] = zspectrum
    spectrum_mod['noise_thres'] = noise_thres
    spectrum_mod['noise_lvl'] = noise_lvl
    spectrum_mod['noise_lvl_hs'] = noise_lvl_hs
    spectrum_mod['cal_const'] = cal_const

    return spectrum_mod

#@profile
def correct_with_weighting(spec_rwp, spec_broad):
    """
    calculate the corrected spectrum with the weighting functions

    steps
    #.
    """
    wipro_min = np.nanmin(spec_rwp['specZ'])
    wipro_zspec_temp = h.fill_with(spec_rwp['specZ'], spec_rwp['specZ_mask'], wipro_min)
    blured = spec_broad['specZ']
    deltaSNR = h.lin2z(spec_rwp['specZ']/wipro_min) - h.lin2z(blured/wipro_min)
    # find the bounds of the particle peak
    ind_particle_peak = np.argwhere(blured > wipro_min)
    if ind_particle_peak.shape[0] > 2:
        bounds_particle_peak = (ind_particle_peak[0]-1, ind_particle_peak[-1])
    else:
        bounds_particle_peak = (0, 0)

    rel = blured / wipro_zspec_temp
    snr_cr = blured / wipro_min
    # allow only values < 1
    # rel[rel > 1] = rel[rel > 1]**(-1)
    rel1m = 1 - rel
    weight_bragg = rel1m
    # where cloud radar reaches noise level, set equal 1
    weight_bragg[blured == wipro_min] = 1
    # running mean on the weight
    convol_kernel = np.array([0.3, 0.5, 1., 0.5, 0.3])
    convol_kernel /= np.sum(convol_kernel)
    weight_bragg = np.convolve(weight_bragg, convol_kernel, mode='same')

    # supress every signal where weight of bragg is less than 0.4
    #weight_bragg[weight_bragg < 0.7] = snr_cr[weight_bragg < 0.7] ** (-1)
    # 0.3 prooved to be too less....
    weight_bragg[weight_bragg < 0.6] = snr_cr[weight_bragg < 0.6] ** (-1)
    # define something like a transition region

    difference = wipro_zspec_temp * weight_bragg

    spec_corr = {'range': spec_rwp['range'], 'ts': spec_rwp['ts'], 'system': 'rwp_corr',
        'specZ': difference,
        'specZ_mask': difference < spec_rwp['noise_thres'],
        'vel': spec_rwp['vel'], 'delta_v': spec_rwp['delta_v'], 'cal_const': spec_rwp['cal_const'],
        'noise_lvl': spec_rwp['noise_lvl'], 'noise_thres': spec_rwp['noise_thres'],
        }
    return spec_corr

#@profile
def correct_with_fuzzy(spec_rwp, spec_broad, savepath=False):
    """
    
    """
    wipro_min = np.nanmin(spec_rwp['specZ'])
    wipro_zspec_temp = h.fill_with(spec_rwp['specZ'], spec_rwp['specZ_mask'], wipro_min)

    wipro_zspec_temp[wipro_zspec_temp < spec_rwp['noise_thres']*(2.75/2.9)] = wipro_min
    blured = spec_broad['specZ']

    # find the bounds of the particle peak
    ind_particle_peak = np.argwhere(blured > wipro_min)
    if ind_particle_peak.shape[0] > 2:
        bounds_particle_peak = (ind_particle_peak[0]-1, ind_particle_peak[-1])
    else:
        bounds_particle_peak = (0, 0)

    #print('calc_fuzzy_spec: deltaSNR ', deltaSNR[bounds_particle_peak[0]:bounds_particle_peak[1]+1])
    # range to scale the spectral reflectivity
    values_range = (h.lin2z(wipro_min), h.lin2z(np.max(blured))+5)

    fuzzy_cloudradar = (h.lin2z(blured)-values_range[0])/(values_range[1] - values_range[0])
    fuzzy_rwp = (h.lin2z(wipro_zspec_temp) - values_range[0]) / (values_range[1] - values_range[0])
    fuzzy_rwp[:30] = 0.
    fuzzy_rwp[200:] = 0.
    # calculate a fuzzy mask
    # correlation of high return of both radars
    # vs anticorrelation high return of rwp and low return of cloud radar
    fuzzy_bragg = (1-fuzzy_cloudradar)*fuzzy_rwp
    fuzzy_bragg[fuzzy_bragg < fuzzy_cloudradar*fuzzy_rwp] = 0.
    # dump everything left of the intercepting point
    fuzzy_bragg[:np.argmax(blured)] = 0.
    fuzzy_bragg[fuzzy_bragg < 0.05] = 0.
    #left edge of the co correlation
    leftedgecocorr = np.argwhere(fuzzy_bragg > 0.01)[0, 0] if np.any(fuzzy_bragg > 0.01) else np.nan

    # and the fuzzy membership for larger than the cloud radar
    fuzzy_bragg2 = (1-fuzzy_cloudradar)*fuzzy_rwp
    fuzzy_bragg2[fuzzy_bragg2 < fuzzy_cloudradar] = 0.
    fuzzy_bragg2[:np.argmax(blured)] = 0.
    fuzzy_bragg2[fuzzy_bragg2 < 0.05] = 0.
    leftedgecr = np.argwhere(fuzzy_bragg2 > 0.01)[0, 0] if np.any(fuzzy_bragg2 > 0.01) else 0

    if np.sum(fuzzy_bragg > fuzzy_cloudradar) < 3:
        logger.debug("set fuzzy bragg to 0, %s", np.sum(fuzzy_bragg > fuzzy_cloudradar))
        fuzzy_bragg[:] = 0.

    # # test for now
    #peaks_fuzzy_bragg = h.detect_peak_simple(fuzzy_bragg, lthres=0.05)
    #print('peaks_fuzzy_bragg ', peaks_fuzzy_bragg, leftedgecr)
    # filter out everything solely under cloud radar
    #peaks_fuzzy_bragg = filter(lambda x: x[1] > leftedgecr+2, peaks_fuzzy_bragg)
    peaks_fuzzy_bragg = recPeakFinder.detect_peak_recursive(fuzzy_bragg, 0.05, lambda thres: thres+0.1)[0]
    difference = wipro_zspec_temp
    difference[fuzzy_bragg < 0.1] = wipro_min

    peaks_wipro = recPeakFinder.detect_peak_recursive(spec_rwp['specZ'], spec_rwp['noise_thres'], lambda thres: thres*1.4)[0]

    def wipro_snr_wrapper(noise, zspectrum):
        return lambda peaks: peaks2snr(noise, zspectrum, peaks)
    wipro_snr = wipro_snr_wrapper(spec_rwp['noise_thres'], spec_rwp['specZ'])

    # if there is a peak within the fuzzy, then restrict to that
    if peaks_fuzzy_bragg and peaks_wipro: # ie list not empty
        logger.debug('=====    fuzzy peak decision =================================================')
        logger.debug('peaks_fuzz_bragg recursive %s %s %s', peaks_fuzzy_bragg, leftedgecocorr, leftedgecr)
        logger.debug('peaks fuzzy snr %s', wipro_snr(peaks_fuzzy_bragg))
        logger.debug('complete snr fuzzy %s %s',
              wipro_snr((peaks_fuzzy_bragg[0][0], peaks_fuzzy_bragg[-1][1])),
              wipro_snr((leftedgecr, peaks_fuzzy_bragg[-1][1])))
        logger.debug('peaks wipro from recursive %s', peaks_wipro)
        logger.debug('velocities wipro %s' + ", ".join(vel2str(peaks2vel(spec_rwp['vel'], peaks_wipro))))
        logger.debug('peaks wipro snr %s', wipro_snr(peaks_wipro))
        peaks_filtered = list(filter(
            lambda peak: peak[1] >= leftedgecr+2 and peak[0] >= leftedgecr-1 and wipro_snr(peak) > 10,
            peaks_wipro))

        #print("peaks completely right of cr ", peaks_filtered)
        if peaks_filtered and wipro_snr((peaks_filtered[0][0], peaks_filtered[-1][1])) \
                < 0.9*wipro_snr((peaks_fuzzy_bragg[0][0], peaks_fuzzy_bragg[-1][1])):
            # somehow lossed a very big peak
            peaks_filtered = list(filter(
                lambda peak: peak[1] >= leftedgecr+3 and wipro_snr(peak) > 10,
                peaks_wipro))
            if peaks_filtered:
                peaks_filtered[0] = (leftedgecr, peaks_filtered[0][1])
            else:
                # hack for the 2015-08-27 case at 0514
                peaks_filtered = [(leftedgecr, peaks_wipro[-1][1])]
            logger.debug("after snr cond %s", peaks_filtered)


        def filter_within(bounds):
            return lambda peak: (bounds[0]-7 <= peak[0] <= bounds[1]
                                 and bounds[0] <= peak[1] <= bounds[1]
                                 and wipro_snr(peak) > 10)

        peaks_wipro_within = [wp for wp, fp in itertools.product(peaks_wipro, peaks_fuzzy_bragg) if
                              filter_within(fp)(wp)]
        #print('peaks_wipro_within', peaks_wipro_within)
        peaks_wipro_within_strict =  list(filter(
            lambda x: x[1] >= leftedgecr and x[0] >= leftedgecr-1, peaks_wipro_within))
        #print('peaks_wipro_within_strict', peaks_wipro_within_strict)

        if peaks_filtered: # ie list not empty
            logger.debug("use peaks filtered %s", peaks_filtered)
            difference[:peaks_filtered[0][0]] = wipro_min
            difference[peaks_filtered[-1][1]:] = wipro_min
        elif peaks_wipro_within_strict:
            logger.debug("use peaks_wipro_within_strict %s", peaks_wipro_within_strict)
            difference[:peaks_wipro_within_strict[0][0]] = wipro_min
            difference[peaks_wipro_within_strict[-1][1]:] = wipro_min
        elif peaks_wipro_within:
            logger.debug("use peaks_wipro_within %s", peaks_wipro_within)
            difference[:peaks_wipro_within[0][0]] = wipro_min
            difference[peaks_wipro_within[-1][1]:] = wipro_min
        #elif wipro_snr((peaks_fuzzy_bragg[0][0], peaks_fuzzy_bragg[-1][1])) > 15.:
        elif wipro_snr((leftedgecr, peaks_fuzzy_bragg[-1][1])) > 15.:
            logger.debug("use snr between leftedge and fuzzy bragg")
            difference[:peaks_fuzzy_bragg[0][0]] = wipro_min
            difference[peaks_fuzzy_bragg[-1][1]:] = wipro_min
        else:
            difference[:] = wipro_min


    if savepath is not False:
        fig, ax = plt.subplots(2, figsize=(8, 5))
        ax[0].step(clara_pix.data['vel_list'], fuzzy_cloudradar, linewidth=1.5, color='red', label='raw cloud')
        ax[0].step(clara_pix.data['vel_list'], fuzzy_rwp, linewidth=1.5, color='green', label='raw RWP')
        ax[0].set_title("SNR spec at " + ts_to_dt(spec_rwp['ts']).strftime("%Y-%m-%d %H%M") +
                     " " + '{:4d}m'.format(int(spec_rwp['range'])), fontweight='semibold', fontsize=13)
        ax[0].set_xlim([-7, 3])
        #ax[0].set_xlim([-4.0, 1.5])


        ax[1].step(clara_pix.data['vel_list'], fuzzy_cloudradar*fuzzy_rwp, linewidth=1.5, color='orange', label='co')
        ax[1].step(clara_pix.data['vel_list'], (1-fuzzy_cloudradar)*fuzzy_rwp, linewidth=1.5, color='skyblue', label='cross')
        ax[1].step(clara_pix.data['vel_list'], fuzzy_cloudradar, linewidth=1.5, color='red')
        ax[1].step(clara_pix.data['vel_list'], fuzzy_bragg, linewidth=1.5, color='black', label='Bragg')
        ax[1].set_xlim([-7, 3])
        #ax[1].set_xlim([-4.0, 1.5])

        for a in ax:
            #a.set_ylim([0, 1])
            a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
            a.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
            a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
            a.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
            a.set_ylabel("Weight", fontweight='semibold')
            a.tick_params(axis='both', which='major', labelsize=14,
                           width=2, length=4)
            a.tick_params(axis='both', which='minor', width=1.4, length=3)
            a.legend(fontsize=11, numpoints=1, loc='best')

        ax[1].set_xlabel("Velocity [m s$\\mathregular{^{-1}}$]", fontweight='semibold')
        ax[0].set_title("fuzzy membership " + ts_to_dt(spec_rwp['ts']).strftime("%Y-%m-%d %H%M") +
                     " " + '{:4d}m'.format(int(spec_rwp['range'])), fontweight='semibold', fontsize=13)
        plt.tight_layout()
        savename = savepath + "/" + ts_to_dt(spec_rwp['ts']).strftime("%Y-%m-%d_%H%M") + "_" \
                   + '{0:04d}m'.format(int(spec_rwp['range'])) + "_fuzzy_membership_transform.png"
        fig.savefig(savename, dpi=250)
        savename = savepath + "/svg/" + ts_to_dt(spec_rwp['ts']).strftime("%Y-%m-%d_%H%M") + "_" \
                   + '{0:04d}m'.format(int(spec_rwp['range'])) + "_fuzzy_membership_transform.svg"
        #fig.savefig(savename)
        plt.close(fig)


    # data_dict = {'sel_height': wipro_pix.data['sel_height'], 'sel_timestamp': wipro_pix.data['sel_timestamp'],
    #              'ntime': wipro_pix.data['ntime'], 'nheight': wipro_pix.data['nheight'],
    #              'noise_lvl': wipro_pix.data['noise_lvl'], 'delta_v': wipro_pix.data['delta_v'],
    #              'noise_thres': wipro_pix.data['noise_thres'],
    #              'zspectrum': difference, 'fuzzy_membership': fuzzy_bragg,
    #              'vel_list': clara_pix.data['vel_list'], 'instr': 'corr'}

    spec_corr = {'range': spec_rwp['range'], 'ts': spec_rwp['ts'], 'system': 'rwp_corr',
        'specZ': difference,
        'fuzzy_membership': fuzzy_bragg,
        'specZ_mask': difference < spec_rwp['noise_thres'],
        'vel': spec_rwp['vel'], 'delta_v': spec_rwp['delta_v'], 'cal_const': spec_rwp['cal_const'],
        'noise_lvl': spec_rwp['noise_lvl'], 'noise_thres': spec_rwp['noise_thres'],
        }

    return spec_corr



def gauss_func(x, m, sd):
    """
    calculate the gaussian function on a given grid
    x: grid
    m: mean
    sd: standard deviation
    """
    a = 1. / (sd * np.sqrt(2. * np.pi))
    return a * np.exp(-(x - m) ** 2 / (2. * sd ** 2))


def fit_res_curve(x, m1, sd1, s1):
    """
    function that calculates the fit residuals
    http://stackoverflow.com/questions/10143905/python-two-curve-gaussian-fitting-with-non-linear-least-squares

    p: parameters [m1, sd1, s1] (mean, standard dev, strength)
    y: real function
    x: grid?
    """
    y_fit = s1 * gauss_func(x, m1, sd1)
    return y_fit

#@profile
def fit_peak(spec_corr, moments):

    """
    fit one peak to the fuzzy membership filtered spectrum

    current initial parameters: p = [mean, width, reflectivity]

    :param spec_corr: wind profiler data pixel
    :return: dataPixel
    """

    logger.debug('=====    fit peak =================================================')

    if moments[0].Z < -25. and not (moments[0].Z == -200 or moments == []):
        logger.debug('# moments fuzzy spec_corr (used for fit) %s', moments)
        width = min(max(moments[0].width, 0.09), 0.45)
        refl = min(max(moments[0].Z, -55), 10)
        p = [moments[0].v, width, h.z2lin(refl)*spec_corr['delta_v']]
        bounds = ([moments[0].v-0.3, max(0.07,width-0.1), h.z2lin(-55.)*spec_corr['delta_v']],
                  [moments[0].v+0.3, 0.45, h.z2lin(10.)*spec_corr['delta_v']])
    else:
        p = [0.5, 0.2, h.z2lin(-20.)*spec_corr['delta_v']]
        bounds = ([-10, 0.02, h.z2lin(-60.)*spec_corr['delta_v']], [+10, 0.45, h.z2lin(10.)*spec_corr['delta_v']])
    logger.debug('p  %s  bounds %s ', p, bounds)

    spec_to_fit = spec_corr['specZ'].copy()
    spec_to_fit[spec_corr['specZ'] < spec_corr['noise_thres']] = 1e-100
    #spec_to_fit -= wipro_pix.data['noise_thres']
    logger.debug('# refl at fit %s %s', h.lin2z(np.sum(spec_corr['specZ'])),
          h.lin2z(np.sum(spec_to_fit)))
    moments_fit = []
    try:
        param_lsq, cov_lsq = curve_fit(fit_res_curve, spec_corr['vel'], spec_to_fit,
                                       p0=p, bounds=bounds)
        err_lsq = np.sqrt(np.diag(cov_lsq))
        logger.info('param lsq (single peak) %s %s ', param_lsq, err_lsq)

        fitted_spectrum = param_lsq[2] * gauss_func(spec_corr['vel'], param_lsq[0], param_lsq[1])
        i = np.argwhere(fitted_spectrum > spec_corr['noise_thres']).ravel()
        if i.shape[0]>2:
            ileft, iright = i[0], i[-1]
        else:
            ileft, iright = -1, -1
        moments_fit.append(
            mom(h.lin2z(np.sum(fitted_spectrum - 1e-100)), param_lsq[0], param_lsq[1],
                ileft, iright, 
                h.lin2z(np.sum(fitted_spectrum - 1e-100)/spec_corr['noise_thres']/3.)))
        logger.debug('moments from fit %s', [h.lin2z(param_lsq[2]/spec_corr['delta_v']), param_lsq[0], param_lsq[1],
               h.lin2z(np.sum(np.sum(fitted_spectrum - 1e-100)))])

    except RuntimeError:
        logger.debug('# not enough rayleigh signal')
        moments_fit = [mom(-200, -99., -99., -1, -1, -200)]
        fitted_spectrum = spec_corr['specZ'].copy()
        fitted_spectrum[:] = 1e-20
        err_lsq = [-99.]

    if moments_fit[0].width <= 0.07 \
            or moments_fit[0].Z < -50 or np.isnan(moments_fit[0].Z):
        # fitted peak too small (testcase eg 2015-06-17_2130 8014)
        moments_fit = [mom(-200, -99., -99., -1, -1, -200)]
        # for the moment do not overwrite if it fails
        #fitted_spectrum = spec_corr['specZ'].copy()
        #fitted_spectrum[:] = 1e-20

    if np.abs(moments_fit[0].width-0.4) < 0.15 and np.abs(moments_fit[0].v-0.5) < 0.01:
        # fit somehow failed
        logger.info('moment fit replicated input')
        moments_fit = [mom(-200, -99., -99., -1, -1, -200)]

    # if np.abs(moments_fit[0].width-p[1]) < 0.15 and np.abs(moments_fit[0].v-p[1]) < 0.01:
    #     # fit somehow failed
    #     print('moment fit replicated input')
    #     moments_fit = [mom(-200, -99., -99., -1, -1, -200)]

    spec_corr = {'range': spec_corr['range'], 'ts': spec_corr['ts'], 'system': 'fit',
        'specZ': fitted_spectrum,
        'apriori_spec': spec_corr['specZ'],
        'specZ_mask': fitted_spectrum < spec_corr['noise_thres'],
        'err_fit': err_lsq,
        'moments': moments_fit,
        'vel': spec_corr['vel'], 'delta_v': spec_corr['delta_v'], 'cal_const': spec_corr['cal_const'],
        'noise_lvl': spec_corr['noise_lvl'], 'noise_thres': spec_corr['noise_thres'],
        }
    return spec_corr


def estimate_flag(bflag, corr_moments, cr_moments, cr_ldr, bounds_unfiltered_moments):
    """
    :bflag: dict for the binary flag

    """
    # do not overwrite bflag here! (this has to be done at top level)
    addbflag = {'low_snr': 0, 
                'plankton': 0,
                'melting_layer': 0}
    flag_doc = {0: 'not influenced',
                1: 'hydromet only',
                2: 'plankton',
                3: 'low snr',
                4: '',
                5: 'melting layer'}

    bins_above_noise = sum(
        [b[1]-b[0] for b in bounds_unfiltered_moments])
    bounds_unfiltered_moments.sort()
    bin_max_span = bounds_unfiltered_moments[-1][-1] - bounds_unfiltered_moments[0][0]

    if bflag["particle_influence"] == 1:
        flag = 1
        if corr_moments[0].snr < 8:
            addbflag['low_snr'] = 1
            flag = 3
        if cr_ldr > -13:
            if cr_moments[0].Z < -3:
                addbflag['plankton'] = 1
                flag = 2
            else:
                addbflag['melting_layer'] = 1
                flag = 5
        else:
            if (len(corr_moments) > 4 
                or bins_above_noise > 140 
                or bin_max_span > 180
                or bflag['hs_higher_noise'] == 1):
                addbflag['melting_layer'] = 1
                flag = 5
    else:
        flag = 0
    return addbflag, flag, flag_doc


def get_index(grid_mids, values):
    diff = np.diff(grid_mids)
    diff = np.concatenate((diff, diff[-1:]))
    edges = np.concatenate((grid_mids-diff/2, grid_mids[-1:]+diff[-1:]/2))
    print(edges)
    ind = np.digitize(np.array(values), edges)-1
    ind[ind > grid_mids.shape[0]-1] = grid_mids.shape[0]-1
    return ind


#@profile
def estimate_error(method, separation, contrast):
    """
    estimate the error for given separation and contrast form Monte Carlo simulation
    
    new version with the faster get_index method (using np.digitize)
    :return:
    """

    loaded = np.load('../data/results_2D_{}_statistics.npz'.format(method))
    #print(loaded.keys())
    index_sep = get_index(loaded['separation'], separation)
    index_con = get_index(loaded['contrast'], contrast)

    est_error_2D = loaded['mean_error'][index_sep, index_con]

    return est_error_2D



def calc_separation(bragg_mom, moments_broad):

    input_vals = np.array([bragg_mom.v, moments_broad.v, moments_broad.width, bragg_mom.width])
    if np.any(input_vals < - 90) or np.any(np.isnan(input_vals)):
        separation = -99.
    else:
        separation = (bragg_mom.v - moments_broad.v) / (moments_broad.width + bragg_mom.width)
    return separation


def calc_contrast(spec_corr, spec_broad, v):
    index_vel = np.argmin(np.abs(spec_corr['vel'] - v))
    assert np.all(spec_corr['vel'] == spec_broad['vel'])
    contrast = h.lin2z(spec_corr['specZ'][index_vel]) - h.lin2z(spec_broad['specZ'][index_vel])
    return contrast


def bflag2str(bflag):
    l = []
    #print("flag keys ", bflag.keys())
    #['plankton', 'small_Z_diff', 'melting_layer', 'too_many_peaks']
    keys = ['mod_calibration', 'unsecure_calibration', 'particle_influence',
            'removed_moments', 'melting_layer', 'plankton', 'low_snr', 'hs_higher_noise']

    if set(keys).issubset(set(bflag.keys())):
        for k in keys:
            l.append(str(bflag[k]))
    else:
        l = ['-1']

    return ''.join(l[::-1]), keys[::-1]


def bin2int(bin):
    #print('bin conversion', bin, int(bin, 2))
    return int(bin, 2)

#@profile
def correct_pixel(cr, rwp, advect, sel_ts, sel_range, visualize=False):
    """
    do the correction for a single pixel
    as done by the old version of mole
    """
    bflag = {}

    if type(sel_ts) is tuple and type(sel_range) is tuple:
        it, ts = sel_ts
        ir, rg = sel_range
    else:
        ts = sel_ts
        rg = sel_range

    logger.info('pixel at {} {}'.format(h.ts_to_dt(ts), rg))
    print('pixel at {} {}'.format(h.ts_to_dt(ts), rg))
    spec = cr.get_spectrum(ts, rg, range_average=True)
    spec_rwp = rwp.get_spectrum(sel_ts, sel_range,
                                interp_vel=spec['vel'])
    vis.plot_spectrum([spec, spec_rwp]) if visualize else None

    #vis.plot_spectrum(spec, spectrum_rwp=spec_rwp)
    assert np.all(spec['vel'] == spec_rwp['vel']), 'vel lists not identical'
    check_consistency([spec, spec_rwp])

    # load the advection speed
    adv_tupel = advect.get_pixel(ts, rg)

    spec_broad = broaden_spectrum(adv_tupel, spec, 
                            # capture the case where the cr noise thres is larger than the rwp
                            cut_thres=max(spec_rwp['noise_thres'], spec['noise_thres']))
    #vis.plot_spectrum(spec, spectrum_rwp=spec_rwp, spectrum_b=spec_broad)
    spec_broad['moments'] = calc_moments(spec_broad)

    # system parameter and calibration
    cal_const = estimate_calibration(spec_rwp, spec_broad)
    flg, cal_corr = check_rwp_calibration(spec_rwp, spec_broad)
    bflag.update(flg)
    
    if bflag['mod_calibration'] == 1:
        spec_rwp = modify_calibration(spec_rwp, cal_corr)
    spec_rwp['moments'] = calc_moments(spec_rwp)
    logger.debug('spec_rwp[moments] %s', mom2str(spec_rwp['moments']))

    bflag.update(check_particle_influence(spec_broad, spec_rwp['noise_thres']))

    spec_rwp, flg = check_noise_level(spec_rwp, extern=spec['noise_thres'])
    #spec_rwp, flg = check_noise_level(spec_rwp)
    bflag.update(flg)

    logger.debug('bflag %s', bflag)
    if bflag['particle_influence']:
        logger.info('particle influence true')
        # two ways of correcting
        # 1. williams
        rwp_weighting = correct_with_weighting(spec_rwp, spec_broad)
        rwp_weighting['moments'] = calc_moments(rwp_weighting)

        #vis.plot_spectrum(spec, spectrum_rwp=spec_rwp, spectrum_b=rwp_weighting)

        # 2. fuzzy/peakfitting
        rwp_fuzzy = correct_with_fuzzy(spec_rwp, spec_broad)
        rwp_fuzzy['moments'] = calc_moments(rwp_fuzzy)
        rwp_fit = fit_peak(rwp_fuzzy, rwp_fuzzy['moments'])

        # filter moments
        bounds_unfiltered_moments = [(mom.ileft, mom.iright) for mom in rwp_weighting['moments']]
        filtered_moments, flg = filter_moments(rwp_weighting['moments'], 
                                               spec_broad['moments'][0].v, spec_broad['moments'][0].width,
                                               spec_broad['moments'][0].iright)
        bflag.update(flg)
        rwp_weighting['moments'] = filtered_moments

        logger.debug('filtered moments %s', mom2str(filtered_moments))

        flg, flag, flag_doc = estimate_flag(bflag, filtered_moments,
                                            spec_broad['moments'], spec['est_ldr'],
                                            bounds_unfiltered_moments)
        bflag.update(flg)



        savepath = '../plots/{}/'.format(h.ts_to_dt(ts).strftime('%Y-%m-%d_%H%M%S'))
        vis.plot_spectrum(
            [spec, spec_broad, rwp_weighting, rwp_fuzzy, spec_rwp, rwp_fit], 
            further_text={'flag': flag, 'bflag': bflag2str(bflag)[0]},
            savepath=savepath) if visualize else None

        metrics = [calc_separation(filtered_moments[0], spec_broad['moments'][0]),
                   calc_contrast(rwp_weighting, spec_broad, filtered_moments[0].v)]
        logger.debug('metrics separation, contrast %s', metrics)
        
        ret = {'bragg_weighting': filtered_moments[0], 'bragg_fit': rwp_fit['moments'][0],
               'spec_rwp': spec_rwp, 'spec_broad': spec_broad,
               'flag': flag,'bflag': bflag, 'metrics': metrics, 'flag_doc': flag_doc,
               'cal_const': cal_const, 'cal_corr': cal_corr}
    else:
        logger.info('no particle influence')
        metrics = [-99, -200]
        ret = {'bragg_weighting': spec_rwp['moments'][0], 'bragg_fit': spec_rwp['moments'][0],
               'spec_rwp': spec_rwp, 'spec_broad': spec_broad,
               'flag': 0,'bflag': bflag, 'metrics': metrics, 
               'cal_const': cal_const, 'cal_corr': cal_corr}

    return ret




class mira():
    """
    handles cloud radar spectra
    """

    def __init__(self, files):
        """
        loads the .nc file via filename
        """
        print("---- cloud radar ---------------------------------------------------")

        self.files = files
        self.fspec = netCDF4.Dataset(self.files['spec'], 'r')
        self.system = "cr"  # system type for plot routine
        # ------------------------------------------------------------

        self.vel_list = self.fspec.variables["velocity"][:]
        # for some reason the velocity list is not sorted
        self.vel_list = np.sort(self.vel_list)
        self.time_list = self.fspec.variables["time"][:].astype(np.int32)
        # correct an offset estimated by plots (comparison with wp)
        # self.time_list = self.time_list-30.
        # for colrawi 2 an offset of 15sec is assumed
        self.time_list = self.time_list - 15
        # self.range_list = f.variables["range"][:]
        # compute mean steps
        self.delta_t = np.mean(np.diff(self.time_list))
        self.delta_v = np.mean(np.diff(self.vel_list))
        self.range_list = self.fspec.variables["range"][:]
        self.delta_h = np.mean(np.diff(self.range_list))
        assert abs(self.delta_h - 28.78) < 2, 'delta h too much off'
        print("load cloud radar file ", self.files['spec'])
        print("velocity, range, time", self.vel_list.shape,
              self.range_list.shape, self.time_list.shape)
        print(self.fspec.variables["Z"].Description)
        print("time range ", self.time_list[:2].astype(int),
              self.time_list[-2:].astype(int),
              h.ts_to_dt(self.time_list[0].astype(int)),
              h.ts_to_dt(self.time_list[-1].astype(int)))
        print("height range ", self.range_list[0], self.range_list[-1])
        print("height resolution [m]", self.delta_h)
        print("time resolution [s]", self.delta_t)
        print("velocity resolution [m/s]", self.delta_v)

        self.fmmclx = netCDF4.Dataset(self.files['mmclx'], 'r')
        print(self.fmmclx.__dict__)
        print(self.fmmclx.variables.keys())
        self.time_list_mmclx = self.fmmclx.variables["time"][:].astype(np.int32)
        self.range_list_mmclx = self.fmmclx.variables["range"][:]

        self.settings = {'decoupling': 22,
                         'thres_factor_co': 3.0,
                         'thres_factor_cx': 3.0}


    def __del__(self):
        """
        close file when object is deleted, sort of an evil hack (mmap, with statement...)
        """
        self.fspec.close()
        self.fmmclx.close()


    def get_profile(self, it, ir=(0,-1)):
        """
        get the mmclx velocity profile
        .. warning:: currently using VEL
        """
        vel = self.fmmclx.variables["VEL"][it, ir[0]:ir[1]]
        vel = np.ma.masked_invalid(vel)
        vel = np.ma.masked_less_equal(vel, -20.)
        Z = self.fmmclx.variables["Zg"][it, ir[0]:ir[1]]
        Z = np.ma.masked_invalid(Z)
        width = self.fmmclx.variables["RMSg"][it, ir[0]:ir[1]]
        width = np.ma.masked_invalid(width)

        return vel, Z, width

    #@profile
    def get_spectrum(self, sel_ts, sel_range, silent=False, range_average=False):
        """
        :sel_ts: either ts or (it, ts)
        :sel_range: either rg or (ir, rg)
        """

        logger.info('sel ts %s', sel_ts)
        if type(sel_ts) is tuple and type(sel_range) is tuple:
            it, sel_ts = sel_ts
            ir, sel_range = sel_range
        else:
            # it = np.where(self.time_list == min(self.time_list, key=lambda t: abs(sel_ts - t)))[0][0]
            it = h.argnearest2(self.time_list, sel_ts)
            # ir = np.where(self.range_list == min(self.range_list, key=lambda t: abs(sel_range - t)))[0][0]
            ir = h.argnearest2(self.range_list, sel_range)
        # itm = np.where(self.time_list_mmclx == min(self.time_list_mmclx, key=lambda t: abs(sel_ts - t)))[0][0]
        itm = h.argnearest2(self.time_list_mmclx, sel_ts)

        assert np.abs(sel_ts - self.time_list[it]) < self.delta_t, 'timestamps more than '+str(self.delta_t)+'s apart'
        assert np.abs(sel_ts - self.time_list_mmclx[itm]) < self.delta_t, 'timestamps more than '+str(self.delta_t)+'s apart'
        assert np.abs(self.time_list[it] - self.time_list_mmclx[itm]) < self.delta_t, 'timestamps more than '+str(self.delta_t)+'s apart'
        logger.debug('cloud radar indexes it %s itm %s it %s', it, itm, ir)
        
        if range_average:
            window = np.array([0.2, 0.6, 1., 0.6, 0.2])
            window /= np.sum(window)
            specZ_region = self.fspec.variables["Z"][:, ir-2:ir+3, it]
            specZ_mask = specZ_region == 0.
            specZ_min = np.min(specZ_region[~specZ_mask]) if not np.all(specZ_mask) else 1e-25
            specZ_region = h.fill_with(specZ_region, specZ_mask, specZ_min*0.9)
            specZ = np.average(specZ_region[:], axis=1, weights=window)
            specZ_mask = specZ < specZ_min*1.0
            specZ = h.fill_with(specZ, specZ_mask, 0.0)

            specSNRco_region = self.fspec.variables["SNRco"][:, ir-2:ir+3, it]
            specSNRco_mask = specSNRco_region == 0.
            specSNRco_min = np.min(specSNRco_region[~specSNRco_mask]) if not np.all(specSNRco_mask) else 1e-25
            # specSNRco_region = h.fill_with(specSNRco_region, specSNRco_mask, specSNRco_min*0.9)
            specSNRco = np.average(h.fill_with(specSNRco_region, specSNRco_mask, specSNRco_min*0.9)[:],
                                   axis=1, weights=window)
            specSNRco_mask = specSNRco < specSNRco_min*1.0
            specSNRco = h.fill_with(specSNRco, specSNRco_mask, 0.0)

            specSNRcx_region = self.fspec.variables["LDR"][:, ir-2:ir+3, it] * specSNRco_region
            specSNRcx_mask = np.logical_or(specSNRcx_region == 0., np.isnan(specSNRcx_region))
            specSNRcx_min = np.min(specSNRcx_region[~specSNRcx_mask]) if not np.all(specSNRcx_mask) else 1e-25
            specSNRcx_region = h.fill_with(specSNRcx_region, specSNRcx_mask, specSNRcx_min*0.9)
            specSNRcx = np.average(specSNRcx_region[:], axis=1, weights=window)
            specSNRcx_mask = specSNRcx < specSNRcx_min*1.0
            specSNRcx = h.fill_with(specSNRcx, specSNRcx_mask, 0.0)

            specLDR = specSNRcx/specSNRco
            specLDR_mask = np.logical_or(specSNRcx_mask, specSNRco_mask)
            specLDR = h.fill_with(specLDR, specLDR_mask, np.nan)
        else:
            specZ = self.fspec.variables['Z'][:,ir,it].ravel()
            specZ_mask = specZ == 0.
            #print('specZ.shape', specZ.shape, specZ)
            specLDR = self.fspec.variables['LDR'][:,ir,it].ravel()
            specLDR_mask = np.isnan(specLDR)
            specSNRco = self.fspec.variables['SNRco'][:,ir,it].ravel()
            specSNRco_mask = (specSNRco == 0.)
            #specSNRco = np.ma.masked_equal(specSNRco, 0)


        noise_thres = 1e-25 if np.all(specZ_mask) else specZ[~specZ_mask].min()*h.z2lin(self.settings['thres_factor_co'])
        spectrum = {'ts': self.time_list[it], 'range': self.range_list[ir], 'vel': self.vel_list,
                    'specZ': specZ[::-1], 'noise_thres': noise_thres, 'system': self.system}
        spectrum['specZ_mask'] = specZ_mask[::-1]
        spectrum['specSNRco'] = specSNRco[::-1]
        spectrum['specSNRco_mask'] = specSNRco_mask[::-1]
        spectrum['specLDR'] = specLDR[::-1]
        spectrum['specLDR_mask'] = specLDR_mask[::-1]
        
        spectrum['specZcx'] = spectrum['specZ']*spectrum['specLDR']
        spectrum['specZcx_mask'] = np.logical_or(spectrum['specZ_mask'], spectrum['specLDR_mask'])
        # print('test specZcx calc')
        # print(spectrum['specZcx_mask'])
        # print(spectrum['specZcx'])
        spectrum['specSNRcx'] = spectrum['specSNRco']*spectrum['specLDR']
        spectrum['specSNRcx_mask'] = np.logical_or(spectrum['specSNRco_mask'], spectrum['specLDR_mask'])

        if np.all(spectrum['specSNRcx_mask']):
            spectrum['minSNRcx'] = 1e-99
        else:
            spectrum['minSNRcx'] = spectrum['specSNRcx'][~spectrum['specSNRcx_mask']].min()
        thresSNRcx = spectrum['minSNRcx']*h.z2lin(self.settings['thres_factor_cx'])
        if np.all(spectrum['specSNRcx_mask']):
            minSNRco = 1e-99
            thresSNRco = 1e-99
            thresdecoup = 1e-99
            # maxSNRco = 1e-99
        else:
            minSNRco = spectrum['specSNRco'][~spectrum['specSNRco_mask']].min()
            thresSNRco = minSNRco*h.z2lin(self.settings['thres_factor_co'])
            thresdecoup = h.z2lin(h.lin2z(spectrum['specSNRco'])-self.settings['decoupling']+1.5)
            # maxSNRco = minSNRco*h.z2lin(self.settings['decoupling'])

        spectrum['validSNRco_mask'] = np.logical_or(spectrum['specSNRco_mask'], spectrum['specSNRco'] < thresSNRco)
        spectrum['validSNRcx_mask'] = np.logical_or(spectrum['specSNRcx_mask'], 
                                                    h.fill_with(spectrum['specSNRcx'], spectrum['specSNRcx_mask'], 1e-30) < thresSNRcx)
        spectrum['validSNRcx_mask'] = np.logical_or(spectrum['validSNRcx_mask'], 
                                                    h.fill_with(spectrum['specSNRcx'], spectrum['specSNRcx_mask'], 1e-30) < thresdecoup)
        spectrum['validSNRco_mask'] = np.logical_or(spectrum['validSNRcx_mask'], spectrum['validSNRco_mask'])
        
        spectrum['validSNRcx'] = spectrum['specSNRcx'].copy()
        spectrum['validSNRcx'][spectrum['validSNRcx_mask']] = 0
        spectrum['validSNRco'] = spectrum['specSNRco'].copy()
        spectrum['validSNRco'][spectrum['validSNRco_mask']] = 1
        
        spectrum['specLDRmasked'] = spectrum['validSNRcx']/spectrum['validSNRco']
        spectrum['specLDRmasked_mask'] = np.logical_or(spectrum['validSNRcx_mask'], spectrum['validSNRco_mask'])
        spectrum['specLDRmasked'][spectrum['specLDRmasked_mask']] = np.nan

        spectrum['decoupling'] = self.settings['decoupling']
        # and the bulk properties from the mmclx file
        spectrum['est_meanvel'] = self.fmmclx.variables["VELg"][it, ir]
        spectrum['est_width'] = self.fmmclx.variables["RMSg"][it, ir]
        spectrum['est_ldr'] = h.lin2z(self.fmmclx.variables["LDRg"][it, ir])

        #print('mira spectrum keys', spectrum.keys())
        return spectrum


class rwp():
    """
    handles radar wind profiler spectra

    :return:
    """

    def __init__(self, files):
        print("---- wind profiler -------------------------------------------------")
        self.files = files
        self.f = netCDF4.Dataset(self.files['rwp'], 'r')
        # prevent the automatic use of fill-value-masking
        # (meaning a strange conversion from string to float...)
        self.f.set_auto_maskandscale(False)
        self.system = "rwp"  # system type for plot routine

        # ------------------------------------------------------------
        # loading whole data at once seems to be a very bad idea
        # self.spectra = self.f.variables["Spectra"]

        self.total_snr_arr = self.f.variables["SNR"][:]
        self.total_snr_arr = np.ma.masked_greater(self.total_snr_arr, 3.00e+38)
        self.meanvelocity = self.f.variables["MeanDopplerVelocity"]
        self.meanvelocity = -self.meanvelocity[:]
        self.meanvelocity = np.ma.masked_greater(self.meanvelocity, 20.)
        self.meanvelocity = np.ma.masked_less(self.meanvelocity, -20.)
        # noise level in db; dimensions (Time, Heights)
        self.noiselevel = self.f.variables["NoiseLevel"][:]
        self.noiselevel = np.ma.masked_greater(self.noiselevel, 3.00e+38)
        self.time_list = self.f.variables["Timestamp"][:]
        # compute mean timestep
        self.delta_t = np.mean(np.diff(self.time_list))
        # build height index
        # self.delta_h = f.variables["HeightSpacing"][0]
        self.delta_h = 93.997
        # correct offset estimated by plots (comparison with cr); half a rangegate: 46.9985
        self.range_list = np.array(
            [i * self.delta_h + (448.0 - 46.9985)
             # [i*self.delta_h + (448.0+46.9985)
             for i in range(self.f.variables["Gates"][0])])
        # 0m/s bin left or right of array center???
        # Nyquist frequency in both directions
        self.delta_v = self.f.variables["NyquistFrequency"][0] / 256.
        # now the zero bin is right of center?

        #self.vel_list = np.array([i * self.delta_v for i in range(-255, 257)]) - 0.5 * self.delta_v  # correct vel_list
        self.vel_list = np.array([i * self.delta_v for i in range(-255, 257)]) # correct vel_list
        # self.vel_list = np.array([i*self.delta_v for i in range(-255, 257)])
        print("# vel as calculated ", self.vel_list[254:257])
        print("load wind profiler file ", self.files['rwp'])
        print("wipro time range ", self.time_list[:2].astype(int),
              self.time_list[-2:].astype(int),
              h.ts_to_dt(self.time_list[0].astype(int)),
              h.ts_to_dt(self.time_list[-1].astype(int)))
        print("height range ", self.range_list[0], self.range_list[-1])
        print("height resolution [m]", self.delta_h)
        print("time resolution [s]", self.delta_t)
        print("velocity resolution [m/s]", self.delta_v)
        print("spectra array dimensions ",
              self.f.variables["Spectra"].dimensions,
              self.f.variables["Spectra"].shape)
        print("nyquist frequency [m/s] ", self.f.variables["NyquistFrequency"][:3])
        # print("spectra units ", f.variables["Spectra"].units)

        self.settings = {'cal_const': 1.45e-15,
                         'thres_factor': 3.0}

    def __del__(self):
        """
        close file when object is deleted, sort of an evil hack (mmap, with statement...)
        """
        self.f.close()

    #@profile
    def get_spectrum(self, sel_ts, sel_range, silent=False, interp_vel=False, cal_corr=None):
        """
        :sel_ts: either ts or (it, ts)
        :sel_range: either rg or (ir, rg)

        :param interp_vel: velocity array to interpolate


        load the windprofiler spectrum, process it
        and height interpolate to the desired (velocity) resolution
        if a velocity list is given, the spectra is interpolated to the given values
        timestamps have to be of standard integer type::

        .. warning:: rwp spectrum is smoothed with a gaussian shaped running window
        """

        if type(sel_ts) is tuple and type(sel_range) is tuple:
            it, sel_ts = sel_ts
            ir, sel_range = sel_range
        else:
            it = np.where(self.time_list == min(self.time_list, key=lambda t: abs(sel_ts - t)))[0][0]
            ir = np.where(self.range_list == min(self.range_list, key=lambda t: abs(sel_range - t)))[0][0]
        assert np.abs(sel_ts - self.time_list[it]) < self.delta_t, 'timestamps more than '+str(self.delta_t)+'s apart'
        logger.info('selected it %s, ir %s', it, ir)

        #ntime = nearest(sel_timestamp, self.time_list, self.delta_t)
        #nheight = nearest(sel_height, self.range_list, self.delta_h)

        spectra = self.f.variables["Spectra"][it, ir, :512].copy()
        est_meanvel = self.meanvelocity[it, ir]
        noise_lvl = h.z2lin(self.noiselevel[it, ir])
        # noise level from measurement
        # calculate signal for system parameter estimation
        raw_spectra = spectra[::-1].copy()
        raw_spectra[raw_spectra < self.settings['thres_factor'] * noise_lvl] = noise_lvl
        #total_signal = np.sum(spectra - noise_lvl)

        # "calibrated" spectrum (should give the reflectivity in Ze)
        snrspectrum = spectra[::-1].copy()
        #spectrum[spectrum < self.settings['thres_factor'] * noise_lvl] = noise_lvl

        self.cal_const = self.settings['cal_const']
        if cal_corr != None and not np.isnan(cal_corr):
            self.cal_const = h.lin2z(self.settings['cal_const']) - cal_corr
            self.cal_const = h.z2lin(self.cal_const)
        logger.info("# systemparameter %s corr %s", self.cal_const, cal_corr)

        zspectrum = self.cal_const * (self.range_list[ir] ** 2) * snrspectrum
        convol_kernel = gauss_func(np.arange(-2, 3), 0, 1)
        # we have to keep the convolution for the gaussian, because sum is not 0
        convol_kernel /= np.sum(convol_kernel)
        zspectrum = np.convolve(zspectrum, convol_kernel, mode='same')
        snrspectrum = np.convolve(snrspectrum, convol_kernel, mode='same')

        # calibrated noise threshold and level
        noise_thres = self.cal_const * (self.range_list[ir] ** 2) * self.settings['thres_factor'] * noise_lvl
        noise_lvl = self.cal_const * (self.range_list[ir] ** 2) * noise_lvl

        spectrum = {'ts': self.time_list[it], 'range': self.range_list[ir], 'vel': self.vel_list,
                    'delta_v': self.delta_v,
                    'system': self.system}
        # interpolation of the spectrum
        if type(interp_vel) != bool:
            wp_interp = scipy.interpolate.interp1d(self.vel_list,
                                                   zspectrum,
                                                   bounds_error=False,
                                                   fill_value=0.0)
            zspectrum = wp_interp(interp_vel)
            wp_interp = scipy.interpolate.interp1d(self.vel_list,
                                                   snrspectrum,
                                                   bounds_error=False,
                                                   fill_value=0.0)
            snrspectrum = wp_interp(interp_vel)
            spectrum['vel'] = interp_vel

        noise_char = h.estimate_noise(zspectrum)
        spectrum['noise_lvl_hs'] = noise_char['noise_mean']
        logger.debug('noise comparison %s %s ', h.lin2z(noise_lvl), h.lin2z(spectrum['noise_lvl_hs']))

        spectrum['specZ_mask'] = zspectrum < noise_thres
        spectrum['specSNRco'] = snrspectrum
        # ['specSNRco_mask', 'system', 'ts', 'minSNRcx', 'specLDRmasked_mask', 'validSNRcx', 'specSNRcx', 'specSNRcx_mask', 
        # 'specZ', 'validSNRco_mask', 'specZ_mask', 'decoupling', 'specZcx', 'noise_thres', 'specLDRmasked', 'specLDR', 
        # 'specZcx_mask', 'vel', 'range', 'validSNRco', 'specLDR_mask', 'validSNRcx_mask', 'specSNRco']
        spectrum.update({'specZ': zspectrum, 'est_meanvel': est_meanvel, 'cal_const': self.cal_const,
                         'noise_thres': noise_thres, 'noise_lvl': noise_lvl,
                         })
        return spectrum


class mole_output():
    """
    handles output

    :return:
    """

    def __init__(self, files):
        print("---- mole -------------------------------------------------")
        print(files['mole'])
        self.files = files
        self.f = netCDF4.Dataset(self.files['mole'], 'r')
        # prevent the automatic use of fill-value-masking
        # (meaning a strange conversion from string to float...)
        self.f.set_auto_maskandscale(False)
        self.system = "mole"  # system type for plot routine

        # ------------------------------------------------------------
        # loading whole data at once seems to be a very bad idea
        # self.spectra = self.f.variables["Spectra"]

        self.time_list = self.f.variables["timestamp"][:]
        # compute mean timestep
        self.delta_t = np.mean(np.diff(self.time_list))
        # build height index
        # self.delta_h = f.variables["HeightSpacing"][0]
        # correct offset estimated by plots (comparison with cr); half a rangegate: 46.9985
        self.range_list = self.f.variables["range"][:]
        self.delta_h = np.mean(np.diff(self.range_list))
        # 0m/s bin left or right of array center???
        # Nyquist frequency in both directions

        print("mole time range ", self.time_list[:2].astype(int),
              self.time_list[-2:].astype(int),
              h.ts_to_dt(self.time_list[0].astype(int)),
              h.ts_to_dt(self.time_list[-1].astype(int)))
        print("height range ", self.range_list[0], self.range_list[-1])
        print("height resolution [m]", self.delta_h)

    def __del__(self):
        """
        close file when object is deleted, sort of an evil hack (mmap, with statement...)
        """
        self.f.close()

    def get_interpolated_profile(self, it, interp_heights):
        """ """

        # load mask
        self.flag = self.f.variables["quality_flag"][it,:]
        self.flag = np.ma.masked_less_equal(self.flag, -1.)
        # load vel data
        self.vair = self.f.variables["v"][it,:]
        self.vair = np.ma.masked_less(self.vair, -30.)
        self.vair = np.ma.masked_where(self.flag > 3, self.vair)

        # interpolate
        f = scipy.interpolate.interp1d(self.range_list[~np.ma.getmaskarray(self.vair[:])],
                                       self.vair[:].compressed(), kind='linear',
                                       bounds_error=False, fill_value=np.nan)
        interp_values = f(interp_heights)

        # this mask is only for the gaps in the interpolation
        f = scipy.interpolate.interp1d(self.range_list,
                                       np.ma.getmaskarray(self.vair[:]).astype(int),
                                       kind='linear',
                                       bounds_error=False, fill_value=1)
        interp_mask = f(interp_heights)
        #interp_values = np.ma.masked_where(interp_mask > 0.3, interp_values)
        interp_values = h.fill_with(interp_values, interp_mask > 0.3, np.nan)

        #now we interpolate the true mask
        f = scipy.interpolate.interp1d(self.range_list[:],
                                       self.flag[:].compressed(), kind='nearest',
                                       bounds_error=False, fill_value=np.nan)
        interp_flag = f(interp_heights)

        return interp_values, interp_flag