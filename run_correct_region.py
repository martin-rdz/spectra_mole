
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import collections

import datetime
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

import spectra_mole
from spectra_mole import h

#reduce logging level for faster run
import logging
logging.getLogger('spectra_mole').setLevel(logging.WARNING)


files = {'spec': '/colrawi/cloudradar/spectra/D20150617_T0000_0000_Lin_zspc2nc_v1_02_standard.nc4',
         'mmclx': '/colrawi/cloudradar/mmclx/20150617_0000.mmclx',
         'rwp': '/colrawi/windprofiler/nc/wp_20150617_043156.nc.iop',
         'cloudnet': '/colrawi/cloudnet/20150617_lindenberg_categorize.nc',
         'mole': '/mole2/output/20150617_1459_mole_output.nc'}

files = {'spec': '/colrawi/cloudradar/spectra/D20150801_T0000_0000_Lin_zspc2nc_v1_02_standard.nc4',
         'mmclx': '/colrawi/cloudradar/mmclx/20150801_0000.mmclx',
         'rwp': '/colrawi/windprofiler/nc/wp_20150801_003008.nc',
         'cloudnet': '/colrawi/cloudnet/20150801_lindenberg_categorize.nc',
         'mole': '/mole2/output/20150801_1230_mole_output.nc'}

#files = {'spec': '../../colrawi2/cloudradar/spectra/D20150617_T0000_0000_Lin_zspc2nc_v1_02_standard.nc4',
#         'mmclx': '../../colrawi2//cloudradar/mmclx/20150617_0000.mmclx',
#         'rwp': '../../colrawi2/windprofiler/nc/wp_20150617_043156.nc.iop',
#         'cloudnet': '../../colrawi2/cloudnet/20150617_lindenberg_categorize.nc'}

print(files['spec'])

# savepath = '../plots/{}/'.format(dt.strftime('%Y-%m-%d_%H%M%S'))
# if not os.path.isdir(savepath):
#     os.makedirs(savepath)

#@profile
def correct_region(bounds, files):
    """ """
    cr = spectra_mole.mira(files)
    rwp = spectra_mole.rwp(files)
    advect = spectra_mole.advection.cloudnet_advect(files['cloudnet'])
    
    coverage = {}
    coverage['mira_begin'] = h.dt_to_ts(bounds['b_dt']) - cr.time_list[0] # positive is ok
    coverage['mira_end'] = cr.time_list[-1] - h.dt_to_ts(bounds['e_dt']) # positive is ok
    coverage['rwp_begin'] = h.dt_to_ts(bounds['b_dt']) - rwp.time_list[0] # positive is ok
    coverage['rwp_end'] = rwp.time_list[-1] - h.dt_to_ts(bounds['e_dt']) # positive is ok
    assert all(i >= -70 for i in list(coverage.values())), "not enough coverage {}".format(str(coverage))
    
    b_it = np.where(rwp.time_list == min(rwp.time_list, key=lambda t: abs(h.dt_to_ts(bounds['b_dt']) - t)))[0][0]
    b_ir = np.where(rwp.range_list == min(rwp.range_list, key=lambda t: abs(bounds['b_rg'] - t)))[0][0]
    e_it = np.where(rwp.time_list == min(rwp.time_list, key=lambda t: abs(h.dt_to_ts(bounds['e_dt']) - t)))[0][0]
    e_ir = np.where(rwp.range_list == min(rwp.range_list, key=lambda t: abs(bounds['e_rg'] - t)))[0][0]
    e_it += 1
    e_ir += 1
    print('it ', b_it, e_it, 'ir ', b_ir, e_ir)
    
    # corrected velocity region
    var = np.empty((e_it - b_it, e_ir - b_ir))
    var[:] = np.nan
    data = collections.defaultdict(var.copy)
    
    data['time_list'] = rwp.time_list[b_it:e_it]
    data['range_list'] = rwp.range_list[b_ir:e_ir]
    data['cal_const_used'] = rwp.settings['cal_const']
    
    spectra_no = (e_it - b_it) * (e_ir - b_ir)
    spectra_ct = 0
    for it in range(b_it, e_it):
        sel_ts = (it, rwp.time_list[it])
        for ir in range(b_ir, e_ir):
            sel_rg = (ir, rwp.range_list[ir])
            corr = spectra_mole.correct_pixel(cr, rwp, advect, sel_ts, sel_rg)
            
            data['flag'][it - b_it, ir - b_ir] = corr['flag']
            if 'flag_doc' in corr.keys():
                data['flag_doc'] = corr['flag_doc']
            data['bflag'][it - b_it, ir - b_ir] = spectra_mole.bin2int(
                spectra_mole.bflag2str(corr['bflag'])[0])
            
            data['bragg_weight_Z'][it - b_it, ir - b_ir] = corr['bragg_weighting'].Z
            data['bragg_weight_v'][it - b_it, ir - b_ir] = corr['bragg_weighting'].v
            data['bragg_weight_width'][it - b_it, ir - b_ir] = corr['bragg_weighting'].width
            data['bragg_fit_Z'][it - b_it, ir - b_ir] = corr['bragg_fit'].Z
            data['bragg_fit_v'][it - b_it, ir - b_ir] = corr['bragg_fit'].v
            data['bragg_fit_width'][it - b_it, ir - b_ir] = corr['bragg_fit'].width
            
            data['rwp_raw_Z'][it - b_it, ir - b_ir] = corr['spec_rwp']['moments'][0].Z
            data['rwp_raw_vel'][it - b_it, ir - b_ir] = corr['spec_rwp']['est_meanvel']
            data['rwp_raw_width'][it - b_it, ir - b_ir] = corr['spec_rwp']['moments'][0].width
            data['rwp_raw_noise'][it - b_it, ir - b_ir] = corr['spec_rwp']['noise_lvl']
            
            data['cr_Z'][it - b_it, ir - b_ir] = corr['spec_broad']['moments'][0].Z
            data['cr_vel'][it - b_it, ir - b_ir] = corr['spec_broad']['moments'][0].v
            data['cr_width'][it - b_it, ir - b_ir] = corr['spec_broad']['moments'][0].width
            data['broadening'][it - b_it, ir - b_ir] = corr['spec_broad']['sigma_b']
            # vterm missing
            
            data['separation'][it - b_it, ir - b_ir] = corr['metrics'][0]
            data['contrast'][it - b_it, ir - b_ir] = corr['metrics'][1]
            
            data['cal_const'][it - b_it, ir - b_ir] = corr['cal_const']
            data['cal_corr'][it - b_it, ir - b_ir] = corr['cal_corr']

            
    #data['error_weight'] = spectra_mole.estimate_error('diff', data['separation'], data['contrast'])
    #data['error_fit'] = spectra_mole.estimate_error('fit', data['separation'], data['contrast'])

    data['error_weight'] = spectra_mole.estimate_error('diff', data['separation'], data['contrast'])
    data['error_fit'] = spectra_mole.estimate_error('fit', data['separation'], data['contrast'])
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print('key is ndarray', key)
            print('contains nans? ', np.any(np.isnan(data[key])))
            #data[key] = np.ma.masked_invalid(data[key], copy=True)
#     self.v_term = self.clara.v_reg - np.ma.masked_where(self.flag_region > 3.0, self.corr_vel_reg)
    print('time_list', data['time_list'])
    print('flag', data['flag'])
    print('cal_corr', data['cal_corr'])
    print('bragg_weight_Z', data['bragg_weight_Z'])
    print('separation', data['separation'])
    print('error fit', data['error_fit'])
    #logger.debug('done with correction')

    return data


bounds = {'b_dt': datetime.datetime(2015, 6, 17, 20, 59),
          'e_dt': datetime.datetime(2015, 6, 17, 21, 0),
          'b_rg': 500, 'e_rg': 1000}

bounds = {'b_dt': datetime.datetime(2015, 6, 17, 17),
          'e_dt': datetime.datetime(2015, 6, 17, 21, 0),
          'b_rg': 500, 'e_rg': 8000}

bounds = {'b_dt': datetime.datetime(2015, 6, 17, 15),
          'e_dt': datetime.datetime(2015, 6, 17, 23, 59),
          'b_rg': 500, 'e_rg': 8000}

# test the melting layer case in 2d
# bounds = {'b_dt': datetime.datetime(2015, 6, 17, 20, 15),
#           'e_dt': datetime.datetime(2015, 6, 17, 23, 59),
#           'b_rg': 2300, 'e_rg': 3400}

# tiny test case
# bounds = {'b_dt': datetime.datetime(2015, 6, 17, 20, 15),
#           'e_dt': datetime.datetime(2015, 6, 17, 20, 17),
#           'b_rg': 500, 'e_rg': 8000}

bounds = {'b_dt': datetime.datetime(2015, 8, 1, 12, 30),
         'e_dt': datetime.datetime(2015, 8, 1, 21, 0),
         'b_rg': 500, 'e_rg': 8000}

data = correct_region(bounds, files)
config = {'output_dir': '../output/'}
spectra_mole.writer.save_data(data, config)




def terminal_region(bounds, files):
    cr = spectra_mole.mira(files)
    mole = spectra_mole.mole_output(files)
    coverage = {}

    coverage['mira_begin'] = h.dt_to_ts(bounds['b_dt']) - cr.time_list[0] # positive is ok
    coverage['mira_end'] = cr.time_list[-1] - h.dt_to_ts(bounds['e_dt']) # positive is ok
    coverage['mole_begin'] = h.dt_to_ts(bounds['b_dt']) - mole.time_list[0] # positive is ok
    coverage['mole_end'] = mole.time_list[-1] - h.dt_to_ts(bounds['e_dt']) # positive is ok
    assert all(i >= -70 for i in list(coverage.values())), "not enough coverage {}".format(str(coverage))

    b_it = np.where(cr.time_list == min(cr.time_list, key=lambda t: abs(h.dt_to_ts(bounds['b_dt']) - t)))[0][0]
    b_ir = np.where(cr.range_list == min(cr.range_list, key=lambda t: abs(bounds['b_rg'] - t)))[0][0]
    e_it = np.where(cr.time_list == min(cr.time_list, key=lambda t: abs(h.dt_to_ts(bounds['e_dt']) - t)))[0][0]
    e_ir = np.where(cr.range_list == min(cr.range_list, key=lambda t: abs(bounds['e_rg'] - t)))[0][0]
    e_it += 1
    e_ir += 1
    print('it ', b_it, e_it, 'ir ', b_ir, e_ir)

    # corrected velocity region
    #var = np.empty((e_it - b_it, cr.range_list.shape[0]))
    var = np.empty((e_it - b_it, e_ir - b_ir))
    var[:] = np.nan
    data = collections.defaultdict(var.copy)
    
    data['time_list'] = cr.time_list[b_it:e_it]
    data['range_list'] = cr.range_list[b_ir:e_ir]
    data['flag_doc'] = mole.f.variables["quality_flag"].comment

    for it in range(b_it, e_it):
    #for it in range(b_it, b_it+10):
        it_mole = h.argnearest2(mole.time_list, cr.time_list[it])

        print('it cr ', it, ' mole ', it_mole)
        print('times cr ', cr.time_list[it],  h.ts_to_dt(cr.time_list[it]), 
                 mole.time_list[it_mole], h.ts_to_dt(mole.time_list[it_mole]))

        if abs(cr.time_list[it] - mole.time_list[it_mole]) > 5:
            # mole profile too far away (on time axis) -> mask
            print('at {} not fitting mole profile'.format(h.ts_to_dt(cr.time_list[it])))

        else:
            vair, flag_air = mole.get_interpolated_profile(it_mole, cr.range_list[b_ir:e_ir])
            vvert, Z, width = cr.get_profile(it, (b_ir, e_ir))
            assert vair.shape == vvert.shape, "shapes of the velocity profiles not fitting"

            vterm = vvert - vair
            #print(vterm.mask[15:25])
            #print(vterm.mask[-15:])
            data['terminal_vel'][it - b_it, :] = vterm
            data['vertical_vel'][it - b_it, :] = vvert
            data['air_vel'][it - b_it, :] = vair
            data['quality_flag'][it - b_it, :] = flag_air
            data['Z'][it - b_it, :] = 10*np.log10(Z)
            data['width'][it - b_it, :] = width
    
    return data, mole.f.__dict__
    
data, mole_attr = terminal_region(bounds, files)
config = {'output_dir': '../output/'}
spectra_mole.writer.save_terminal(data, config, mole_attr)