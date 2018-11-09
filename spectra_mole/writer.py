#! /usr/bin/env python3
# coding=utf-8

"""
Author: radenz@tropos.de

writer for the data
"""
import datetime
import netCDF4
import numpy as np
import subprocess
import toml
from . import helpers as h

def save_item(dataset, item_data):
    """
    Save an item to the dataset with the data given as a dict

    Args:
        dataset (:obj:netCDF4.Dataset): netcdf4 Dataset to add
        item_data (dict): with the data to add, for example:

    ==================  ===============================================================
     Key                 Example                            
    ==================  ===============================================================
     ``var_name``        Z                                  
     ``dimension``       ('time', 'height')                 
     ``arr``             self.corr_refl_reg[:].filled()     
     ``long_name``       "Reflectivity factor"              
     **optional**                                             
     ``comment``         "Wind profiler reflectivity factor corrected by cloud radar"
     ``units``           "dBz"                              
     ``missing_value``   -200.                              
     ``plot_range``      [-50., 20.]                        
     ``plot_scale``      "linear"                           
     ``vartype``         np.float32                         
    ==================  ===============================================================
    """

    if 'vartype' in item_data.keys():
        item = dataset.createVariable(item_data['var_name'], item_data['vartype'], item_data['dimension'])
    else:
        item = dataset.createVariable(item_data['var_name'], np.float32, item_data['dimension'])

    item[:] = item_data['arr']
    item.long_name = item_data['long_name']
    if 'comment' in item_data.keys():
        item.comment = item_data['comment']
    if 'units' in item_data.keys():
        item.units = item_data['units']
    if 'units_html' in item_data.keys():
        item.units_html = item_data['units_html']
    if 'missing_value' in item_data.keys():
        item.missing_value = item_data['missing_value']
    if 'plot_range' in item_data.keys():
        item.plot_range = item_data['plot_range']
    if 'plot_scale' in item_data.keys():
        item.plot_scale = item_data['plot_scale']
    if 'axis' in item_data.keys():
        item.axis = item_data['axis']


def get_git_hash():
    label = subprocess.check_output(["git", "describe", "--always"])
    return label


def save_data(data, config):
    """write the data to a netcdf file
    standard path and filename %Y%m%d_%H%M_mole_output.nc 

    Args:
        data (dict): output data from correct_region
        config (dict): configuration
                
    """
    # convert the times to cloudnet fomat                               
    dt_list = [h.ts_to_dt(ts) for ts in data['time_list']]
    hours_cn = np.array([dt.hour + dt.minute / 60. + dt.second / 3600. for dt in dt_list])
                                                                          
    filename = config['output_dir'] + config['b_dt'].strftime("%Y%m%d_%H%M") + "_mole_output.nc" 
    print("writing results to ", filename)
    #dataset = netCDF4.Dataset(filename, 'w', format='NETCDF4') 
    dataset = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    dim_time = dataset.createDimension('time', data['time_list'].shape[0])
    dim_height = dataset.createDimension('height', data['range_list'].shape[0])
    dim_mode = dataset.createDimension('mode', 1)
    dim_mom = dataset.createDimension('moments', 2) 
    #                                                 
    mode = dataset.createVariable('system_parameter', np.float32, ('mode',))   
    mode[:] = data['cal_const_used']                                      
    mode.comment = 'Wind profiler system parameter used for calibration'
    #                                                                   
    times = dataset.createVariable('timestamp', np.int32, ('time',)) 
    times[:] = data['time_list'].astype(np.int32) 
    times.units = 'Unix timestamp [s]'                  
                                                          
    times_cn = dataset.createVariable('time', np.float32, ('time',)) 
    times_cn[:] = hours_cn.astype(np.float32)                           
    times_cn.units = "hours since " + dt_list[0].strftime('%Y-%m-%d') + " 00:00:00 +00:00"  
    times_cn.long_name = "Decimal hours from midnight UTC"  
    times_cn.axis = "T"

    # self.save_item(dataset, {'var_name': 'height', 'dimension': ('height', ),
    #                         'arr': data['range_list'].astype(np.float32),
    #                         'long_name': "height of rangegate center",
    #                         'units': "[m]"})
    # cloudnet compatible range
    save_item(dataset, {'var_name': 'range', 'dimension': ('height', ),
                        'arr': data['range_list'].astype(np.float32),
                        'long_name': "Range from antenna to the centre of each range gate",
                        'units': "m", 'axis': 'Z'})

    save_item(dataset, {'var_name': 'Z', 'dimension': ('time', 'height'),
                        'arr': data['bragg_weight_Z'], 'long_name': "Reflectivity factor",
                        'comment': "Wind profiler reflectivity factor corrected by cloud radar (only Bragg contribution)",
                        'units': "dBZ", 'missing_value': -200., 'plot_range': (-50., 20.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'v', 'dimension': ('time', 'height'),
                        'arr': data['bragg_weight_v'], 'long_name': "Vertical velocity",
                        'comment': "Wind profiler vertical velocity corrected by cloud radar (only Bragg contribution)",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (-2., 2.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'width', 'dimension': ('time', 'height'),
                        'arr': data['bragg_weight_width'], 'long_name': "Spectral width",
                        'comment': "Wind profiler spectral width (standard deviation) corrected by cloud radar (only Bragg contribution)",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (0.01, 4.),
                        'plot_scale': "logarithmic"})
    
    save_item(dataset, {'var_name': 'Z_fit', 'dimension': ('time', 'height'),
                        'arr': data['bragg_fit_Z'], 'long_name': "Reflectivity factor",
                        'comment': "Wind profiler reflectivity factor corrected by cloud radar (only Bragg contribution)",
                        'units': "dBZ", 'missing_value': -200., 'plot_range': (-50., 20.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'v_fit', 'dimension': ('time', 'height'),
                        'arr': data['bragg_fit_v'], 'long_name': "Vertical velocity",
                        'comment': "Wind profiler vertical velocity corrected by cloud radar (only Bragg contribution)",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (-2., 2.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'width_fit', 'dimension': ('time', 'height'),
                        'arr': data['bragg_fit_width'], 'long_name': "Spectral width",
                        'comment': "Wind profiler spectral width (standard deviation) corrected by cloud radar (only Bragg contribution)",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (0.01, 4.),
                        'plot_scale': "logarithmic"})

    save_item(dataset, {'var_name': 'Z_raw', 'dimension': ('time', 'height'),
                        'arr': data['rwp_raw_Z'], 'long_name': "Reflectivity factor",
                        'comment': "Wind profiler reflectivity factor raw",
                        'units': "dBZ", 'missing_value': -200., 'plot_range': (-50., 20.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'v_raw', 'dimension': ('time', 'height'),
                        'arr': data['rwp_raw_vel'], 'long_name': "Vertical velocity",
                        'comment': "Wind profiler vertical velocity raw",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (-2., 2.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'width_raw', 'dimension': ('time', 'height'),
                        'arr': data['rwp_raw_width'], 'long_name': "Spectral width",
                        'comment': "Wind profiler vertical velocity raw",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (0.01, 4.),
                        'plot_scale': "logarithmic"})

    save_item(dataset, {'var_name': 'noise_lvl', 'dimension': ('time', 'height'),
                        'arr': data['rwp_raw_noise'], 'long_name': "Reflectivity factor",
                        'comment': "Wind profiler noise level",
                        'units': "dBZ", 'missing_value': -200., 'plot_range': (-60., 20.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'Z_cr', 'dimension': ('time', 'height'),
                        'arr':data['cr_Z'][:], 'long_name': "cloud radar reflectivity",
                        'comment': "cloud radar reflectivity [self calculated]",
                        'units': "dBZ", 'missing_value': -200., 'plot_range': (-50., 20.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'v_cr', 'dimension': ('time', 'height'),
                        'arr': data['cr_vel'], 'long_name': "cloud radar mean velocity",
                        'comment': "cloud radar mean velocity [self calculated]",
                        'units': "m s-1", 'missing_value': -99., 'plot_range': (-2., 2.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'width_cr', 'dimension': ('time', 'height'),
                        'arr': data['cr_width'], 'long_name': "cloud radar width",
                        'comment': "cloud radar width [self calculated]",                        
                        'missing_value': -99., 'plot_range': (0.01, 4.),
                        'plot_scale': "linear"})
    broad = dataset.createVariable('sigma_broadening', np.float32, ('time', 'height'))
    broad[:, :] = data['broadening']
    broad.long_name = "beam width broadening sigma"

    cal_const = dataset.createVariable('est_cal_const', np.float32, ('time', 'height'))
    cal_const[:, :] = data['cal_const'][:]
    cal_const.long_name = "estimated calibration constant"
    
    cal_corr = dataset.createVariable('cal_corr', np.float32, ('time', 'height'))
    cal_corr[:, :] = data['cal_corr'][:]
    cal_corr.long_name = "potential correction of the calibration"

    save_item(dataset, {'var_name': 'separation', 'dimension': ('time', 'height'),
                        'arr': data['separation'], 'long_name': "peak separation",
                        'comment': "peak separation",
                        'units': "", 'missing_value': -99., 'plot_range': (0., 5.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'contrast', 'dimension': ('time', 'height'),
                        'arr': data['contrast'], 'long_name': "contrast",
                        'comment': "contrast",
                        'units': "dBZ", 'missing_value': -99., 'plot_range': (0., 40.),
                        'plot_scale': "linear"})

    save_item(dataset, {'var_name': 'error_diff', 'dimension': ('time', 'height'),
                        'arr': data['error_weight'], 'long_name': "error for weighting function",
                        'comment': "error for weighting function",
                        'units': "m s-1", 'missing_value': -99., 'plot_range': (-0.3, 0.3),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'error_fit', 'dimension': ('time', 'height'),
                        'arr': data['error_fit'], 'long_name': "error of peak fitting",
                        'comment': "error for peak fitting",
                        'units': "m s-1", 'missing_value': -99., 'plot_range': (-0.3, 0.3),
                        'plot_scale': "linear"})

    #save_item(dataset, {'var_name': 'mira_v_term', 'dimension': ('time', 'height'),
    #                             'arr': self.v_term[:], 'long_name': "cloud radar terminal velocity [self calculated]",
    #                             'comment': "cloud radar terminal velocity [self calculated]",
    #                             'units': "m s-1", 'missing_value': -99., 'plot_range': (-2., 2.),
    #                             'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'quality_flag', 'dimension': ('time', 'height'),
                        'arr': data['flag'][:], 'long_name': "Quality Flag",
                        'comment': str(data['flag_doc']),
                        'missing_value': -1, 'plot_range': (0, 8),
                        'plot_scale': "linear"})
    keys = ['mod_calibration', 'unsecure_calibration', 'particle_influence',
            'removed_moments', 'melting_layer', 'plankton', 'low_snr']
    save_item(dataset, {'var_name': 'binary_flag', 'dimension': ('time', 'height'),
                        'arr': data['bflag'][:], 'long_name': "Binary Flag",
                        'comment': str(keys),
                        'missing_value': -1, 'plot_range': (0, 127),
                        'plot_scale': "linear"})

    with open('output_meta.toml') as output_meta:
            meta_info = toml.loads(output_meta.read())

    dataset.description = meta_info['description_air']
    #dataset.settings = str(self.bound)
    dataset.location = meta_info["location"]
    dataset.institution = meta_info["institution"]
    dataset.contact = meta_info["contact"]
    dataset.creation_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    #dataset.mole_commit_id = get_git_hash()
    dataset.day = dt_list[0].day
    dataset.month = dt_list[0].month
    dataset.year = dt_list[0].year
    dataset.close()


def save_terminal(data, config, mole_globalattr):
    """write the data to a netcdf file
    standard path and filename %Y%m%d_%H%M_mole_output.nc 

    Args:
        data (dict): output data from terminal_region
        config (dict): configuration
                
    """
    dt_list = [h.ts_to_dt(ts) for ts in data['time_list']]
    hours_cn = np.array([dt.hour + dt.minute / 60. + dt.second / 3600. for dt in dt_list])
                                                                          
    filename = config['output_dir'] + config['b_dt'].strftime("%Y%m%d_%H%M") + "_mole_terminal_output.nc" 
    print("writing results to ", filename)
    #dataset = netCDF4.Dataset(filename, 'w', format='NETCDF4') 
    dataset = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    dim_time = dataset.createDimension('time', data['time_list'].shape[0])
    dim_height = dataset.createDimension('height', data['range_list'].shape[0])
    #                                                                                                                
    times = dataset.createVariable('timestamp', np.int32, ('time',)) 
    times[:] = data['time_list'].astype(np.int32) 
    times.units = 'Unix timestamp [s]'                  
                                                          
    times_cn = dataset.createVariable('time', np.float32, ('time',)) 
    times_cn[:] = hours_cn.astype(np.float32)                           
    times_cn.units = "hours since " + dt_list[0].strftime('%Y-%m-%d') + " 00:00:00 +00:00"  
    times_cn.long_name = "Decimal hours from midnight UTC"  
    times_cn.axis = "T"

    save_item(dataset, {'var_name': 'range', 'dimension': ('height', ),
                        'arr': data['range_list'].astype(np.float32),
                        'long_name': "Range from antenna to the centre of each range gate",
                        'units': "m", 'axis': 'Z'})

    save_item(dataset, {'var_name': 'v_term', 'dimension': ('time', 'height'),
                        'arr': data['terminal_vel'], 'long_name': "Terminal particle velocity",
                        'comment': "Terminal particle velocity (corrected by mole-RWP data)",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (-4., 4.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'v_air', 'dimension': ('time', 'height'),
                        'arr': data['air_vel'], 'long_name': "Vertical air velocity",
                        'comment': "Vertical air velocity (corrected by mole-RWP data)",
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -99., 'plot_range': (-2., 2.),
                        'plot_scale': "linear"})
    save_item(dataset, {'var_name': 'quality_flag', 'dimension': ('time', 'height'),
                        'arr': data['quality_flag'], 'long_name': "Quality Flag",
                        'comment': str(data['flag_doc']),
                        'units': "m s-1", 'units_html': "m s<sup>-1</sup>",
                        'missing_value': -1, 'plot_range': (0, 8),
                        'plot_scale': "linear"})

    save_item(dataset, {'var_name': 'Z', 'dimension': ('time', 'height'),
                        'arr': data['Z'], 'long_name': "Reflectivity",
                        'comment': "Reflectivity",
                        'units': "dBZ", 'missing_value': -200., 'plot_range': (-50., 20.),
                        'plot_scale': "linear"})

    if 'history' in mole_globalattr and 'description' in mole_globalattr:
        print('found old history')
        history = str(mole_globalattr['history']) + "/n" + mole_globalattr['creation_time'] + str(mole_globalattr['description'])
    elif 'description' in mole_globalattr:
        history = mole_globalattr['creation_time'] + ' ' + str(mole_globalattr['description'])
    else:
        history = "none"

    with open('output_meta.toml') as output_meta:
        meta_info = toml.loads(output_meta.read())

    dataset.description = meta_info['description_terminal']
    dataset.location = meta_info["location"]
    dataset.institution = meta_info["institution"]
    dataset.contact = meta_info["contact"]

    mole_globalattr['history'] = history
    mole_globalattr['creation_time'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    #mole_globalattr['mole_commit_id'] = get_git_hash()
    print('git id', get_git_hash())
    for key, value in mole_globalattr.items():
        # print(key, value)
        setattr(dataset, key, value)