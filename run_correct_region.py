
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import collections

import datetime
import traceback
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

import spectra_mole
from spectra_mole import h

#reduce logging level for faster run
import logging
logging.getLogger('spectra_mole').setLevel(logging.WARNING)

import toml


# savepath = '../plots/{}/'.format(dt.strftime('%Y-%m-%d_%H%M%S'))
# if not os.path.isdir(savepath):
#     os.makedirs(savepath)


def convert_bounds(bounds_from_file):
    defaults = {'b_t': '00:30:00', 
                'e_t': '23:59:00',
                'b_rg': 500, 'e_rg':8000}

    def timedelta(s):
        return datetime.datetime.strptime(s, "%H:%M:%S") - datetime.datetime.strptime("00:00", "%H:%M")
    def datefromstring(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d")

    defaults.update(bounds_from_file)
    
    return {'b_dt': datefromstring(defaults['date'])+timedelta(defaults['b_t']),
            'e_dt': datefromstring(defaults['date'])+timedelta(defaults['e_t']),
            'b_rg': defaults['b_rg'], 'e_rg': defaults['e_rg']} 


toml_file = '../regions_config.toml'
with open(toml_file) as f:
    all_bounds = toml.loads(f.read())

toml_file = '../filenames.toml'
with open(toml_file) as f:
    all_files = toml.loads(f.read())

print('available cases ', all_bounds.keys())
selected_cases = sorted(list(all_bounds.keys()))
#selected_cases = ['20150617a']
selected_cases = ['20150801_dev']
#selected_cases = ['20150603']

# error in the first shot
#selected_cases = ['20150605a', '20150608a', '20150615', '20150617a', '20150618a', '20150626',
#                  '20150629', '20150707a', '20150707b', '20150722',
#                  '20150725', '20150801a', '20150803', '20150806', '20150807', '20150808',
#                  '20150809', '20150810', '20150813a', '20150816', '20150817', '20150825',
#                  '20150826', '20150828a']
#selected_cases = ['20150707a', '20150725', '20150801a', '20150813a', '20150828a']
#selected_cases = ['20150617a']

failed_dt = []
for case in selected_cases[:]:
    print('case ', case)
    bounds = convert_bounds(all_bounds[case])
    files = all_files[case]
    print('bounds', bounds)
    print('files', files)

    try:
        data = spectra_mole.correct_region(bounds, files)
        config = {'output_dir': '../output/', 'b_dt': bounds['b_dt']}
        spectra_mole.writer.save_data(data, config)


        data, mole_attr = spectra_mole.terminal_region(bounds, files)
        config = {'output_dir': '../output/', 'b_dt': bounds['b_dt']}
        spectra_mole.writer.save_terminal(data, config, mole_attr)
    except Exception as e:
        print(e.args)
        failed_dt.append([case, "error: {0}".format(e), "traceback: {}".format(traceback.format_exc())])
 

print('!!! failed datetimes ')
for elem in failed_dt:
    print(elem)
