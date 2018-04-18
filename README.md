# spectra mole

### Requirements

spectra mole requires python3 with following packages:
```
matplotlib==1.5.1
numpy==1.11.0
netCDF4==1.3.1
numba==0.37.0
scipy==1.0.1
```

### Setup

The spectra mole software package should be included in a file structure similar to this example:
```
├── raw_data
│   ├── [cloud radar, rwp, ...]
├── output
│   ├── 20150617_1459_mole_output.nc
│   └── 20150617_1459_mole_terminal_output.nc
├── plots
└── spectra_mole [the software under version control]
    ├── cases_from_csv.ipynb
    ├── data
    │   ├── results_2D_diff_statistics.npz
    │   └── results_2D_fit_statistics.npz
    ├── dev_calibration_corr.ipynb
    ├── mole_the_old_way.ipynb
    ├── plot_2d.ipynb
    ├── README.md
    ├── requirements.txt
    ├── run_correct_region.py
    └── spectra_mole
        ├── advection.py
        ├── helpers.py
        ├── __init__.py
        ├── recPeakFinder.py
        ├── viridis.py
        ├── VIS_Colormaps.py
        ├── vis.py
        └── writer.py

```

### License
Copyright 2018, Martin Radenz
[MIT License](http://www.opensource.org/licenses/mit-license.php)