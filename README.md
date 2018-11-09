# spectra mole

Software for combining the Doppler spectra of a radar wind profiler and a cloud radar.
The signal of falling particles is separeted from the clear air Bragg scattering using an
algorithm proposed by Radenz et al [2018, AMT](<https://doi.org/10.5194/amt-11-5925-2018>).

<a href="https://doi.org/10.5281/zenodo.1419486"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1419486.svg" alt="DOI"></a>


Technical documentation is available at [spectra_mole-doc](https://martin-rdz.github.io/spectra_mole-doc/)

### Requirements

spectra mole requires python3 with following packages:
```
matplotlib==1.5.1
numpy==1.11.0
netCDF4==1.3.1
numba==0.37.0
scipy==1.0.1
toml==0.9.2
```

### Setup

The spectra mole software package should be included in a file structure similar to this example:
```
├── raw_data
│   ├── [cloud radar, rwp, ...]
│   └── [... as defined in filenames.toml]
├── output
│   └── [standard directiory for the output netcdf files]
├── plots
├── filenames.toml      [defines the filenames for each day]
├── regions_config.toml [defines the time that is processed]
├── cases.csv           [cases for the single spectrum processing]
└── spectra_mole        [the software under version control]
    ├── data
    │   ├── results_2D_diff_statistics.npz
    │   └── results_2D_fit_statistics.npz
    ├── cases_from_csv.ipynb
    ├── dev_calibration_corr.ipynb
    ├── mole_the_old_way.ipynb
    ├── plot_2d_cr_grid.py
    ├── plot_2d.ipynb
    ├── plot_2d_rwp_grid.py
    ├── plot_2d_terminal.ipynb
    ├── README.md
    ├── requirements.txt
    ├── run_correct_region.py
    └── spectra_mole
        ├── advection.py
        ├── attenuation.py
        ├── helpers.py
        ├── __init__.py
        ├── recPeakFinder.py
        ├── viridis.py
        ├── VIS_Colormaps.py
        ├── vis.py
        └── writer.py

```

Please update your meta information in the `output_meta.toml` file.

### Run with Docker

Deployment may be simplified using a docker image which includes all the dependencies.
A Dockerfile is included in this repository.

    # build the Dockerfile
    docker build -t mole-runtime .
    # run an interactive bash on this image
    docker run -v `pwd`:/mole2 -v [folder with data]:/colrawi -p 8890:8890 -it mole-runtime /bin/bash
    # or to build the documentation as well
    docker run -v `pwd`/..:/mole2 -v [folder with data]:/colrawi -p 8890:8890 -it mole-runtime /bin/bash


### References
Radenz, M., Bühl, J., Lehmann, V., Görsdorf, U., and Leinweber, R.: Combining cloud radar and radar wind profiler for a value added estimate of vertical air motion and particle terminal velocity within clouds, Atmos. Meas. Tech., 11, 5925-5940, <https://doi.org/10.5194/amt-11-5925-2018>, 2018.


### License
Copyright 2018, Martin Radenz
[MIT License](<http://www.opensource.org/licenses/mit-license.php>)