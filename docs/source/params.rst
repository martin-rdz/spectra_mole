#####################
Adjustable parameters
#####################


General
-------

.. topic:: plankton_mask

    :function: :func:`spectra_mole.broaden_spectrum`
    :purpose: ignore all bins with an ldr greater 
        (optinally add a maximum range for this threshold)
    :standard: -13


.. topic:: Q value

    :function: several
    :purpose: factor between noise_level and noise_threshold
    :standard: 3


.. topic:: plausible moments

    :function: :func:`spectra_mole.filter_moments`
    :purpose: filter the list of rwp moments for physically unplausible values. Bragg peak has to be right of cr peak and has a certain width
    :standard: `(v_moment > v_cr + sigma_cr & width_moment > 0.08 and right_bound_moment > right_bound_cr)`


.. topic:: rwp calibration fine

    :function: :func:`spectra_mole.check_rwp_calibration`
    :purpose: if there is a peak in the cloud radar, check if the rwp particle peak is close to it
    :standard: 
        - less than 2dB difference -> ok
        - less then 24dB difference and correction factor less 20dB -> correct calibration
        - else mark as unreliable



.. topic:: compare given noise level with Hildebrand and Sehkon

    :function: :func:`spectra_mole.check_noise_level`
    :purpose: ind elevated noise level (e.g. clutter, melting layer) by comparison with the 
        noise level from the standard signal processing algorithm
    :standard: noise level estimated with HS more than 4 dB larger 
        than standard signal processing noise level


.. topic:: velocity range for calibration

    :function: :func:`spectra_mole.estimate_calibration`
    :purpose: use only power within a small spectral band for calibration to exclude Bragg and Mie contribution
    :standard: bin 91 to 118 (vel -3.0 to -0.7)


weighting function method
-------------------------

.. topic:: weighting function smoothing

    :function: :func:`spectra_mole.correct_with_weighting`
    :purpose: supress noise in the weighting function
    :standard: convol_kernel = [0.3, 0.5, 1., 0.5, 0.3]/2.6


.. topic:: weighting function threshold

    :function: :func:`spectra_mole.correct_with_weighting`
    :purpose: weights below this threshold are supressed by the snr^-1
    :standard: 0.6


peak fitting method
-------------------

.. topic:: fuzzy membership minimum peak width

    :function: :func:`spectra_mole.correct_with_fuzzy`
    :standard: 3 bins


.. topic:: peak right of cloud radar

    :function: :func:`spectra_mole.correct_with_fuzzy`
    :standard: ight > cr_edge + 2 & left > cr_edge -1 & within the raw RWP peak


.. topic:: boundaries in the fitting algorithm

    :function: :func:`spectra_mole.fit_peak`
    :purpose: constrain the fitting algorithm to the properties of clear air bragg peaks
    :standard: 
        - v: ``[-10, 10]``, width: ``[0.02, 0.45]``, Z: ``[-60, -10]``
        - slightly modified if peak with highest reflectivity is below -25dBZ
