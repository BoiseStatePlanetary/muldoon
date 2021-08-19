"""
Test the functionality of Muldoon
"""

import numpy
import numpy as np
from numpy.random import normal
from muldoon import met_timeseries
from muldoon.utils import modified_lorentzian

def test_detrend_pressure_timeseries():

    # Create time-series
    time = np.linspace(-1, 1, 1000)
    baseline = 0.
    slope = 1.
    t0 = 0.
    DeltaP = 1.
    Gamma = 0.1
    profile = modified_lorentzian(time, baseline, slope, t0, DeltaP, Gamma) +\
            normal(scale=slope/10., size=len(time))
    mt = met_timeseries(time, profile)

    # Detrend
    window_size = 10*Gamma
    detrended_pressure = mt.detrend_pressure_timeseries(window_size)

    # Make sure detrend is within 
    assert np.isclose(np.std(mt.detrended_pressure), 0.2, atol=0.1)

def test_apply_lorentzian_matched_filter():
    # Test the matched filter analysis

    # Create time-series
    time = np.linspace(-1, 1, 1000)
    baseline = 0.
    slope = 1.
    t0 = 0.
    DeltaP = 1.
    Gamma = 0.01
    profile = modified_lorentzian(time, baseline, slope, t0, DeltaP, Gamma) +\
            normal(scale=slope/10., size=len(time))
    mt = met_timeseries(time, profile)

    # Detrend
    window_size = 10*Gamma
    detrended_pressure = mt.detrend_pressure_timeseries(window_size) 

    # Calculate filter
    conv = mt.apply_lorentzian_matched_filter(10.*Gamma, 1./np.pi)

    # Find the maximum
    mx_ind = np.argmax(mt.convolution)

    # Make sure convolution returns a strong peak at the right time
    assert ((np.abs(mt.time[mx_ind]) < Gamma) & (mt.convolution[mx_ind] > 7.))
