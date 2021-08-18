"""
Test the functionality of Muldoon
"""

import numpy
import numpy as np
from numpy.random import normal
from muldoon import met_timeseries
from muldoon.utils import modified_lorentzian

# Create time-series
time = np.linspace(-1, 1, 1000)
baseline = 0.
slope = 1.
t0 = 0.
DeltaP = 1.
Gamma = 0.1
profile = modified_lorentzian(time, baseline, slope, t0, DeltaP, Gamma) +
    normal(scale=slope/10., size=len(time))
mt = met_timeseries(time, profile)

def test_detrend_pressure_timeseries():
    # Detrend
    window_size = 10*Gamma
    detrended_pressure = mt.detrend_pressure_timeseries(window_size)

    # Make sure detrend is within 
    assert np.isclose(np.std(mt.detrended_pressure), 0.2, atol=0.1)
