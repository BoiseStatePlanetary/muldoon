"""
Test the functionality of Muldoon
"""

import numpy
import numpy as np
from numpy.random import normal
from muldoon import met_timeseries as met
from muldoon import utils 
from muldoon.read_data import *

# Create time-series
time = np.linspace(-1, 1, 1000)
baseline = 0.
slope = 1.
t0 = 0.
DeltaP = 1.
Gamma = 0.01
right_answer = np.array([baseline, slope, t0, DeltaP, Gamma])
profile = utils.modified_lorentzian(time, baseline, slope, t0, DeltaP, Gamma) +\
    normal(scale=slope/20., size=len(time))
mt = met.PressureTimeseries(time, profile)

# Create temperature time-series
DeltaT = -10. #K
temp_profile = utils.modified_lorentzian(time, baseline, slope, t0, DeltaT,
        Gamma) + normal(scale=slope/20., size=len(time))
tt = met.TemperatureTimeseries(time, temp_profile, 
        pressure_vortex_popts = [right_answer],
        pressure_vortex_uncs = [np.array([0, 0, 0, 0, 0])])

# Detrend
window_size = Gamma
detrended_data = mt.detrend_timeseries_boxcar(window_size)

# Calculate filter
conv = mt.apply_lorentzian_matched_filter(2.*mt.sampling, 1./np.pi)

# Max mis-match between fit and right answers allowed
num_sigma = 5.

def test_detrend_timeseries_boxcar():

    # Make sure detrend is within 
    assert np.isclose(np.std(mt.detrended_data), 0.1, atol=0.1)

def test_apply_lorentzian_matched_filter():
    # Test the matched filter analysis

    # Find the maximum
    mx_ind = np.argmax(mt.convolution)

    # Make sure convolution returns a strong peak at the right time
    assert ((np.abs(mt.time[mx_ind]) < 2.*Gamma) &\
            (mt.convolution[mx_ind] > 5.))

def test_find_vortices():

    # Test find_vortices
    vortices = mt.find_vortices()
    # Make sure the max peak in the convolution is where the vortex is
    assert(mt.time[mt.peak_indices[0]] < 2.*Gamma)

def test_fit_vortex():

    vortices = mt.find_vortices()

    # Test vortex fit
    old_popt, old_unc = utils.fit_vortex(vortices[0], [0., 1., 0., 1., 0.01], 
                          [[-1, -1, np.min(vortices[0]["time"]), 0, 0],
                           [1, 1, np.max(vortices[0]["time"]), 2, 1]],
                          sigma=vortices[0]["scatter"])

    # Make sure best-fit parameters all match the right answers 
    assert(np.max(np.abs(old_popt - right_answer)/old_unc) < num_sigma)

def test_init_params_bounds():
    vortices = mt.find_vortices()

    # Test fit parameter initialization and bounds calculation
    init_params = mt._determine_init_params(vortices[0])
    bounds = mt._determine_bounds(vortices[0], init_params)
    
    popt, unc = utils.fit_vortex(vortices[0], init_params, bounds)

    # Make sure best-fit parameters all match the right answers
    assert(np.max(np.abs(popt - right_answer)/unc) < num_sigma)

def test_fit_all_vortices():
    # Make sure fit_all_vortices works, too
    vortices = mt.find_vortices()
    popts, uncs = mt.fit_all_vortices()

    # Make sure best-fit parameters all match the right answers
    assert(np.max(np.abs(popts[0] - right_answer)/uncs[0]) < num_sigma)

def test_retrieve_vortices():
    tt.retrieve_vortices()

    # Make sure it retrieves the right number of data points
    assert(len(tt.vortices[0]["time"]) == 30)
