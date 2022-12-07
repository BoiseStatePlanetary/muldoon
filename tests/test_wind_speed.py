from math import isnan
import numpy
import numpy as np
from numpy.random import normal
from muldoon import utils
from muldoon import met_timeseries as met


# Create time-series
time = np.linspace(-1, 1, 1000)
baseline = 0.
ambient_speed_before = 0.
ambient_speed_after = 0.
max_speed = 9999999
t0 = 0.
Gamma = 0.01
#right_answer = np.array([baseline, time, t0, Gamma])
#profile = utils.wind_vortex_profile(max_speed, ambient_speed_before, baseline, time, t0, Gamma)
#wt = met.WindSpeedTimeseries(time, profile, 
#        popts = [right_answer],
#        uncs = [np.array([0, 0, 0, 0, 0])])
num_sigma = 5.

#def test_retrieve_vortices():
    # Make sure it retrieves the right number of data points
#    wt.retrieve_vortices()
#    assert(len(wt.vortices[0]["time"]) == 30)

    # And test retrieve_vortices for pressure time series
#    new_wt = met.WindSpeedTimeseries(time, profile, 
#           popts=[right_answer], uncs=[np.array([0, 0, 0, 0, 0])])
#    new_wt.retrieve_vortices()
#    assert(len(new_wt.vortices[0]["time"]) == 30)

