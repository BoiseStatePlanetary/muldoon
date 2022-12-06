from multiprocessing.dummy import Array
import pytest
import numpy as np
from muldoon import met_timeseries as met
from muldoon import utils as util
from muldoon.read_data import *

pressure_pds4_file = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089/sol_0001/WE__0001___________DER_PS__________________P02.xml'
pressure_csv_file = './tests/WE__0001___________DER_PS__________________P02.csv'
wind_pds4_file = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0180_0299/sol_0190/WE__0190___________DER_WS__________________P02.xml'
wind_csv_file = './tests/WE__0190___________DER_WS__________________P02.csv'

time = 22.1
max_wind_speed = 23
u_1 = 3.5
slope = 1.
t0 = 0.
Gamma = 0.01 
expected = 0.0000910710

def test_vortex_wind():
    """
    Tests the vortex wind function
    """
    v_act = 25
    r = .75
    D_act = 0.01
    expected = 0.3333185

    # rounded to 7 digits after the decimal
    v_r = round(util.vortex_wind(v_act, D_act, r), 7)

    if (v_r == expected):
        assert True
    if (v_r != expected):
        assert False

def test_wind_profile():
    """
    Tests the vortex wind profile function
    """
    # rounded to 9 digits after the decimal
    v_t = round(util.wind_vortex_profile(max_wind_speed, u_1, slope, time, t0, Gamma), 9)

    if(v_t == expected):
        assert True 
    if(v_t != expected):
        assert False

# Tests that the vortex returned is of type dictionary
def test_get_vortex_type():
    result = []
    result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED')
    vortex = util.get_vortex(result)

    if(type(vortex) == dict):
        assert True
    else:
        assert False

# Test for getting the time and data arrays. Verifying the data types returned and their values
def test_get_vortex_use():
    result = []
    result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED')
    vortex = util.get_vortex(result)

    print(vortex)

    if(type(vortex.get("time")) == np.ndarray and type(vortex.get("data")) == np.ndarray):
        assert True
    else:
        assert False
