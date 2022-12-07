from math import isnan
from unittest import result
import numpy as np
from numpy.random import normal
from muldoon import met_timeseries as met
from muldoon import utils 
from muldoon.read_data import *
from pds4_tools import pds4_read
import pandas as pd

pressure_pds4_file = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089/sol_0001/WE__0001___________DER_PS__________________P02.xml' #remote
# pressure_pds4_file = './tests/WE__0001___________DER_PS__________________P02.xml' #local
pressure_csv_file = './tests/WE__0001___________DER_PS__________________P02.csv'
wind_pds4_file = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0180_0299/sol_0190/WE__0190___________DER_WS__________________P02.xml' #remote
# wind_pds4_file = 'WE__0190___________DER_WS__________________P02.xml' #local
wind_csv_file = './tests/WE__0190___________DER_WS__________________P02.csv'
ats_csv_file = './tests/WE__0010___________CAL_ATS_________________P01.csv'
ats_pds4_file = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_calibrated_env/sol_0000_0089/sol_0010/WE__0010___________CAL_ATS_________________P01.xml'

# Test not needed, since read_data is now a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_file_verification_pds4_file_exist():
#     _, file_status =  read_data(pressure_pds4_file) 
#     assert(file_status == 1)

# Test not needed, since read_data is now a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_file_verification_pds4_file_does_not_exist():
#     filename = 'https://wrong_link.xml'
#     _, file_status =  read_data(filename)
#     # wrong link
#     assert(file_status == 0)

# Test not needed, since read_data is now a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_file_verification_csv_file_exist():
#     _, file_status =  read_data(pressure_csv_file)
#     assert(file_status == 1)

# Test not needed, since read_data is now a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_file_verification_csv_file_does_not_exist():
#     filename = './tests/wrong_file.csv'
#     _, file_status =  read_data(filename)
#     # wrong csv file
#     assert(file_status == 0)

# Test not needed, since time_to_hours_decimal is a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_time_to_hours_decimal_ltst():
#     times_str = ['15:28:01']
#     delta_sols = [0.0]
#     time_field = 'LTST'
#     expected = [15.46694444]
#     result = time_to_hours_decimal(times_str, delta_sols,time_field)
#     assert(round(result[0], 8) == expected[0])

# Test not needed, since time_to_hours_decimal is a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_time_to_hours_decimal_lmst():
#     times_str = ['16:05:31.315']
#     delta_sols = [0.0]
#     time_field = 'LMST'
#     expected = [16.09203194]
#     result = time_to_hours_decimal(times_str, delta_sols,time_field)
#     assert(round(result[0], 8) == expected[0])

# Test not needed, since process_data is now a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_process_data_ltst():
#     result,_ = process_data(pressure_pds4_file,'ltst', None)
#     expected = [15.46694444]
#     assert(round(result[0], 8) == expected[0])

# Test not needed, since process_data is now a private function.
# Rename read_data function in read_data.py (remove "__" prefix) 
# to perform futher testing.
# def test_process_data_lmst():
#     result,_ = process_data(pressure_csv_file,'lmst', None)
#     expected = [16.09203194]
#     assert(round(result[0], 8) == expected[0])

def test_read_Perseverance_PS_data_csv():
    _,result = read_Perseverance_PS_data(pressure_csv_file, None, 'LTST')
    expected = [715.96]
    assert(round(result[0], 2) == expected[0])

def test_read_Perseverance_PS_data_pds4():
    _,result = read_Perseverance_PS_data(pressure_pds4_file, None, 'LTST')
    expected = [715.96]
    assert(round(result[0], 2) == expected[0])


def test_read_Perseverance_WS_data_pds4_hws():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED')
    expected = [1.63]
    assert(round(result[0], 2) == expected[0])

def test_read_Perseverance_WS_data_csv_hws():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED')
    expected = [1.63]
    assert(round(result[0], 2) == expected[0])

def test_read_Perseverance_WS_data_pds4_hwsu():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED_UNCERTAINTY')
    expected = [0.0]
    assert(round(result[0], 2) == expected[0])

def test_read_Perseverance_WS_data_csv_hwsu():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED_UNCERTAINTY')
    assert(isnan(result[0]))

def test_read_Perseverance_WS_data_pds4_vws():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='lmst', wind_field='vws')
    expected = [0.0]
    assert(result[0] == expected[0])

def test_read_Perseverance_WS_data_csv_vws():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='lmst', wind_field='vws')
    assert(isnan(result[0]))

def test_read_Perseverance_WS_data_pds4_vwsu():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='lmst', wind_field='vwsu')
    expected = [0.0]
    assert(result[0] == expected[0])

def test_read_Perseverance_WS_data_csv_vwsu():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='lmst', wind_field='vwsu')
    assert(isnan(result[0]))

def test_read_Perseverance_WS_data_pds4_wd():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='lmst', wind_field='wd')
    expected = [269.33]
    assert(result[0] == expected[0])

def test_read_Perseverance_WS_data_csv_wd():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='lmst', wind_field='wd')
    expected = [269.33]
    assert(result[0] == expected[0])
    
def test_read_Perseverance_WS_data_pds4_wdu():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='lmst', wind_field='wdu')
    expected = [0.0]
    assert(result[0] == expected[0])

def test_read_Perseverance_WS_data_csv_wdu():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='lmst', wind_field='wdu')
    assert(isnan(result[0]))

def test_read_Perseverance_WS_data_pds4_bbufr():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='lmst', wind_field='bbufr')
    expected = [0.0]
    assert(result[0] == expected[0])

def test_read_Perseverance_WS_data_csv_bbufr():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='lmst', wind_field='bbufr')
    assert(isnan(result[0]))

def test_read_Perseverance_WS_data_pds4_rs():
    _,result = read_Perseverance_WS_data(wind_pds4_file, sol=None, time_field='lmst', wind_field='rs')
    expected = [1]
    assert(result[0] == expected[0])

def test_read_Perseverance_WS_data_csv_rs():
    _,result = read_Perseverance_WS_data(wind_csv_file, sol=None, time_field='lmst', wind_field='rs')
    expected = [1]
    assert(result[0] == expected[0])

def test_read_Perseverance_ATS_data_ATS1():
    _,result = read_Perseverance_ATS_data(ats_csv_file,1,'LTST',None)
    expected = 240.67
    assert(result[0]== expected)

def test_read_Perseverance_ATS_data_ATS2():
    _,result = read_Perseverance_ATS_data(ats_csv_file,2,'LTST',None)
    expected = 257.01
    assert(result[0]== expected)

def test_read_Perseverance_ATS_data_ATS3():
    _,result = read_Perseverance_ATS_data(ats_csv_file,3,'LTST',None)
    expected = 240.06
    assert(result[0]== expected)

def test_read_Perseverance_ATS_data_ATS4():
    _,result = read_Perseverance_ATS_data(ats_csv_file,4,'LTST',None)
    expected = 243.92
    assert(result[0]== expected)

def test_read_Perseverance_ATS_data_ATS5():
    _,result = read_Perseverance_ATS_data(ats_csv_file,5,'LTST',None)
    expected = 250.99
    assert(result[0]== expected)


