from numpy.random import normal
from scipy.stats import median_abs_deviation as mad
from muldoon.met_timeseries import *
from muldoon.met_timeseries import PressureTimeseries
from muldoon.met_timeseries import TemperatureTimeseries
from muldoon.utils import break_at_gaps, modified_lorentzian, fit_vortex, write_out_plot_data
from muldoon.read_data import *
import numpy as np
import matplotlib.pyplot as plt
from starlette.responses import StreamingResponse
import pandas as pd
from io import BytesIO


#function to create our basic plot
def create_fig():
    fig = plt.figure(figsize=(10,10))
    return fig


#creates an ATS graph given a url directed towards MEDA website
def create_ATS_Graph(ATSurl):
    fig = create_fig()
    ax = fig.add_subplot(111)
    temperature_time, temperature = read_Perseverance_ATS_data(ATSurl, which_ATS=1)
    ax.scatter(temperature_time, temperature)
    buf = BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    return StreamingResponse(buf)


#ATS detrended data
def create_ATS_Graph_Detrended(ATSurl):
    fig = create_fig()
    ax = fig.add_subplot(111)
    time, temperature = read_Perseverance_ATS_data(ATSurl, which_ATS=1)
    tt = TemperatureTimeseries(time, temperature)
    window_size = 1 # 1 hour
    detrended_temperature = tt.detrend_timeseries_boxcar(window_size)
    ax.scatter(tt.time, tt.detrended_data, marker='.')
    buf = BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    return StreamingResponse(buf)
    
#creates an PS graph given a url directed towards MEDA website
def create_PS_Graph(PSurl):
    fig = create_fig()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    pressure_time, pressure = read_Perseverance_PS_data(PSurl)
    ax2.scatter(pressure_time, pressure, color='orange')
    buf = BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    return StreamingResponse(buf)

#creates an PS graph with color coded data given a url directed towards MEDA website
def create_PS_Graph_with_gaps(PSurl):
    fig = create_fig()
    ax = fig.add_subplot(111)
    time, pressure = read_Perseverance_PS_data(PSurl)
    times, pressures = break_at_gaps(time, pressure)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i in range(len(times)):
    #     plt.axvline(time[0:-1][gaps][i])
        ax.scatter(times[i], pressures[i], marker='.')
    ax.grid(True)
    buf = BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    return StreamingResponse(buf)

#Detrended PS data
def create_PS_Graph_detrend(PSurl):
    fig = create_fig()
    ax = fig.add_subplot(111)
    time, pressure = read_Perseverance_PS_data(PSurl)
    mt = MetTimeseries(time, pressure)
    window_size = 500./3600 # window size is 500 seconds
    mt.detrend_timeseries_boxcar(window_size)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(mt.time, mt.detrended_data, marker='.')
    ax.grid(True)
    buf = BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    return StreamingResponse(buf)


#Vortex data for PS
def create_PS_Vortex_Signals(PSurl):
    time, pressure = read_Perseverance_PS_data(PSurl)
    pt = PressureTimeseries(time, pressure)
    window_size = 500./3600 # window size is 500 seconds
    pt.detrend_timeseries_boxcar(window_size)
    matched_filter_width = 2.*pt.sampling
    matched_filter_depth = 1./np.pi
    distance_between_peaks = 30
    detection_threshold = 7.
    conv = pt.apply_lorentzian_matched_filter(matched_filter_width, matched_filter_depth)
    vortices = pt.find_vortices(detection_threshold=detection_threshold, distance=distance_between_peaks)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.plot(pt.time, pt.convolution, ls='', marker='.')
    ax.axhline(detection_threshold)

    for i in range(len(pt.peak_indices)):
        ax.axvline(pt.time[pt.peak_indices[i]], color="orange", zorder=-1)
    buf = BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    return StreamingResponse(buf)



#Shows both ATS and PS on one graph
def create_ATS_PS_Graph(PSurl,ATSurl):
    fig = create_fig()
    ax = fig.add_subplot(111)
    temperature_time, temperature = read_Perseverance_ATS_data(ATSurl, which_ATS=1)
    ax.scatter(temperature_time, temperature)
    ax2 = ax.twinx()
    pressure_time, pressure = read_Perseverance_PS_data(PSurl)
    ax2.scatter(pressure_time, pressure, color='orange')
    buf = BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    return StreamingResponse(buf)
