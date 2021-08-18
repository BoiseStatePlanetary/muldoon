import numpy as np
from scipy.signal import find_peaks, peak_widths, boxcar
from astropy.convolution import convolve as astropy_convolve

__all__ = ['met_timeseries']


class met_timeseries(object):
    """
    Process and analyze a meteorological time-series to search for vortices
    """

    def __init__(self, time, pressure, windspeed=None, wind_direction=None):
        """
        Args:
            time (float, array): time of meteorological time-series
            pressure (float, array): pressure measurements
            windspeed (float, array, optional): wind speed measurements
            wind_direction (float, array, optional): wind velocity aziumth
        """

        self.time = time
        self.pressure = pressure

        if(windspeed is not None):
            self.windspeed = windspeed

        if(wind_direction is not None):
            self.wind_direction = wind_direction

        # Filtered time-series
        self.detrended_pressure = None
        self.detrended_windspeed = None
        self.detrended_wind_direction = None

        # Time-series filters
        self.pressure_trend = None
        self.windspeed_trend = None
        self.wind_direction_trend = None

    def detrend_pressure_timeseries(self, window_width):
        """
        Applies boxcar filter to pressure time-series

        Args:
            window_width (float): width of window in the same units as time

        Returns:
            detrended pressure time-series (float array)

        """

        # Calculate number of points for window_width
        delta_t = np.median(self.time[1:] - self.time[0:-1])
        window_size = int(window_width/delta_t)
        # Check that window_size is odd
        if(window_size % 2 == 0): 
            window_size += 1

        self.pressure_trend = astropy_convolve(self.pressure,
                boxcar(window_size), boundary='extend', 
                preserve_nan=True)

        self.detrended_pressure = self.pressure - self.pressure_trend

        return self.detrended_pressure
