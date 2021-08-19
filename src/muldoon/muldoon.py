import numpy as np
from scipy.signal import find_peaks, peak_widths, boxcar
from astropy.convolution import convolve as astropy_convolve
from scipy.stats import mode
from statsmodels.robust import mad

import utils

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
        # Calculate the sampling rate
        self.sampling = mode(time[1:] - time[0:-1]).mode[0]

        self.pressure = pressure

        if(windspeed is not None):
            self.windspeed = windspeed

        if(wind_direction is not None):
            self.wind_direction = wind_direction

        # Filtered pressure time-series
        self.detrended_pressure = None

        # Time-series filters
        self.pressure_trend = None

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

        self.window_size = window_size

        self.pressure_trend = astropy_convolve(self.pressure,
                boxcar(window_size), boundary='extend', 
                preserve_nan=True)

        self.detrended_pressure = self.pressure - self.pressure_trend
        self.detrended_pressure_scatter = np.nanstd(self.detrended_pressure)

        return self.detrended_pressure

    def write_out_detrended_timeseries(self, filename="out.csv", mode="w",
            test_mode=False):
        """
        Write out formatted text file of detrended time-series

        Args:
            filename (str, optional): path of file to which to write out data
            mode (str, optional): write mode; defaults to over-write
            test_mode (bool, optional): whether to actually write out file

        """

        if(self.detrended_pressure is None):
            raise ValueError("Need to detrend pressure!")

        # Construct write string
        write_str = "# time, pressure\n"

        for i in range(len(self.time) - 1):
            write_str += "%g, %g\n" %\
                    (self.time[i], self.detrended_pressure[i])

        # Don't write a new-line character for the last entry
        write_str += "%g, %g" % (self.time[-1], self.detrended_pressure[-1])
            
        if(~test_mode):
            f = open(filename, mode)
            f.write(write_str)
            f.close()

        return write_str

    def apply_lorentzian_matched_filter(self, lorentzian_fwhm,
            lorentzian_depth, num_fwhms=6.):
        """
        Applies Lorentzian matched filter to detrended pressure to find
        vortices

        Args:
            lorentzian_fwhm (float): the full-width/half-max of the matched
            filter
            lorentzian_depth (float): the depth of the filter; probably wants to be 1./np.pi
            num_fwhms (float, optional): how many full-width/half-maxes to generate matched filter; defaults to 6

        Returns:
            Results from matched filter (float array)

        """

        lorentzian_time = np.arange(-num_fwhms/2*lorentzian_fwhm, 
                num_fwhms/2.*lorentzian_fwhm, self.sampling)
        lorentzian = utils.modified_lorentzian(lorentzian_time, 0., 0., 0., 
                lorentzian_depth, lorentzian_fwhm)

        # Make sure the matched filter isn't wider than the signal itself
        if(len(lorentzian_time) > len(self.time)):
            raise ValueError("lorentzian_time is wider than detrended "+\
                    "pressure!")

        convolution =\
            np.convolve(self.detrended_pressure/\
            self.detrended_pressure_scatter, 
                    lorentzian, mode='same')

        # Shift and normalize
        med = np.nanmedian(convolution)
        md = mad(convolution)
        self.convolution = (convolution - med)/md

        return self.convolution

    def make_conditioned_data_figure(fig=None):
        """
        Make figure showing the data conditioning and analysis process -
        like Figure 1 of Jackson et al. (2021)

        """

        return
