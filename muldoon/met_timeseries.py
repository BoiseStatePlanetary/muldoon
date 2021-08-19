import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, boxcar
from astropy.convolution import convolve as astropy_convolve
from scipy.stats import mode
from statsmodels.robust import mad

import muldoon.utils as utils

__all__ = ['MetTimeseries']


class MetTimeseries(object):
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

        # convolution of matched filter
        self.convolution = None

        self.peak_indices = None
        self.peak_widths = None

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
            lorentzian_depth (float): the depth of the filter; probably wants
            to be 1./np.pi
            num_fwhms (float, optional): how many full-width/half-maxes to
            generate matched filter; defaults to 6

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

    def find_vortices(self, detection_threshold=5, distance=20):
        """
        Finds distinct peaks in the matched-filter convolution, presumably
        vortex signals

        Args: 
            detection_threshold (float, optional): threshold for peak detection
            distance (int, optional): min number of point between peaks

        Returns:
            list of times and pressures for each vortex

        """

        if(self.convolution is None):
            raise ValueError("Run apply_lorentzian_matched_filter first!")

        ex = find_peaks(self.convolution, distance=distance)
        ind = self.convolution[ex[0]] >= detection_threshold

        pk_wds, _, _, _ = peak_widths(self.convolution, ex[0][ind])

        self.peak_indices = np.searchsorted(self.time, self.time[ex[0]][ind])
        self.peak_widths = pk_wds

        # Collect the vortices and sort by strength of convolution signal
        srt_ind = np.argsort(self.convolution[self.peak_indices])[::-1]

#       vortices = []
#       for ind in srt_ind:
#           # Use original, unfiltered data
#           vortices.append(
#           vortex = np.array([LTST_and_sol[ex[mx_ind] - matched_filter_num_fwhm*mx_width:
#                               ex[mx_ind] + matched_filter_num_fwhm*mx_width],
#                  sol_data['PRESSURE'][ex[mx_ind] - matched_filter_num_fwhm*mx_width:
#                                       ex[mx_ind] + matched_filter_num_fwhm*mx_width]])

        # Sort by strength of the signal
        srt_ind = np.argsort(self.convolution[self.peak_indices])[::-1]
        
        return self.peak_indices[srt_ind], self.peak_widths[srt_ind]

#   def fit_vortex(vortex, init_params, bounds, rescale_uncertainties=True, zoomed_in=None):

#   x, y = condition_vortex(vortex)

#   if(zoomed_in is not None):
#       ind = np.abs(x - init_params[2]) < zoomed_in
#       x = x[ind]
#       y = y[ind]

    # First fit out the long-term slope
#   fit_params = np.polyfit(x, y, 1)
#   detrended_data = y - np.polyval(fit_params, x)

#   popt, pcov = curve_fit(modified_lorentzian, x, y, p0=init_params, bounds=bounds)
#   ymod = modified_lorentzian(x, *popt)

#   if(rescale_uncertainties):
#       sd = mad(y - ymod)
#       red_chisq = redchisqg(y, ymod, deg=5, sd=sd)

#       pcov *= np.sqrt(red_chisq)

#   return popt, np.sqrt(np.diag(pcov))


    def make_conditioned_data_figure(self, fig=None, figsize=(10, 10), 
            aspect_ratio=16./9):
        """
        Make figure showing the data conditioning and analysis process -
        like Figure 1 of Jackson et al. (2021)

        Args:
            fig (matplotlib figure obj, optional): the figure object to use
            figsize (2x1 list, optional): inches x inches figure size
            aspect_ratio (float, optional): figure aspect ratio

        Returns:
            figure and all axes

        """

        # Boise State official colors in hex
        # https://www.boisestate.edu/communicationsandmarketing/brand-standards/colors/
        BoiseState_blue = "#0033A0"
        BoiseState_orange = "#D64309"

        if(self.peak_indices is None):
            raise ValueError("Run find_vortices first!")

        if(fig is None):
            fig = plt.figure(figsize=(figsize[0]*aspect_ratio, figsize[1]))

        # Add axes
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(223, sharex=ax1)
        ax3 = fig.add_subplot(222)
        ax4 = fig.add_subplot(224)

        ### Raw data ###
        ax1.plot(self.time, self.pressure, 
                marker='.', ls='', color=BoiseState_blue)
        ax1.text(0.05, 0.8, "(a)", fontsize=48, transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(labelsize=24, labelbottom=False)
        ax1.set_ylabel(r'$P\,\left({\rm Pa}\right)$', fontsize=36)


        ### Filtered data ###
        ax2.plot(self.time, self.detrended_pressure, 
                marker='.', ls='', color=BoiseState_blue)
        ax2.text(0.05, 0.05, "(b)", fontsize=48, transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(labelsize=24)
        ax2.set_xlabel("Time (hours)", fontsize=36)
        ax2.set_ylabel(r'$\Delta P\,\left( {\rm Pa} \right)$', fontsize=36)


        ### Convolution ###
        ax3.plot(self.time, self.convolution, 
                color=BoiseState_blue, ls='', marker='.')
        ax3.text(0.05, 0.8, "(c)", fontsize=48, transform=ax3.transAxes)
        ax3.grid(True)
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax3.tick_params(labelsize=24, labelleft=False, labelright=True)
        ax3.set_ylabel(r'$\left( F \ast \Delta P \right)$', fontsize=36)


        ### Fit vortex ###
        # Add lines in all plots highlighting the detections
        for cur_ex in self.peak_indices:
            ax1.axvline(self.time[cur_ex], 
                    color=BoiseState_orange, zorder=-1, ls='--', lw=3)
            ax2.axvline(self.time[cur_ex], 
                    color=BoiseState_orange, zorder=-1, ls='--', lw=3)
            ax3.axvline(self.time[cur_ex], 
                    color=BoiseState_orange, zorder=-1, ls='--', lw=3)

        return fig, ax1, ax2, ax3, ax4

