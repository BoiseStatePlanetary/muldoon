import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, boxcar
from astropy.convolution import convolve as astropy_convolve
from scipy.stats import mode
from statsmodels.robust import mad

import muldoon.utils as utils

__all__ = ['MetTimeseries']


class MetTimeseries(object):
    """
    A meteorological time-series 
    """

    def __init__(self, time, data):
        """
        Args:
            time (float, array): time of meteorological time-series
            data (float, array): measurements
        """

        self.time = time
        # Calculate the sampling rate
        self.sampling = mode(time[1:] - time[0:-1]).mode[0]

        self.data = data

        # Filtered time-series
        self.detrended_data = None
        self.window_width = None
        self.detrended_data_scatter = None

        # Time-series filters
        self.data_trend = None

    def detrend_timeseries_boxcar(self, window_width, 
            deal_with_gaps=True):
        """
        Applies boxcar filter to time-series

        Args:
            window_width (float): width of window in the same units as time
            deal_with_gaps (bool, optional): whether to check for gaps

        Returns:
            detrended time-series (float array)

        """

        self.window_width = window_width

        # Queue up to deal with gaps
        local_time = [self.time]
        local_data = [self.data]

        if(deal_with_gaps):
            local_time, local_data = utils.break_at_gaps(self.time,
                    self.data)
            
        # Calculate number of points for window_width
        window_size = int(window_width/self.sampling)
        # Check that window_size is odd
        if(window_size % 2 == 0): 
            window_size += 1

        self.window_size = window_size

        # Empty out the Nones
        self.detrended_data = np.array([])

        # Detrend, piece at a time
        for i in range(len(local_time)):

            local_data_trend = astropy_convolve(local_data[i],
                    boxcar(window_size), boundary='extend', 
                    preserve_nan=True)

            if(self.data_trend is None):
                self.data_trend = local_data_trend
            else:
                self.data_trend = np.append(self.data_trend,
                        local_data_trend)

            self.detrended_data =\
                    np.append(self.detrended_data, 
                            local_data[i] - local_data_trend)

        self.detrended_data_scatter = np.nanstd(self.detrended_data)

        return self.detrended_data

    def write_out_detrended_timeseries(self, data_name="data", 
            filename="out.csv", mode="w", test_mode=False):
        """
        Write out formatted text file of detrended time-series

        Args:
            filename (str, optional): path of file to which to write out data
            data_name (str, optional): name of data time-series
            mode (str, optional): write mode; defaults to over-write
            test_mode (bool, optional): whether to actually write out file

        """

        if(self.detrended_data is None):
            raise ValueError("Need to detrend data!")

        # Construct write string
        write_str = "# time, %s\n" % (data_name)

        for i in range(len(self.time) - 1):
            write_str += "%g, %g\n" %\
                    (self.time[i], self.detrended_data[i])

        # Don't write a new-line character for the last entry
        write_str += "%g, %g" % (self.time[-1], self.detrended_data[-1])
            
        if(~test_mode):
            f = open(filename, mode)
            f.write(write_str)
            f.close()

        return write_str

class PressureTimeseries(MetTimeseries):
    """
    A meteorological time-series tailored to pressure and used for 
    looking for vortices

    """

    def __init__(self, time, pressure_data):
        """
        Args:
            time (float, array): time of meteorological time-series
            pressure_data (float, array): pressure measurements
        """
        super().__init__(time, pressure_data)

        # matched filter parameters
        self.matched_filter_fwhm = None
        self.matched_filter_depth = None

        # detection threshold
        self.detection_threshold = None

        # FWHMs for vortices that are detected
        self.num_fwhms = None

        # convolution of matched filter
        self.convolution = None

        self.peak_indices = None
        self.peak_widths = None

        # Collection of time-series for individual vortices
        self.vortices = None

        # Vortex fit parameters and uncertainties
        self.popts = None
        self.uncs = None

    def apply_lorentzian_matched_filter(self, lorentzian_fwhm=None,
            lorentzian_depth=1./np.pi, num_fwhms=6.):
        """
        Applies Lorentzian matched filter to detrended pressure to find
        vortices

        Args:
            lorentzian_fwhm (float, optional): the full-width/half-max of the matched filter
            lorentzian_depth (float, optional): the depth of the filter; probably wants to be 1./np.pi
            num_fwhms (float, optional): how many full-width/half-maxes to
            generate matched filter; defaults to 6

        Returns:
            Results from matched filter (float array)

        """

        if(self.detrended_data_scatter is None):
            raise ValueError("Run detrend_timeseries_boxcar first!")

        if(lorentzian_fwhm is None):
            lorentzian_fwhm = 2.*self.sampling

        self.matched_filter_fwhm = lorentzian_fwhm
        self.matched_filter_depth = lorentzian_depth

        self.num_fwhms = num_fwhms

        lorentzian_time = np.arange(-num_fwhms/2*lorentzian_fwhm, 
                num_fwhms/2.*lorentzian_fwhm, self.sampling)
        lorentzian = utils.modified_lorentzian(lorentzian_time, 0., 0., 0., 
                lorentzian_depth, lorentzian_fwhm)

        # Make sure the matched filter isn't wider than the signal itself
        if(len(lorentzian_time) > len(self.time)):
            raise ValueError("lorentzian_time is wider than detrended "+\
                    "pressure!")

        convolution =\
            np.convolve(self.detrended_data/\
            self.detrended_data_scatter, 
                    lorentzian, mode='same')

        # Shift and normalize
        med = np.nanmedian(convolution)
        md = mad(convolution)
        self.convolution = (convolution - med)/md

        return self.convolution

    def find_vortices(self, detection_threshold=5, distance=20, fwhm_factor=6,
            min_num_points=5):
        """
        Finds distinct peaks in the matched-filter convolution, presumably
        vortex signals

        Args: 
            detection_threshold (float, optional): threshold for peak detection
            distance (int, optional): min number of point between peaks
            fwhm_factor (int, optional): when returning vortices, how wide a time window to return
            min_num_points (int, optional): minimum number of points required

        Returns:
            list of times and pressures for each vortex

        """

        self.detection_threshold = detection_threshold

        if(self.convolution is None):
            raise ValueError("Run apply_lorentzian_matched_filter first!")

        ex = find_peaks(self.convolution, distance=distance)
        ind = self.convolution[ex[0]] >= detection_threshold

        pk_wds, _, _, _ = peak_widths(self.convolution, ex[0][ind])

        # Sort from largest peak to smallest
        srt_ind = np.argsort(self.convolution[ex[0][ind]])[::-1]
        self.peak_indices = ex[0][ind][srt_ind]
        self.peak_widths = pk_wds[srt_ind]

        self.vortices = []
        for i in range(len(self.peak_indices)):
            mn_ind = int(self.peak_indices[i] -\
                    fwhm_factor/2*int(self.peak_widths[i]))
            mx_ind = int(self.peak_indices[i] +\
                    fwhm_factor/2*int(self.peak_widths[i]))

            # Make sure there are enough points in the vortex
            if(mx_ind - mn_ind > min_num_points):
                # Use original, unfiltered data
                self.vortices.append({"time": self.time[mn_ind:mx_ind],
                    "data": self.data[mn_ind:mx_ind],
                    "scatter": self.detrended_data_scatter*\
                            np.ones_like(self.time[mn_ind:mx_ind])})

        return self.vortices

    def fit_all_vortices(self, 
            filepath=None, figsize=(10, 10), aspect_ratio=16./9,
            pressure_units="Pa", time_units="Hours", vortex_time_units="s"):
        """
        Fit all vortices with modified Lorentzian and return fit parameters and
        uncertainties

        Args:
            filepath (str, optional): If not None, gives the filepath and filename stem to which to write vortex figures
            fig (matplotlib figure, optional): figure into which to plot
            figsize (float tuple, optional): size of figure
            aspect_ratio (float, optional): figure aspect ratio
            pressure/time/vortex_time_units (str, optional): axes labels

        Returns:
            list of two arrays, the first with best fit parameters and the
            second with uncertainties

        """

        if(self.vortices is None):
            raise ValueError("Run find_vortices first!")
        if(len(self.vortices) == 0):
            raise ValueError("There are no vortices!")

        self.popts = list()
        self.uncs = list()
        for i in range(len(self.vortices)):
            # Estimate initial parameters
            init_params = self._determine_init_params(self.vortices[i])

            # Estimate bounds
            bounds = self._determine_bounds(self.vortices[i], init_params)

            try:
                popt, unc = utils.fit_vortex(self.vortices[i], init_params, 
                        bounds, sigma=self.vortices[i]["scatter"])

                self.popts.append(popt)
                self.uncs.append(unc)

                # Make figure
                if(filepath is not None):
                    fig = plt.figure(figsize=(figsize[0]*aspect_ratio,
                        figsize[1]))

                    ax = fig.add_subplot(111)

                    # Remember! Times are in hours!
                    time = self.vortices[i]["time"]
                    ydata = self.vortices[i]["data"] - popt[0]
                    yerr = self.vortices[i]["scatter"]
                    model_func = lambda time, popt:\
                            utils.modified_lorentzian(time, *popt) - popt[0]
                    model = utils.plot_vortex(time, 
                            popt[2], ydata, model_func, popt, ax, yerr=yerr)

                    ax.text(0.05, 0.05, "%s, vortex %i" % (filepath, i),
                            fontsize=24, transform=ax.transAxes)
                    ax.grid(True)
                    ax.tick_params(labelsize=24)
                    ax.set_xlabel(r'$t - t_0\,\left( {\rm %s} \right)$' %\
                        (vortex_time_units), fontsize=36)
                    ax.set_ylabel(r'$\Delta P\,\left( {\rm %s} \right)$' %\
                        (pressure_units), fontsize=36)

                    fig.savefig("%s_vortex%i.jpg" % (filepath, i), 
                            bbox_inches="tight", dpi=500)

                    # Write out data
                    filename = filepath + "_data_vortex%i.csv" % (i)
                    utils.write_out_plot_data((time - popt[2])*3600., 
                            ydata - popt[0],
                            "Time", "DeltaP", yerr=yerr, filename=filename)

                    filename = filepath + "_model_vortex%i.csv" % (i)
                    utils.write_out_plot_data((time - popt[2])*3600., model, 
                            "Time", "DeltaP", filename=filename)

            except:
                print("fit_all_vortices: vortex %i couldn't be fit!" % i)

            # Reset ax
            fig = None
            ax = None
        return self.popts, self.uncs

    def _determine_init_params(self, vortex, 
            init_baseline=None, init_slope=None, init_t0=None, 
            init_DeltaP=None, init_Gamma=None):
        """
        Estimate reasonable initial parameters for fitting a vortex pressure
        signal

        Args:
            vortex (dict of float arrays): vortex["time"] - time, 
            vortex["data"] - pressure
            init_* (float): initial parameters

        Returns:
            float array of initial parameter values

        """

        x = vortex["time"]
        y = vortex["data"]

        # Initial fit to background trend
        fit_params = np.polyfit(x, y, 1)
        detrended_y = y - np.polyval(fit_params, x)

        if(init_baseline is None):
            init_baseline = np.median(y)

        if(init_slope is None):
            init_slope = (y[-1] - y[0])/(x[-1] - x[0])

        if(init_t0 is None):
            init_t0 = x[np.argmin(detrended_y)]

        if(init_DeltaP is None):
            init_DeltaP = np.max(detrended_y) - np.min(detrended_y) 

        if(init_Gamma is None):
            init_Gamma = 5.*self.sampling

        return np.array([init_baseline, init_slope, init_t0, init_DeltaP, 
            init_Gamma])

    def _determine_bounds(self, vortex, init_params,
            slope_fac=10., Gamma_fac=10.):
        """
        Estimate reasonable bounds on fit parameters

        Args:
            vortex (dict of float arrays): vortex["time"] - time,
            vortex["data"] - pressure
            init_params (float array): initial parameters in following order:
                init_baseline, init_slope, init_t0, init_DeltaP, init_Gamma
            slope_fac (float): maximum factor for slope upper bound
            Gamma_fac (float): maximum factor for Gamma upper bound
    
        Returns:
            float array with lower and upper bounds on fit parameters

        """

        x = vortex["time"]
        y = vortex["data"]

       # Initial fit to background trend
        fit_params = np.polyfit(x, y, 1)
        detrended_y = y - np.polyval(fit_params, x)

        # Baseline probably doesn't exceed minimum or maximum y
        mn_baseline = np.min(y)
        mx_baseline = np.max(y)

        # Slope unlikely to exceed overall slope
        overall_slope = (y[-1] - y[0])/(x[-1] - x[0])
        mn_slope = -slope_fac*np.abs(overall_slope)
        mx_slope = slope_fac*np.abs(overall_slope)

        mn_t0 = np.min(x)
        mx_t0 = np.max(x)

        # Can't have negative delta P's
        mn_deltaP = 0.
        mx_deltaP = np.max(detrended_y) - np.min(detrended_y)

        mn_Gamma = 2.*self.sampling # Nyquist sampling
        mx_Gamma = np.max([Gamma_fac*init_params[4], x[-1] - x[0]])

        return ([mn_baseline, mn_slope, mn_t0, mn_deltaP, mn_Gamma],
                [mx_baseline, mx_slope, mx_t0, mx_deltaP, mx_Gamma])

    def make_conditioned_data_figure(self, which_vortex=0, 
            fig=None, figsize=(10, 10), aspect_ratio=16./9,
            pressure_units="Pa", time_units="Hours", vortex_time_units="s",
            write_filename=None):
        """
        Make figure showing the data conditioning and analysis process -
        like Figure 1 of Jackson et al. (2021)

        Args:
            which_vortex (int): which vortex within vortices to plot
            fig (matplotlib figure obj, optional): the figure object to use
            figsize (2x1 list, optional): inches x inches figure size
            aspect_ratio (float, optional): figure aspect ratio
            pressure/time/vortex_time_units (str, optional): units to label axes
            write_filename (str, optional): filename stem to use for writing out

        Returns:
            figure and all axes

        """

        # Boise State official colors in hex
        # boisestate.edu/communicationsandmarketing/brand-standards/colors/
        BoiseState_blue = "#0033A0"
        BoiseState_orange = "#D64309"

        if(self.peak_indices is None):
            raise ValueError("Run find_vortices first!")
        if(self.popts is None):
            raise ValueError("Run fit_all_vortices first!")

        if(fig is None):
            fig = plt.figure(figsize=(figsize[0]*aspect_ratio, figsize[1]))

        # Add axes
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(223, sharex=ax1)
        ax3 = fig.add_subplot(222)
        ax4 = fig.add_subplot(224)

        ### Raw data ###
        ax1.plot(self.time, self.data, 
                marker='.', ls='', color=BoiseState_blue)
        ax1.text(0.05, 0.8, "(a)", fontsize=48, transform=ax1.transAxes)
        ax1.grid(True)
        ax1.tick_params(labelsize=24, labelbottom=False)

        if(write_filename is not None):
            filename = write_filename + "panel_a.csv"
            utils.write_out_plot_data(self.time, self.data, 
                    "Time", "Pressure", filename=filename)

        ### Filtered data ###
        ax2.plot(self.time, self.detrended_data, 
                marker='.', ls='', color=BoiseState_blue)
        ax2.text(0.05, 0.05, "(b)", fontsize=48, transform=ax2.transAxes)
        ax2.grid(True)
        ax2.tick_params(labelsize=24)
        ax2.set_xlabel("Time (%s)" % (time_units), fontsize=36)
        ax2.set_ylabel(r'$\Delta P\,\left( {\rm %s} \right)$' %\
                (pressure_units), fontsize=36)

        if(write_filename is not None):
            filename = write_filename + "panel_b.csv"
            utils.write_out_plot_data(self.time, self.detrended_data,
                    "Time", "Detrended_Pressure", filename=filename)


        ### Convolution ###
        ax3.plot(self.time, self.convolution, 
                color=BoiseState_blue, ls='', marker='.')
        ax3.text(0.05, 0.8, "(c)", fontsize=48, transform=ax3.transAxes)
        ax3.grid(True)
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax3.tick_params(labelsize=24, labelleft=False, labelright=True)
        ax3.set_ylabel(r'$\left( F \ast \Delta P \right)$', fontsize=36)

        if(write_filename is not None):
            filename = write_filename + "panel_c.csv"
            utils.write_out_plot_data(self.time, self.convolution,
                    "Time", "Convolution", filename=filename)


        # Add lines in all plots highlighting the detections
        for cur_ex in self.peak_indices:
            ax1.axvline(self.time[cur_ex], 
                    color=BoiseState_orange, zorder=-1, ls='--', lw=3)
            ax2.axvline(self.time[cur_ex], 
                    color=BoiseState_orange, zorder=-1, ls='--', lw=3)
            ax3.axvline(self.time[cur_ex], 
                    color=BoiseState_orange, zorder=-1, ls='--', lw=3)


        ### Fit vortex ###
        model_func = lambda time, popt:\
                utils.modified_lorentzian(time, *popt) - popt[0]

        # Remember! Times are in hours!
        time = self.vortices[which_vortex]["time"]
        ydata = self.vortices[which_vortex]["data"] -\
                self.popts[which_vortex][0]
        yerr = self.vortices[which_vortex]["scatter"]
        model = utils.plot_vortex(time, self.popts[which_vortex][2], ydata, 
                model_func, self.popts[which_vortex], ax4, yerr=yerr)

        ax4.text(0.05, 0.05, "(d)", fontsize=48, transform=ax4.transAxes)
        ax4.grid(True)
        ax4.yaxis.set_label_position("right")
        ax4.yaxis.tick_right()
        ax4.tick_params(labelsize=24, labelleft=False, labelright=True)

        ax4.set_xlabel(r'$t - t_0\,\left( {\rm %s} \right)$' %\
                (vortex_time_units), fontsize=36)
        ax4.set_ylabel(r'$\Delta P\,\left( {\rm %s} \right)$' %\
                (pressure_units), fontsize=36)

        if(write_filename is not None):
            filename = write_filename + "panel_d_data.csv"
            utils.write_out_plot_data(time, ydata, "Time", "DeltaP", 
                    yerr=yerr, filename=filename)

            filename = write_filename + "panel_d_model.csv"
            utils.write_out_plot_data(time, model, "Time", "DeltaP", 
                    filename=filename)

        return fig, ax1, ax2, ax3, ax4

    def pressure_timeseries_injection_recovery(self,
        mn_deltaP=0.1, mx_deltaP=10., num_deltaPs=10,
        mn_Gamma=2.0, mx_Gamma=200., num_Gammas=10,
        num_trials=10,
        num_fwhms=3.):
        """
        Performs an injection/recovery for vortex signal within a given raw pressure time-series

        Args:
            mn/mx (floats, optional): min/max values for deltaP and Gamma arrays
            num (int, optional): number of points in the deltaP/Gamma arrays
            num_trials (int, optional): number of trials at each deltaP/Gamma
            num_fwhms (float, optional): number of FWHMs for matched_filter

        Returns:
            deltaP, Gamma, and detection fraction arrays (float arrays)
        """

        deltaP_grid = 10.**(np.linspace(np.log10(mn_deltaP), np.log10(mx_deltaP), num_deltaPs))
        Gamma_grid = 10.**(np.linspace(np.log10(mn_Gamma), np.log10(mx_Gamma), num_Gammas))/3600. # convert to hours

        # Whether or not a detection was made
        detection_statistics =\
                np.zeros([num_deltaPs, num_Gammas, num_trials])
        for i in range(num_deltaPs):
            for j in range(num_Gammas):
                for k in range(num_trials):

                    # Generate vortex at random time
                    correct_t0 = choice(self.time)
                    vortex_signal = utils.modified_lorentzian(self.time, 0., 
                            0., correct_t0, deltaP_grid[i], Gamma_grid[j])

                    # Inject into flipped detrended time-series
                    synthetic_time_series = self.detrended_data +\
                            vortex_signal + self.data_trend

                    # Make new object
                    new_mt = MetTimeseries(self.time, synthetic_time_series)
                    new_mt.detrend_timeseries_boxcar(self.window_width)

                    new_mt.apply_lorentzian_matched_filter(
                            self.matched_filter_fwhm, 
                            self.matched_filter_depth, num_fwhms=self.num_fwhms)

                    new_mt.find_vortices(detection_threshold=\
                            self.detection_threshold)

                    # If find_vortices found the vortex
                    if(np.any(np.abs(new_mt.time[new_mt.peak_indices] -\
                            correct_t0) < Gamma_grid[j])):
                        detection_statistics[i,j,k] = 1.
        detection_statistics = np.mean(detection_statistics, axis=2)

        # Convert Gamma back to seconds
        return deltaP_grid, Gamma_grid*3600., detection_statistics

class TemperatureTimeseries(MetTimeseries):
    def __init__(self, time, temperature_data, 
            pressure_vortex_popts=None, pressure_vortex_uncs=None):
        """
        Args:
            time (float, array): time of meteorological time-series
            temperature_data (float, array): temperature measurements
            pressure_vortex_params/uncs (list of float arrays): the best-fit 
            t0, Delta P, and FWHM values and uncertainties for a set of vortices
            detected in the pressure time-series collected contemporaneously
        """
        super().__init__(time, temperature_data)

        # The best-fit t0, Delta P, and FWHM values for associated vortices
        # detected in the contemporaneously collected pressure time-series
        self.pressure_vortex_popts = pressure_vortex_popts
        self.pressure_vortex_uncs = pressure_vortex_uncs

        # The temperature vortex signals
        self.vortices = None

    def retrieve_vortices(self, fwhm_factor=6, min_num_points=5, 
            include_scatter=True):
        """
        Retrieves the times and temperature data corresponding to vortices
        identified in pressure data

        Args:
            fwhm_factor (int, optional): when returning vortices, how wide a time window to return
            min_num_points (int, optional): minimum number of points
            include_scatter (bool, optional): whether or not to include a 
            calculated scatter

        Returns:
            list of times and temperatures for each vortex

        """
    
        if(self.pressure_vortex_popts is None):
            raise ValueError("self.pressure_vortex_popts is None!")

        self.vortices = []

        for i in range(len(self.pressure_vortex_popts)):

            # Grab all the points within the vortex signal
            ind = np.abs(self.time - self.pressure_vortex_popts[i][2]) <\
                    fwhm_factor*self.pressure_vortex_popts[i][4]/2.

            if(len(self.time[ind]) > min_num_points):
                # Use original, unfiltered data.
                #
                # Use the point-to-point scatter to estimate uncertainty.
                if(include_scatter):
                    self.vortices.append({"time": self.time[ind],
                        "data": self.data[ind], 
                        "scatter": mad(self.data[ind][1:] -\
                                self.data[ind][:-1])*\
                                np.ones_like(self.data[ind])})
                else:
                    self.vortices.append({"time": self.time[ind],
                        "data": self.data[ind]})

    def _determine_init_params(self, vortex, init_t0, init_Gamma,
            init_baseline=None, init_slope=None, init_Delta=None):
        """
        Estimate reasonable initial parameters for fitting a vortex pressure
        signal

        Args:
            vortex (dict of float arrays): vortex["time"] - time, 
            vortex["data"] - temperature
            init_* (float): initial parameters

        Returns:
            float array of initial parameter values

        """

        x = vortex["time"]
        y = vortex["data"]

        # Initial fit to background trend
        fit_params = np.polyfit(x, y, 1)
        detrended_y = y - np.polyval(fit_params, x)

        if(init_baseline is None):
            init_baseline = np.median(y)

        if(init_slope is None):
            init_slope = (y[-1] - y[0])/(x[-1] - x[0])

        if(init_Delta is None):
            init_Delta = -np.abs(np.max(detrended_y) - np.min(detrended_y))

        return np.array([init_baseline, init_slope, init_t0, init_Delta, 
            init_Gamma])

    def _determine_bounds(self, vortex, init_params,
            slope_fac=10., Gamma_fac=3.):
        """
        Estimate reasonable bounds on fit parameters

        Args:
            vortex (dict of float arrays): vortex["time"] - time,
            vortex["data"] - temperature
            init_params (float array): initial parameters in following order:
                init_baseline, init_slope, init_t0, init_DeltaP, init_Gamma
            slope_fac (float): maximum factor for slope upper bound
            Gamma_fac (float): maximum factor for Gamma upper bound
    
        Returns:
            float array with lower and upper bounds on fit parameters

        """

        x = vortex["time"]
        y = vortex["data"]

       # Initial fit to background trend
        fit_params = np.polyfit(x, y, 1)
        detrended_y = y - np.polyval(fit_params, x)

        # Baseline probably doesn't exceed minimum or maximum y
        mn_baseline = np.min(y)
        mx_baseline = np.max(y)

        # Slope unlikely to exceed overall slope
        overall_slope = (y[-1] - y[0])/(x[-1] - x[0])
        mn_slope = -slope_fac*np.abs(overall_slope)
        mx_slope = slope_fac*np.abs(overall_slope)

        # t0 between the t0 +- 0.5*Gamma_fac*Gamma from the pressure vortex
        mn_t0 = init_params[2] - 0.5*Gamma_fac*init_params[4]
        mx_t0 = init_params[2] + 0.5*Gamma_fac*init_params[4]
                
        # Can't have positive delta Ts - because I've defined profile as < 0
        mn_delta = -np.abs(np.max(detrended_y) - np.min(detrended_y))
        mx_delta = 0.

        mn_Gamma = 2.*self.sampling # Nyquist sampling
        mx_Gamma = Gamma_fac*init_params[4]

        return ([mn_baseline, mn_slope, mn_t0, mn_delta, mn_Gamma],
                [mx_baseline, mx_slope, mx_t0, mx_delta, mx_Gamma])
