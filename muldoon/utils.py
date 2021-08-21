"""
Utility functions for muldoon
"""

import numpy as np
from scipy.optimize import curve_fit
from statsmodels.robust import mad

def modified_lorentzian(t, baseline, slope, t0, DeltaP, Gamma):
    """
    Pressure profile for a vortex

    Args:
        t (float array): time
        baseline (float): pressure baseline against which vortex excursion
        occurs
        slope (float): slope against which excursion occurs
        t0 (float): central time for vortex excursion
        DeltaP (float): depth of pressure excursion
        Gamma (float): full-width/half-max duration of excursion

    Returns:
        Pressure excursion for a vortex (float array)

    """
    # Equation 7 from Kahapaa+ (2016)
    return baseline + slope*(t - t0) - DeltaP/(((t - t0)/(Gamma/2.))**2 + 1)

def redchisqg(ydata,ymod,deg=2,sd=None):
    """
    Returns the reduced chi-square error statistic for an arbitrary model,
    chisq/nu, where nu is the number of degrees of freedom. If individual
    standard deviations (array sd) are supplied, then the chi-square error
    statistic is computed as the sum of squared errors divided by the standard
    deviations. See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    ydata,ymod,sd assumed to be Numpy arrays. deg integer.

    Usage:
    chisq=redchisqg(ydata,ymod,n,sd)

    where
    ydata : data
    ymod : model evaluated at the same x points as ydata
    n : number of free parameters in the model
    sd : uncertainties in ydata

    Rodrigo Nemmen
    http://goo.gl/8S1Oo
    """
    # Chi-square statistic
    if(np.any(sd == None)):
        chisq=np.sum((ydata-ymod)**2)
    else:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )

    # Number of degrees of freedom assuming 2 free parameters
    nu=ydata.size - 1. - deg

    return chisq/nu


def fit_vortex(vortex, init_params, bounds, sigma=None, 
        rescale_uncertainties=True):
    """
    Fits modified Lorentzian to pressure profile

    Args:
        vortex (dict of float arrays): ["time"] - times, ["pressure"] - pressures
        init_params (float array): initial values including baseline, slope, initial central time, initial delta P, and initial duration
        bounds (float array): bounds on fit parameters listed in the same order as in init_params
        sigma (float, float array, optional): per-point uncertainties
        rescale_uncertainties (bool, optional): whether to rescale uncertainties on fit parameters by sqrt(reduced chi-squared)

    Returns:
        fit parameters (float array) and uncertainties (float array)

    """

    x = vortex["time"]
    y = vortex["pressure"]

    if(sigma is not None):
        popt, pcov = curve_fit(modified_lorentzian, x, y, p0=init_params,
                bounds=bounds, sigma=sigma)
    else:
        popt, pcov = curve_fit(modified_lorentzian, x, y, p0=init_params,
                bounds=bounds)
    ymod = modified_lorentzian(x, *popt)

    if(rescale_uncertainties):
        if(sigma is None):
            sd = mad(y - ymod)
        else:
            sd = sigma
        red_chisq = redchisqg(y, ymod, deg=5, sd=sd)

        pcov *= np.sqrt(red_chisq)

    return popt, np.sqrt(np.diag(pcov))

def write_out_plot_data(x, y, x_label, y_label, 
        xerr=None, yerr=None, filename="out.csv", mode="w", test_mode=False):
    """
    Write out formatted text file of plot data

    Args:
        x/y (float array): x/y points to write out
        x_label/y_label (str): labels for columns
        xerr/yerr (float array, optional): associated uncertainties
        filename (str, optional): path of file to which to write out data
        mode (str, optional): write mode; defaults to over-write
        test_mode (bool, optional): whether to actually write out file

    """

    # Construct write string
    write_str = "# %s, %s" % (x_label, y_label)

    if(xerr is not None):
        write_str += ", %s_err" % x_label
    if(yerr is not None):
        write_str += ", %s_err" % y_label

    write_str += "\n"

    for i in range(len(x) - 1):
        write_str += "%g, %g" % (x[i], y[i])

        if(xerr is not None):
            write_str += ", %g" % (xerr[i])
        if(yerr is not None):
            write_str += ", %g" % (yerr[i])

        write_str += "\n"

    # Don't write a new-line character for the last entry
    write_str += "%g, %g" % (x[-1], y[-1])
    if(xerr is not None):
        write_str += ", %g" % (xerr[i])
    if(yerr is not None):
        write_str += ", %g" % (yerr[i])

    if(~test_mode):
        f = open(filename, mode)
        f.write(write_str)
        f.close()

    return write_str

