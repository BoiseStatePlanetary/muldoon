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


def fit_vortex(vortex, init_params, bounds, rescale_uncertainties=True):
    """
    Fits modified Lorentzian to pressure profile

    Args:
        vortex (dict of float arrays): ["time"] - times, ["pressure"] - pressures
        init_params (float array): initial values including baseline, slope, initial central time, initial delta P, and initial duration
        bounds (float array): bounds on fit parameters listed in the same order as in init_params
        rescale_uncertainties (bool, optional): whether to rescale uncertainties on fit parameters by sqrt(reduced chi-squared)

    Returns:
        fit parameters (float array) and uncertainties (float array)

    """

    x = vortex["time"]
    y = vortex["pressure"]

    popt, pcov = curve_fit(modified_lorentzian, x, y, p0=init_params,
            bounds=bounds)
    ymod = modified_lorentzian(x, *popt)

    if(rescale_uncertainties):
        sd = mad(y - ymod)
        red_chisq = redchisqg(y, ymod, deg=5, sd=sd)

        pcov *= np.sqrt(red_chisq)

    return popt, np.sqrt(np.diag(pcov))

def determine_init_params(vortex,
                          init_baseline=None, init_slope=None, init_t0=None, init_DeltaP=None, init_Gamma=None):

    x, y = condition_vortex(vortex)
    fit_params = np.polyfit(x, y, 1)
    detrended_y = y - np.polyval(fit_params, x)
   
    if(init_baseline is None):
        init_baseline = np.median(y)

    if(init_slope is None):
        init_slope = fit_params[0]

    if(init_t0 is None):
        init_t0 = x[np.argmin(detrended_y)]

    if(init_DeltaP is None):
        init_DeltaP = 10.

    if(init_Gamma is None):
        init_Gamma = 2./3600.

    return np.array([init_baseline, init_slope, init_t0, init_DeltaP, init_Gamma])

