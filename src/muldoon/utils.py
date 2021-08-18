"""
Utility functions for muldoon
"""

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
