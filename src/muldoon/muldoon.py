def __init__(time, pressure, windspeed=None, wind_direction=None):
    """
    Process and analyze a meteorological time-series to search for vortices

    Args:
        time (float, array): time of meteorological time-series
        pressure (float, array): pressure measurements
        windspeed (float, array, optional): wind speed measurements
        wind_direction (float, array, optional): wind velocity aziumth

    Returns:
        meteorological time-series object
    """

    self.time = time
    self.pressure = pressure

    if(windspeed is not None):
        self.windspeed = windspeed

    if(wind_direction is not None):
        self.wind_direction = wind_direction
