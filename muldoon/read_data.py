"""
A collections of routines to read in data from various missions
"""
import numpy as np
import pandas as pd
from pds4_tools import pds4_read
import matplotlib.pyplot as plt


def read_Perseverance_PS_data(filename, sol=None, time_field='LTST',start=0,end=0):
    """
    Read in Perseverance MEDA PS data - https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/

    Args:
        filename (str): path to CSV file/PDS4 file
        sol (int, optional): which is the primary sol; if not given, will
        determine from filename
        time_field (str, optional): which time base to use. LTST or LMST
        start(int, optional): start index
        end(int, optional): end index

    Returns:
        time, pressure (float array): times and pressures, times in seconds
        since midnight of sol associated with filename
    """
    __check_file_type('PS', filename)
    time_field = __check_time_field(time_field)
    FIELD = 'PRESSURE'
    time, data= __process_data(filename, time_field=time_field, sol=sol)
    
    pressure = []

    if filename.endswith('.xml'):
        pressure = data['TABLE'][FIELD]
    else:
        pressure = data['PRESSURE'].values

    pressure=__array_slice(pressure,start,end)
    time=__array_slice(time,start,end)

    return time, pressure


def read_Perseverance_ATS_data(filename, which_ATS=1, time_field='LTST',sol=None, start=0, end=0):
    """
    Read in Perseverance MEDA ATS data - https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/

    Args:
        filename (str): path to CSV file/PDS4 file
        which_ATS (int or str): which of the five ATS sensors to read in;
        if "all", all of the ATS time series are returned
        time_field (str, optional): which time base to use. LTST or LMST

    Returns:
        time, pressure (float array): times and pressures, times in seconds
        since midnight of sol associated with filename

    """
    __check_file_type('ATS', filename)
    which_ATS = __check_ATS_field(which_ATS)
    time_field=__check_time_field(time_field)
    
    # Note: ATS measures at 2 Hz, so there will be some duplicate LTST-values!
    time, data = __process_data(filename, time_field=time_field, sol=sol)

    if which_ATS == 'ALL':
        if filename.endswith('.xml'):
            ATS1 = data['TABLE']["ATS_LOCAL_TEMP1"]
            ATS2 = data['TABLE']["ATS_LOCAL_TEMP2"]
            ATS3 = data['TABLE']["ATS_LOCAL_TEMP3"]
            ATS4 = data['TABLE']["ATS_LOCAL_TEMP4"]
            ATS5 = data['TABLE']["ATS_LOCAL_TEMP5"]
        else:
            ATS1 = pd.read_csv(filename)["ATS_LOCAL_TEMP1"].values
            ATS2 = pd.read_csv(filename)["ATS_LOCAL_TEMP2"].values
            ATS3 = pd.read_csv(filename)["ATS_LOCAL_TEMP3"].values
            ATS4 = pd.read_csv(filename)["ATS_LOCAL_TEMP4"].values
            ATS5 = pd.read_csv(filename)["ATS_LOCAL_TEMP5"].values

        ATS1 = __array_slice(ATS1,start,end)
        ATS2 = __array_slice(ATS2,start,end)
        ATS3 = __array_slice(ATS3,start,end)
        ATS4 = __array_slice(ATS4,start,end)
        ATS5 = __array_slice(ATS5,start,end)
        time=__array_slice(time,start,end)

        return time, [ATS1, ATS2, ATS3, ATS4, ATS5]

    else:
        if filename.endswith('.xml'):
            temperature = data['TABLE'][which_ATS]
        else: 
            temperature = data[which_ATS].values;

    temperature = __array_slice(temperature,start,end)
    time=__array_slice(time,start,end)
    
    return time, temperature

    
def read_Perseverance_WS_data(filename, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED',start=0,end=0):
    """
    Read in WS data - (link)

    Args:
        filename (str): path to CSV file/PDS4 file
        sol (int, optional): which is the primary sol; if not given, will
        determine from filename
        time_field (str, optional): which time base to use. LTST or LMST
        wind_field: wind field data in consideration
        start(int, optional): start index
        end(int, optional): end index

    Returns:
        time, ws1-8 (float array): times and dimensions of wind speed, times in seconds
        since midnight of sol associated with filename
    """
    __check_file_type('WS', filename)
    wind_field = __check_wind_field(wind_field)
    time_field = __check_time_field(time_field)
    time, data= __process_data(filename, time_field=time_field, sol=sol)

    if(filename.endswith('.xml')):
        wind_data_col = data['TABLE'][wind_field]
        wind_data = np.array(wind_data_col)

    else:
        wind_data = data[wind_field].values
            

    wind_data=__array_slice(wind_data,start,end)

    # Limits plots to only values less than 1000
    wind_array = []
    
    for d in wind_data:
        if d >= 1000:
            d = 0
        wind_array.append(d)
    
    wind_data = np.array(wind_array)

    time=__array_slice(time,start,end)
    return time, wind_data

def make_seconds_since_midnight(filename, time_field='LTST', sol=None, start=0, end=0):
    """
    The MEDA data provide times in the LTST field in the format "sol hour:minute:second".

    Args:
        filename (str): path to CSV file/PDS4 file
        time_field (str, optional): name of time field to analyze
        sol (int, optional): which is the primary sol; if not given, will
        determine from filename

    Returns:
        time: number of seconds in each row since midnight of the primary sol for that file
        data: processed data from file
        """
    time, _ = __process_data(filename, time_field,sol)
    time=__array_slice(time,start,end)
    return time

#################################
#        Helper Function        #
#################################
def __process_data(filename, time_field='LTST', sol=None):
    """
    The MEDA data provide times in the LTST field in the format "sol hour:minute:second".

    Args:
        filename (str): path to CSV file/PDS4 file
        time_field (str, optional): name of time field to analyze
        sol (int, optional): which is the primary sol; if not given, will
        determine from filename

    Returns:
        time: number of seconds in each row since midnight of the primary sol for that file
        data: processed data from file
        """

    if(sol is None):
        primary_sol = __which_sol(filename)
    else:
        primary_sol = sol
    data, _ = __read_data(filename)
    time_field = __check_time_field(time_field)

    sols_str = [] #Sol is a solar day on Mars
    times_str = []
    delta_sols = []

    # Grab the sols and times associated with each row
    if(time_field == "LTST"):
        if(filename.endswith('.xml')):
            time_field_col = data['TABLE'][time_field]
            ltst_values = np.array(time_field_col)
            np.array(map(str, ltst_values))
            ltst_split = np.char.split(ltst_values,sep=' ') #splits fromchar 0 to space " "

            for x in range(len(ltst_values)):
                sols_str.append(float(ltst_split[x][0]))
                times_str.append(ltst_split[x][1])
            
            for x in range(len(ltst_values)):
                delta_sols.append(sols_str[x] - float(primary_sol))
        else:
            sols_str = data[time_field].str.split(expand=True)[0].values #splits from char 0 to space " "
            times_str = data[time_field].str.split(expand=True)[1].values
            delta_sols = sols_str.astype(float) - float(primary_sol)

    elif(time_field == "LMST"):
        if(filename.endswith('.xml')):
            time_field_col = data['TABLE'][time_field]
            lmst_values = np.array(time_field_col)
            lmst_split = np.char.split(lmst_values,sep='M') #splits fromchar 0 to space " "
            
            for x in range(len(lmst_values)):
                times_str.append(lmst_split[x][1])
        else:
            times_str = data[time_field].str.split("M", expand=True)[1].values

    time = __time_to_hours_decimal(times_str, delta_sols, time_field)

    return time, data

def __time_to_hours_decimal(times_str, delta_sols,time_field):
    """
    Time coversion for easy usability

    Args:
        times_str: time string for file
        time_field (str, optional): name of time field to analyze
        delta_sol (int, optional): difference between primary sol and sol from file
    Returns:
        converted time
    """
    time_field = __check_time_field(time_field)
    if(time_field=='LTST'):
        time = np.array([float(times_str[i].split(":")[0]) +\
                delta_sols[i]*24. + float(times_str[i].split(":")[1])/60 +\
                float(times_str[i].split(":")[2])/3600. for i in
                range(len(times_str))])

    elif(time_field=='LMST'):
        time = np.array([float(times_str[i].split(":")[0]) +\
                float(times_str[i].split(":")[1])/60 +\
                float(times_str[i].split(":")[2])/3600. for i in
                range(len(times_str))])

    return time

def __which_sol(filename):
    """
    Based on the filename, returns the sol corresponding to a data file

    Args:
        filename (str): path to CSV file/PDS4 file

    Returns:
        sol (int) associated with that file
    """

    ind = filename.find("WE__")

    return int(filename[ind+4:ind+8])

def __read_data(filename:str):
    """
    Verifies the existence of provide file

    Args:
        filename (str): path to CSV file/PDS4 file
        
    Returns:
        data: processed data from provided file
        file_status: 1 if file exists, else expception is thrown
    """

    file_status = 0
    data = " "
    error_message = "\n===>Filenotfounderror: There is no file " + filename + " in the working directory. Please check file name or path\n"
    if filename.endswith('.xml'):
        try:
            data = pds4_read(filename,lazy_load=True)
            # data.info()
            file_status = 1
        except Exception as e:
            print(error_message)
            file_status = 0
    else:
        print("Processing file: " + filename)
        try:
            data = pd.read_csv(filename)
            file_status = 1
        except Exception as e:
            print(error_message)
            file_status = 0
    print('===================================================================')
     
    return data, file_status

def __check_file_type(data_requested:str, filename:str):
    """
    Checks if requested data is found in the
    input file

    Args:
        data requested: str
        filename

    Returns:
            void
    """
    error_message = "\n===>Please check file type. '" + data_requested + "' data is not contained in " + filename
    data_requested = data_requested.upper()
    if data_requested in filename:
        return
    else:
        raise Exception(error_message)


def __check_time_field(time_field:str):
    """
    Checks if time_field is a valid input
    updates wind_field to the right format

    Args:
        time_field

    Returns:
        time_field
    """
    time_field = time_field.upper()
    accepted_strings = ['LTST', 'LMST']
    if time_field in accepted_strings:
        return time_field
    else:
        print('--------------------------')
        print('Accepted time_field input:')
        for x in accepted_strings:
            print(x)
        print('--------------------------')
        raise Exception("\n===>time_field= '" + time_field + "' is not an accepted time_field input")

def __check_wind_field(wind_field:str):
    """
    Checks if wind_field is a valid input and 
    updates wind_field to the right format

    Args:
        time_field

    Returns:
        time_field
    """
    wind_field = wind_field.upper()
    accepted_strings = {
    "HWS": "HORIZONTAL_WIND_SPEED",
    "HWSU": "HORIZONTAL_WIND_SPEED_UNCERTAINTY",
    "VWS": "VERTICAL_WIND_SPEED",
    "VWSU": "VERTICAL_WIND_SPEED_UNCERTAINTY",
    "WD": "WIND_DIRECTION",
    "WDU": "WIND_DIRECTION_UNCERTAINTY",
    "BBUFR": "BOTH_BOOMS_USED_FOR_RETRIEVAL",
    "RS": "ROVER_STILL",
    }

    if wind_field in accepted_strings.keys():
        return accepted_strings[wind_field]
    elif wind_field in accepted_strings.values():
        return wind_field
    else:
        print('Please check the wind_field argument')
        print('----------------------------------------')
        print('Accepted wind_field input:\n')
        for x, y in accepted_strings.items():
            print(x + ' - ' + y)
        print('----------------------------------------')
        raise Exception("wind_field= '" + wind_field + "' is not an accepted wind_field input")
    
def __check_ATS_field(ats_field):
    """
    Checks if wind_field is a valid input and 
    updates wind_field to the right format

    Args:
        ats_field

    Returns:
        ats_field
    """
    if isinstance(ats_field, str):
        ats_field = ats_field.upper()

    accepted_strings = {
    1: "ATS_LOCAL_TEMP1",
    2: "ATS_LOCAL_TEMP2",
    3: "ATS_LOCAL_TEMP3",
    4: "ATS_LOCAL_TEMP4",
    5: "ATS_LOCAL_TEMP5",
    0: 'ALL'
    }
    if ats_field == 'all' or ats_field == 'ALL':
        return accepted_strings[0]
    elif ats_field in accepted_strings.keys():
        return accepted_strings[ats_field]
    elif ats_field in accepted_strings.values():
        return ats_field
    else:
        print('Please check the ats_field argument')
        print('----------------------------')
        print('Accepted wind_field input:\n')
        for x, y in accepted_strings.items():
            print(str(x) + ' or ' + y)
        print('----------------------------')
        raise Exception("\n===>ats_field= '" + str(ats_field) + "' is not an accepted ats_field input")
    
def __array_slice(data, start=0, end=0):
    """
    Array slicing option - helps take elements from one 
    given index to another given index.

    Args:
        data: array to slice
        start: start element
        end: end element

    Returns:
        sliced array
    """   
    if type(data) is np.ndarray:
        data=data
    else:
        data = np.array(data)

    if(start>0 and end>start):
        data=(data[start:end])
    elif((start>0 and end<start)):
        data=(data[start:])
    elif(start==0 and end > 0):
        data=(data[:end])
    return data

#################################
#           plots               #
#################################
def plot_Perseverance_ATS_data(filename, which_ATS=1, time_field='LTST', save_file=False, scatter=False, start=0,end=0):
    """
    Read in Perseverance MEDA ATS data - https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/

    Args:
        filename (str): Path to CSV file/PDS4 file (HTTP addres also valid)
        which_ATS (int or str): Which of the five ATS sensors to read in;
        if "all", all of the ATS time series are returned
        time_field (str, optional): Which time base to use (LTST/LMST; defaults to LTST)
        save_file (bool/str, optional): If True, saves a local copy as a default name, if specified string, saves local
        copy under that filename; will save in working directory
        scatter: Produces a scatter plot if True, else produces line plot
        start(int, optional): start index
        end(int, optional): end index

    Returns:
        time, temperature plot (float array): times and temperature, times in hours
        since midnight of sol associated with filename

    """
    time,result = read_Perseverance_ATS_data(filename,which_ATS,time_field,None,start=start,end=end)
    which_ATS = __check_ATS_field(which_ATS)
    plt.title(which_ATS + " vs TIME")
    plt.ylabel(which_ATS)

    plt.xlabel(time_field + " TIME")
    if(scatter==True):
        plt.scatter(time,result)
    else:
        plt.plot(time,result)

    if(save_file == True):
        save_file = 'ATS Figure 1'
        plt.savefig(save_file)
    elif(type(save_file) == str):
        plt.savefig(save_file)
    plt.show()

def plot_Perseverance_PS_data(filename, sol=None, time_field = 'LTST', start=0, end=0, scatter=False, save_file=False):
    """

    Args:
        filename (str): Path to CSV file/PDS4 file (HTTP address also valid)
        sol (int, optional): Which sol to use; will determine from file if not specified
        time_field (str, optional): Which time base to use (LTST/LMST; defaults to LTST)
        start(int, optional): Start index
        end(int, optional): End index
        scatter (bool, optional): Produces a scatter plot if True, else produces line plot
        save_file (bool/str, optional): If True, saves a local copy as a default name, if specified string, saves local
        copy under that filename; will save in working directory

    Returns:
        time, pressure plot: Times and pressure, time is in hours
        since midnight of sol associated with filename

    """
    time, pressure = read_Perseverance_PS_data(filename, sol, time_field, start = start, end = end)

    # Equivalent of ATS and ATS handling/what to do with?
    # Then just label and process?
    plt.title("PRESSURE versus TIME")
    plt.xlabel("TIME")
    plt.ylabel("PRESSURE")
    
    if (scatter):
        plt.scatter(time, pressure)
    else:
        plt.plot(time, pressure)

    # Ensure rendering
    if (save_file == True):
        save_file = 'Pressure Figure 1'
        plt.savefig(save_file)
    elif (type(save_file) == str):
        plt.savefig(save_file)

    plt.show()

def plot_Perseverance_WS_data(filename, sol=None, time_field='LTST', wind_field='HORIZONTAL_WIND_SPEED',start=0,end=0, scatter=False, save_file=False):
    """

    Args:
        filename (str): Path to CSV file/PDS4 file (HTTP address also valid)
        sol (int, optional): Which sol to use; will determine from file if not specified
        time_field (str, optional): Which time base to use (LTST/LMST; defaults to LTST)
        start(int, optional): Start index
        end(int, optional): End index
        scatter (bool, optional): Produces a scatter plot if True, else produces line plot
        save_file (bool/str, optional): If True, saves a local copy as a default name, if specified string, saves local
        copy under that filename; will save in working directory

    Returns:
        time, wind plot: Times and wind, time is in hours
        since midnight of sol associated with filename

    """
    time, wind = read_Perseverance_WS_data(filename=filename, sol=sol, time_field=time_field, wind_field=wind_field, start=start, end=end)

    plt.title(wind_field +" versus TIME")
    plt.xlabel("TIME")
    plt.ylabel(wind_field)
    
    if (scatter):
        plt.scatter(time, wind)
    else:
        plt.plot(time, wind)

    # Ensure rendering
    if (save_file == True):
        save_file = 'Wind Figure 1'
        plt.savefig(save_file)
    elif (type(save_file) == str):
        plt.savefig(save_file)

    plt.show()
