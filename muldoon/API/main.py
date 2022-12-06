from argparse import FileType
from urllib import response
from fastapi import *
import requests
from Create_Graphs import *



#create our fast api app
app = FastAPI()

#URL checker 
def checkUrl(PSurl):
    get = requests.get(PSurl)
    return get.status_code


@app.get("/PS/{sol}")
async def root(sol: str):
    sol = {"sol": sol}
    val = sol["sol"]

    PSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P01.CSV'
    PSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P02.CSV'


    if checkUrl(PSurl2) == 200:
        return create_PS_Graph(PSurl2)
    elif checkUrl(PSurl) == 200:
        return create_PS_Graph(PSurl)
    else: 
        raise HTTPException(status_code=404, detail="Item not found")  


@app.get("/PSGap/{sol}")
async def root(sol: str):
    sol = {"sol": sol}
    val = sol["sol"]

    PSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P01.CSV'
    PSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P02.CSV'


    if checkUrl(PSurl2) == 200:
        return create_PS_Graph_with_gaps(PSurl2)
    elif checkUrl(PSurl) == 200:
        return create_PS_Graph_with_gaps(PSurl)
    else: 
        raise HTTPException(status_code=404, detail="Item not found") 


@app.get("/PSDetrend/{sol}")
async def root(sol: str):
    sol = {"sol": sol}
    val = sol["sol"]

    PSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P01.CSV'
    PSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P02.CSV'


    if checkUrl(PSurl2) == 200:
        return create_PS_Graph_detrend(PSurl2)
    elif checkUrl(PSurl) == 200:
        return create_PS_Graph_detrend(PSurl)
    else: 
        raise HTTPException(status_code=404, detail="Item not found")


@app.get("/PSVortex/{sol}")
async def root(sol: str):
    sol = {"sol": sol}
    val = sol["sol"]

    PSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P01.CSV'
    PSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P02.CSV'


    if checkUrl(PSurl2) == 200:
        return create_PS_Vortex_Signals(PSurl2)
    elif checkUrl(PSurl) == 200:
        return create_PS_Vortex_Signals(PSurl)
    else: 
        raise HTTPException(status_code=404, detail="Item not found")

@app.get("/ATS/{sol}")
async def root(sol: str):
    sol = {"sol": sol}
    val = sol["sol"]

    ATSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_calibrated_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________CAL_ATS_________________P01.CSV'
    ATSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_calibrated_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________CAL_ATS__________________P02.CSV'


    if checkUrl(ATSurl) == 200:
        return create_ATS_Graph(ATSurl)
    elif checkUrl(ATSurl2) == 200:
        return create_ATS_Graph(ATSurl2)
    else: 
        raise HTTPException(status_code=404, detail="Item not found")


@app.get("/ATSDetrend/{sol}")
async def root(sol: str):
    sol = {"sol": sol}
    val = sol["sol"]

    ATSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_calibrated_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________CAL_ATS_________________P01.CSV'
    ATSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_calibrated_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________CAL_ATS__________________P02.CSV'


    if checkUrl(ATSurl) == 200:
        return create_ATS_Graph_Detrended(ATSurl)
    elif checkUrl(ATSurl2) == 200:
        return create_ATS_Graph_Detrended(ATSurl2)
    else: 
        raise HTTPException(status_code=404, detail="Item not found")   

         


@app.get("/ATS_PS/{sol}")
async def root(sol: str):
    sol = {"sol": sol}
    val = sol["sol"]    
    
    
    PSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P01.CSV'
    PSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_derived_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________DER_PS__________________P02.CSV'
    ATSurl = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_calibrated_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________CAL_ATS_________________P01.CSV'
    ATSurl2 = 'https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/data_calibrated_env/sol_0000_0089' + '/sol_00' + val + '/WE__00' + val + '___________CAL_ATS__________________P02.CSV'

    if(checkUrl(PSurl2) == 200 and checkUrl(ATSurl)) == 200:
        return create_ATS_PS_Graph(PSurl2,ATSurl)
    elif(checkUrl(PSurl) == 200 and checkUrl(ATSurl) == 200):
        return create_ATS_PS_Graph(PSurl,ATSurl)
    else: 
        raise HTTPException(status_code=404, detail="File path not found for {} or {}".format(PSurl,ATSurl))  
