#%matplotlib inline
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pycomlink as pycml
import xarray as xr

def schleiss(): 
    cml['waa_schleiss'] = pycml.processing.wet_antenna.waa_schleiss_2013( #waa metoda schleiss
    rsl=cml.trsl,
    baseline=cml.baseline,
    wet=cml.wet,
    waa_max=2.2,
    delta_t=1,
    tau=15,
)
    

def Pastorek():
    cml['waa_pastorek'] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs( #waa metoda pastorek
    A_obs=cml.A,
    f_Hz=cml.frequency * 1e9,
    pol=cml.polarization,
    L_km=cml.length,
)
    
def Leijnse():
    cml['waa_leijnse'] = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs( #waa metoda leijnse
    A_obs=cml.A,
    f_Hz=cml.frequency * 1e9,
    pol=cml.polarization,
    L_km=cml.length,
)





fig, ax = plt.subplots()

cmls = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_cml_data.nc') #cteni dat
radar_along_cml = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_path_averaged_reference_data.nc')

cml_list = pycml.io.examples.get_75_cmls()

cml = cml_list[0]

cml['tsl'] = cml.tsl.where(cml.tsl != 255.0)
cml['rsl'] = cml.rsl.where(cml.rsl != -99.9)
cml['trsl'] = cml.tsl - cml.rsl

cml['wet'] = cml.trsl.rolling(time=60, center=True).std(skipna=False) > 0.8

cml['wet_fraction'] = (cml.wet==1).sum() / (cml.wet==0).sum()

cml['baseline'] = pycml.processing.baseline.baseline_constant( #referencni hodnota, dalsi funkce je baseline_linear
    trsl=cml.trsl, 
    wet=cml.wet, 
    n_average_last_dry=5,
)

cml['A'] = cml.trsl - cml.baseline
cml['A'] = cml.A.where(cml.A >= 0, 0)

f_GHz = 23.0

A_rain = np.logspace(-11, 1, 1000)
A_rain[0] = 0

R = pycml.processing.k_R_relation.calc_R_from_A(A_rain, L_km=10.0, pol='V', f_GHz=f_GHz, R_min=0) #vypocet intezity deste

interpolator1 = pycml.spatial.interpolator.IdwKdtreeInterpolator( #interpoluje hodnoty zjistene pomoci waa metod
    nnear=15, 
    p=2, 
    exclude_nan=True, 
    max_distance=0.3,
)

interpolator2 = pycml.spatial.interpolator.OrdinaryKrigingInterpolator #druhy zpusob interpolovani


