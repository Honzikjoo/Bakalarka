
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycomlink as pycml
import xarray as xr
import pandas as pd


data_path = pycml.io.examples.get_example_data_path()
cmls = xr.open_dataset(data_path + '/example_cml_data.nc')

cmls = cmls.isel(cml_id = [0, 10, 370])

cmls['tsl'] = cmls.tsl.where(cmls.tsl != 255.0)
cmls['rsl'] = cmls.rsl.where(cmls.rsl != -99.9)
cmls['tl'] = cmls.tsl - cmls.rsl 
cmls['tl'] = cmls.tl.interpolate_na(dim='time', method='linear', max_gap='5min')

cmls['wet'] = cmls.tl.rolling(time=60, center=True).std() > 0.8

cmls['wet_fraction'] = (cmls.wet==1).sum() / (cmls.wet==0).sum()

cmls['baseline'] = pycml.processing.baseline.baseline_constant(trsl=cmls.tl, wet=cmls.wet, n_average_last_dry=5)


fig, ax = plt.subplots(figsize=(12,3))

print('Enter start:')
start = input ()#'2018-05-13T22'
print('Enter end:')
end = input ()#'2018-05-14'
cml_plot = cmls.sel(time = slice(start, end)).isel(cml_id = 0, channel_id = 0)

cml_plot['wet'] = cml_plot.fillna(0).wet.astype(bool)



cmls['A'] = cmls.tl - cmls.baseline
cmls['A'] = cmls.A.where(cmls.A >= 0, 0)

cmls['R'] = pycml.processing.k_R_relation.calc_R_from_A(A=cmls.tl - cmls.baseline, L_km=cmls.length, f_GHz=cmls.frequency/1e9, pol=cmls.polarization,)

waa = input('Enter waa method: ')

if  waa == 'schleiss':
    cmls['waa_schleiss'] = pycml.processing.wet_antenna.waa_schleiss_2013(
    rsl=cmls.tl,
    baseline=cmls.baseline,
    wet=cmls.wet,
    waa_max=2.2,
    delta_t=1,
    tau=15,
    )

if  waa == 'leijnse':  
    cmls['waa_leijnse'] = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
    A_obs=cmls.A,
    f_Hz=cmls.frequency,
    pol=cmls.polarization,
    L_km=cmls.length,
    )

if  waa == 'pastorek':
    cmls['waa_pastorek'] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
    A_obs=cmls.A,
    f_Hz=cmls.frequency,
    pol=cmls.polarization,
    L_km=cmls.length,
)
