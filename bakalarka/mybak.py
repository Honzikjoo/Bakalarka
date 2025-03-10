
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycomlink as pycml
import xarray as xr
import pandas as pd


def Schleiss (): 
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

def waa():
    cml['waa'] = pycml.processing.wet_antenna.waa_schleiss_2013(
    rsl=cml.trsl, 
    baseline=cml.baseline, 
    wet=cml.wet, 
    waa_max=2.2, 
    delta_t=1, 
    tau=15,
)

def baseline():
    cml['baseline'] = pycml.processing.baseline.baseline_constant(
    trsl=cml.tl, 
    wet=cml.wet, 
    n_average_last_dry=5,
)


data_path = pycml.io.examples.get_example_data_path()
cmls = xr.open_dataset(data_path + '/example_cml_data.nc')
#cmls = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_cml_data.nc') #cteni dat
#path_ref= xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_path_averaged_reference_data.nc')

#xr = read_cmlh5_file_to_xarray("filename") cteni dat - samotna funkce


cml_list = [cmls.isel(cml_id=i) for i in range(len(cmls.cml_id))]

for cml in cml_list:
    cml['tsl'] = cml.tsl.where(cml.tsl != 255.0)
    cml['rsl'] = cml.rsl.where(cml.rsl != -99.9)
    cml['trsl'] = cml.tsl - cml.rsl

for cml in cml_list:
    cml['trsl'] = cml.trsl.interpolate_na(dim='time', method='linear', max_gap='5min')

cml = cml_list[0].copy()

cml['wet'] = cml.trsl.rolling(time=60, center=True).std(skipna=False) > 0.8

cml['baseline'] = pycml.processing.baseline.baseline_constant(trsl=cml.trsl, wet=cml.wet, n_average_last_dry=5)




cmls = cmls.isel(cml_id = [0, 10, 370])

cmls['tsl'] = cmls.tsl.where(cmls.tsl != 255.0)
cmls['rsl'] = cmls.rsl.where(cmls.rsl != -99.9)
cmls['tl'] = cmls.tsl - cmls.rsl 
cmls['tl'] = cmls.tl.interpolate_na(dim='time', method='linear', max_gap='5min')


cmls['wet_rsd'] = cmls.tl.rolling(time=60, center=True).std() > 0.8

cmls['baseline_rsd'] = pycml.processing.baseline.baseline_constant(trsl=cmls.tl, wet=cmls.wet_rsd, n_average_last_dry=5)




fig, ax = plt.subplots(figsize=(12,3))

start = '2018-05-13T22'
end = '2018-05-14'
cml_plot = cmls.sel(time = slice(start, end)).isel(cml_id = 0, channel_id = 0)

cml_plot['wet_rsd'] = cml_plot.fillna(0).wet_rsd.astype(bool)

fig, axs = plt.subplots(figsize=(12,5))

cml_plot.tl.plot.line(x='time', ax=axs, label = 'TL');

cml_plot['wet_rsd'][0] = 0 
cml_plot['wet_rsd'][-1] = 0 
wet_start = np.roll(cml_plot.wet_rsd, -1) & ~cml_plot.wet_rsd
wet_end = np.roll(cml_plot.wet_rsd, 1) & ~cml_plot.wet_rsd
for wet_start_i, wet_end_i in zip(
    wet_start.data.nonzero()[0],
    wet_end.data.nonzero()[0],
):
    axs.axvspan(cml_plot.time.data[wet_start_i], cml_plot.time.data[wet_end_i], color='b', alpha=0.1)

    cml_plot.baseline_rsd.plot.line(x='time', ax=axs, label ='baseline');

axs.set_title('');
axs.set_xlabel('')

axs.set_ylabel('rsd')

axs.legend(loc = 'upper right')
plt.show()



start = '2018-05-15T22'
end = '2018-05-16T12'
cml_plot = cmls.sel(time = slice(start, end)).isel(cml_id = 2, channel_id = 0)

cml_plot['wet_rsd'] = cml_plot.fillna(0).wet_rsd.astype(bool)

fig, axs = plt.subplots(figsize=(12,5))

cml_plot.tl.plot.line(x='time', ax=axs, label = 'TL');

cml_plot['wet_rsd'][0] = 0 
cml_plot['wet_rsd'][-1] = 0 
wet_start = np.roll(cml_plot.wet_rsd, -1) & ~cml_plot.wet_rsd
wet_end = np.roll(cml_plot.wet_rsd, 1) & ~cml_plot.wet_rsd
for wet_start_i, wet_end_i in zip(
    wet_start.data.nonzero()[0],
    wet_end.data.nonzero()[0],
):
    axs.axvspan(cml_plot.time.data[wet_start_i], cml_plot.time.data[wet_end_i], color='b', alpha=0.1)

cml_plot.baseline_rsd.plot.line(x='time', ax=axs, label ='baseline');

axs.set_title('');
axs.set_xlabel('')

axs.set_ylabel('rsd')

axs.legend(loc = 'upper right')
plt.show()





cml['A'] = cml.trsl - cml.baseline
cml['A'] = cml.A.where(cml.A >= 0, 0)

#Schleiss
#Pastorek
#Leijnse

cml['waa_leijnse'] = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
    A_obs=cml.A,
    f_Hz=cml.frequency,
    pol=cml.polarization,
    L_km=cml.length,
)

cml['waa_pastorek'] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
    A_obs=cml.A,
    f_Hz=cml.frequency,
    pol=cml.polarization,
    L_km=cml.length,
    A_max=5,
)

cml['waa_schleiss'] = pycml.processing.wet_antenna.waa_schleiss_2013(
    rsl=cml.trsl,
    baseline=cml.baseline,
    wet=cml.wet,
    waa_max=2.2,
    delta_t=1,
    tau=15,
)



for waa_method in ['leijnse', 'pastorek', 'schleiss']:
    cml[f'A_rain_{waa_method}'] = cml.trsl - cml.baseline - cml[f'waa_{waa_method}']
    cml[f'A_rain_{waa_method}'] = cml[f'A_rain_{waa_method}'].where(cml[f'A_rain_{waa_method}'] >= 0, 0)
    cml[f'R_{waa_method}'] = pycml.processing.k_R_relation.calc_R_from_A(
        A=cml[f'A_rain_{waa_method}'], L_km=float(cml.length), f_GHz=cml.frequency/1e9, pol=cml.polarization
    )
cml['R'] = pycml.processing.k_R_relation.calc_R_from_A(
        A=cml.trsl - cml.baseline, L_km=float(cml.length), f_GHz=cml.frequency/1e9, pol=cml.polarization,)





fig, axs = plt.subplots(2, 1, figsize=(12,5), sharex=True)

plt.sca(axs[0])
cml.isel(channel_id=0).trsl.plot.line(x='time', alpha=0.5, label='TRSL')
plt.gca().set_prop_cycle(None)
cml.isel(channel_id=0).baseline.plot.line(x='time', linestyle=':', label='baseline without WAA');
(cml.baseline + cml.waa_schleiss).isel(channel_id=0).plot.line(x='time', label='baseline with WAA schleiss');
plt.ylabel('TRSL (dB)')
(cml.baseline + cml.waa_leijnse).isel(channel_id=0).plot.line(x='time', label='baseline with WAA leijnse');
plt.ylabel('TRSL (dB)')
(cml.baseline + cml.waa_pastorek).isel(channel_id=0).plot.line(x='time', label='baseline with WAA pastorek');
plt.ylabel('TRSL (dB)')
axs[0].legend()

cml['A'] = cml.trsl - cml.baseline - cml.waa_schleiss
cml['A'].values[cml.A < 0] = 0
cml['B'] = cml.trsl - cml.baseline - cml.waa_leijnse
cml['B'].values[cml.B < 0] = 0
cml['C'] = cml.trsl - cml.baseline - cml.waa_pastorek
cml['C'].values[cml.C < 0] = 0
cml['A_no_waa_correct'] = cml.trsl - cml.baseline
cml['A_no_waa_correct'].values[cml.A_no_waa_correct < 0] = 0 

plt.sca(axs[1])
cml.A_no_waa_correct.isel(channel_id=0).plot.line(x='time', linestyle=':', label='without WAA');
plt.gca().set_prop_cycle(None)
cml.A.isel(channel_id=0).plot.line(x='time', label='with WAA schleiss');
cml.B.isel(channel_id=0).plot.line(x='time', label='with WAA leijnse');
cml.C.isel(channel_id=0).plot.line(x='time', label='with WAA pastorek');
plt.ylabel('path attenuation\nfrom rain (dB)');
axs[1].set_title('');
axs[1].legend()

axs[1].set_xlim(pd.to_datetime('2018-05-16 23:00:00'), pd.to_datetime('2018-05-17 06:00:00'));





ds_cmls = xr.concat(cml_list, dim='cml_id')

cml['A'] = cml.trsl - cml.baseline
cml['A'] = cml.A.where(cml.A >= 0, 0)
cml['A_no_waa_correct'] = cml.trsl - cml.baseline
cml['A_no_waa_correct'].values[cml.A_no_waa_correct < 0] = 0 

cmls['R'] = pycml.processing.k_R_relation.calc_R_from_A(A=cml.A, L_km=float(cml.length), f_GHz=cml.frequency/1e9, pol=cml.polarization)

cmls_R_1h = ds_cmls.R.resample(time='1h', label='right').mean().to_dataset()

cmls_R_1h['lat_center'] = (cmls_R_1h.site_a_latitude + cmls_R_1h.site_b_latitude)/2
cmls_R_1h['lon_center'] = (cmls_R_1h.site_a_longitude + cmls_R_1h.site_b_longitude)/2

idw_interpolator = pycml.spatial.interpolator.IdwKdtreeInterpolator(
    nnear=15, 
    p=2, 
    exclude_nan=True, 
    max_distance=0.3,
)

def plot_cml_lines(ds_cmls, ax):
    ax.plot(
        [ds_cmls.site_a_longitude, ds_cmls.site_b_longitude],
        [ds_cmls.site_a_latitude, ds_cmls.site_b_latitude],
        'k',
        linewidth=1,
    )

R_grid = idw_interpolator(
    x=cmls_R_1h.lon_center, 
    y=cmls_R_1h.lat_center, 
    z=cmls_R_1h.R.isel(channel_id=1).sum(dim='time').where(ds_cmls.wet_fraction < 0.3), 
    resolution=0.01,
)

bounds = np.arange(0, 80, 5.0)
bounds[0] = 1
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')

fig, ax = plt.subplots(figsize=(8, 6))
pc = plt.pcolormesh(
    idw_interpolator.xgrid, 
    idw_interpolator.ygrid, 
    R_grid, 
    shading='nearest', 
    cmap='turbo',
    norm=norm,
)
plot_cml_lines(cmls_R_1h, ax=ax)
fig.colorbar(pc, label='Rainfall sum in mm');
plt.show()
