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

fig, axs = plt.subplots(1, figsize=(12,5), sharex=True)
cml_plot.tl.plot.line(x='time', ax=axs, label = 'TL');

cml_plot['wet'][0] = 0 
cml_plot['wet'][-1] = 0 
wet_start = np.roll(cml_plot.wet, -1) & ~cml_plot.wet
wet_end = np.roll(cml_plot.wet, 1) & ~cml_plot.wet
for wet_start_i, wet_end_i in zip(
    wet_start.data.nonzero()[0],
    wet_end.data.nonzero()[0],
):
    axs.axvspan(cml_plot.time.data[wet_start_i], cml_plot.time.data[wet_end_i], color='b', alpha=0.1)


cml_plot.baseline.plot.line(x='time', ax=axs, label ='baseline');

axs.set_title('');
axs.set_xlabel('')
axs.set_ylabel('rsd')


cmls['A'] = cmls.tl - cmls.baseline
cmls['A'] = cmls.A.where(cmls.A >= 0, 0)

cmls['R'] = pycml.processing.k_R_relation.calc_R_from_A(A=cmls.tl - cmls.baseline, L_km=cmls.length, f_GHz=cmls.frequency/1e9, pol=cmls.polarization,)


cmls['waa_schleiss'] = pycml.processing.wet_antenna.waa_schleiss_2013(
    rsl=cmls.tl,
    baseline=cmls.baseline,
    wet=cmls.wet,
    waa_max=2.2,
    delta_t=1,
    tau=15,
)

cmls['waa_leijnse'] = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
    A_obs=cmls.A,
    f_Hz=cmls.frequency,
    pol=cmls.polarization,
    L_km=cmls.length,
)

cmls['waa_pastorek'] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
    A_obs=cmls.A,
    f_Hz=cmls.frequency,
    pol=cmls.polarization,
    L_km=cmls.length,
)


cmls['A_rain_leijnse'] = cmls.tl - cmls.baseline - cmls.waa_leijnse
cmls['A_rain_leijnse'] = cmls.A_rain_leijnse.where(cmls.A_rain_leijnse >= 0, 0)

cmls['A_rain_pastorek'] = cmls.tl - cmls.baseline - cmls.waa_pastorek
cmls['A_rain_pastorek'] = cmls.A_rain_pastorek.where(cmls.A_rain_pastorek >= 0, 0)
    
fig, axs = plt.subplots(1, figsize=(18, 10), sharex=True)
plt.sca(axs)

cmls.tl.isel(channel_id=0).plot.line(x='time', figsize=(18, 4), label='TL', color='k', zorder=10)
cmls.baseline.isel(channel_id=0).plot.line(x='time', label='baseline', color='C0')

(cmls.baseline + cmls.waa_leijnse).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_Leijnse', color='C1')
(cmls.baseline + cmls.waa_pastorek).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_Pastorek', color='C2')
(cmls.baseline + cmls.waa_schleiss).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_Schleiss', color='C3')

plt.ylabel('total path attenuation in dB')
plt.legend();



cmls_R_1h = cmls.R.resample(time='1h', label='right').mean().to_dataset()

cmls_R_1h['lat_center'] = (cmls_R_1h.site_a_latitude + cmls_R_1h.site_b_latitude)/2
cmls_R_1h['lon_center'] = (cmls_R_1h.site_a_longitude + cmls_R_1h.site_b_longitude)/2

def plot_cml_lines(ds_cmls, ax):
    ax.plot(
        [ds_cmls.site_a_longitude, ds_cmls.site_b_longitude],
        [ds_cmls.site_a_latitude, ds_cmls.site_b_latitude],
        'k',
        linewidth=1,
    )


idw_interpolator = pycml.spatial.interpolator.IdwKdtreeInterpolator(
    nnear=15, 
    p=2, 
    exclude_nan=True, 
    max_distance=0.3,
)

R_grid = idw_interpolator(
    x=cmls_R_1h.lon_center, 
    y=cmls_R_1h.lat_center, 
    z=cmls_R_1h.R.isel(channel_id=1).sum(dim='time').where(cmls.isel(channel_id=1).wet_fraction < 0.3), 
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
