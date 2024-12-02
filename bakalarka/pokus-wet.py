import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pycomlink as pycml


cmls = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_cml_data.nc') #cteni dat
radar_along_cml = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_path_averaged_reference_data.nc')

# select 3 different CMLs to study
cmls = cmls.isel(cml_id = [0, 10, 370])

# Remove outliers, compute tl and interpolate missing values
cmls['tsl'] = cmls.tsl.where(cmls.tsl != 255.0)
cmls['rsl'] = cmls.rsl.where(cmls.rsl != -99.9)
cmls['tl'] = cmls.tsl - cmls.rsl # calculate total loss (previous TRSL)
cmls['tl'] = cmls.tl.interpolate_na(dim='time', method='linear', max_gap='5min')

# 1. wet dry detection using rsd
cmls['wet_rsd'] = cmls.tl.rolling(time=60, center=True).std() > 0.8

    # Determine baseline RSD
cmls['baseline_rsd'] = pycml.processing.baseline.baseline_constant(
    trsl=cmls.tl, 
    wet=cmls.wet_rsd, 
    n_average_last_dry=5
)


start = '2018-05-13T22'
end = '2018-05-14'
cml_plot = cmls.sel(time = slice(start, end)).isel(cml_id = 0, channel_id = 0)

# convert float to bool for plotting the shaded areas
cml_plot['wet_rsd'] = cml_plot.fillna(0).wet_rsd.astype(bool)

fig, axs = plt.subplots(figsize=(12,5))

#fig, axs = plt.subplots(1, 1, figsize=(12,5), sharex=True)
cml_plot.tl.plot.line(x='time', ax=axs, label = 'TL');

# shaded rsd
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

# convert float to bool for plotting the shaded areas
cml_plot['wet_rsd'] = cml_plot.fillna(0).wet_rsd.astype(bool)

fig, axs = plt.subplots(figsize=(12,5))
#fig, axs = plt.subplots(3, 1, figsize=(12,5), sharex=True)
cml_plot.tl.plot.line(x='time', ax=axs, label = 'TL');

# shaded rsd
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
