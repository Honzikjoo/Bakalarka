
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycomlink as pycml
import xarray as xr
import pandas as pd


def schleiss (): 
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


data_path = pycml.io.examples.get_example_data_path()
cmls = xr.open_dataset(data_path + '/example_cml_data.nc')
#cmls = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_cml_data.nc') #cteni dat
path_ref= xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_path_averaged_reference_data.nc')

# select 3 different CMLs to study




fig, ax = plt.subplots(figsize=(12,3))
#xr = read_cmlh5_file_to_xarray("filename") cteni dat - samotna funkce

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

cml ['R'] = pycml.processing.k_R_relation.calc_R_from_A(A_rain, L_km=10.0, pol='V', f_GHz=f_GHz, R_min=0) #vypocet intezity deste




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


fig, ax = plt.subplots(figsize=(12,3))

cml.trsl.sel(time=slice('2018-05-13', '2018-05-15')).plot.line(x='time', alpha=0.5)
plt.gca().set_prop_cycle(None)
cml.baseline.sel(time=slice('2018-05-13', '2018-05-15')).plot.line(x='time');
plt.gca().set_prop_cycle(None)
plt.ylabel('TRSL');
plt.show()





ds_cmls = xr.concat(cml_list, dim='cml_id')

cml['A'] = cml.trsl - cml.baseline
cml['A'] = cml.A.where(cml.A >= 0, 0)

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
