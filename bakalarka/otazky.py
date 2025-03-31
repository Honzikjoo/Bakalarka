import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycomlink as pycml
import xarray as xr
import pandas as pd

#nacitani dat
data_path = pycml.io.examples.get_example_data_path()
radar_along_cml = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_path_averaged_reference_data.nc')
cmls = xr.open_dataset(data_path + '/example_cml_data.nc')
path_ref = xr.open_dataset(data_path + '/example_path_averaged_reference_data.nc')


#dani dat z cmls do cml ? jaky je rozdil ?
cml_list = [cmls.isel(cml_id=i) for i in range(len(cmls.cml_id))]
cml = cml_list[0]

cmls = cmls.isel(cml_id = [0, 10, 370])

cmls['tsl'] = cmls.tsl.where(cmls.tsl != 255.0) #stanoveni mezi ?
cmls['rsl'] = cmls.rsl.where(cmls.rsl != -99.9) #stanoveni mezi ?
cmls['tl'] = cmls.tsl - cmls.rsl #vypocet celkovych ztrat
cmls['tl'] = cmls.tl.interpolate_na(dim='time', method='linear', max_gap='5min') #interpolace neznamych hodnot po 5 minutach

cmls['wet'] = cmls.tl.rolling(time=60, center=True).std() > 0.8 #zakladni rozpoznani mezi wet/dry obdobim

cmls['baseline'] = pycml.processing.baseline.baseline_constant(trsl=cmls.tl, wet=cmls.wet, n_average_last_dry=5)#stanoveni baseline od ktere se vypocty odviji


print('Enter start:')
start = input ()#'2018-05-13T22'
print('Enter end:')#stanoveni zacatku a konce dat pro ktere dane mereni delame
end = input ()#'2018-05-14'
cml_plot = cmls.sel(time = slice(start, end)).isel(cml_id = 0, channel_id = 0)#pomoc pri vzkreslovani grafu ?

cml_plot['wet'] = cml_plot.fillna(0).wet.astype(bool)#zmena floatu na bool kvuli lepsimu vykreslovani ?

fig, ax = plt.subplots(figsize=(12,5), sharex=True)#vytvoreni grafu
cml_plot.tl.plot.line(x='time', ax=ax, label = 'TL');#zadani souradnic a os grafu

cml_plot['wet'][0] = 0 
cml_plot['wet'][-1] = 0 
wet_start = np.roll(cml_plot.wet, -1) & ~cml_plot.wet
wet_end = np.roll(cml_plot.wet, 1) & ~cml_plot.wet
for wet_start_i, wet_end_i in zip(
    wet_start.data.nonzero()[0],
    wet_end.data.nonzero()[0],
):
    ax.axvspan(cml_plot.time.data[wet_start_i], cml_plot.time.data[wet_end_i], color='b', alpha=0.1)#samotne nastaveni grafu ?

cml_plot.baseline.plot.line(x='time', ax=ax, label ='baseline');
ax.set_ylabel('wet')
ax.legend(loc = 'upper right')#nastaveni nazvu a legenda
plt.show()#samotne vykresleni grafu









cml['tsl'] = cml.tsl.where(cml.tsl != 255.0)
cml['rsl'] = cml.rsl.where(cml.rsl != -99.9)
cml['trsl'] = cml.tsl - cml.rsl
cml['trsl'] = cml.trsl.interpolate_na(dim='time', method='linear', max_gap='5min')

cml['wet'] = cml.trsl.rolling(time=60, center=True).std(skipna=False) > 0.8#stejne jako pred tim jen v cml protoze mi to neslo rozjet s cmls, znovu problem s jejich rozdilem ...
cml['wet_fraction'] = (cml.wet==1).sum() / len(cml.time)#podil mezi destovym obdobim a casem, pozdeji pouzit pro interpolaci ... proc potreba a co znamena ?

cml['baseline'] = pycml.processing.baseline.baseline_constant(trsl=cml.trsl, wet=cml.wet, n_average_last_dry=5,)

cml['A'] = cml.trsl - cml.baseline
cml['A'] = cml.A.where(cml.A >= 0, 0)#A je pomocna promenna protoze waa funkce nemaji primo pristup k R tak se dostava z A


cml['waa_schleiss'] = pycml.processing.wet_antenna.waa_schleiss_2013(rsl=cml.trsl,baseline=cml.baseline,wet=cml.wet,waa_max=2.2,delta_t=1,tau=15,)
 
cml['waa_leijnse'] = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(A_obs=cml.A,f_Hz=cml.frequency,pol=cml.polarization,L_km=cml.length,)

cml['waa_pastorek'] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(A_obs=cml.A,f_Hz=cml.frequency,pol=cml.polarization,L_km=cml.length,)#jednotlive waa metody a parametry ktere potrebuji k funkconsti s pomoci pycomlinku







for waa_method in ['leijnse', 'pastorek', 'schleiss']:
    cml[f'A_rain_{waa_method}'] = cml.trsl - cml.baseline - cml[f'waa_{waa_method}']#prednastaveni waa metod aby zahrnovaly trsl a baseline
    cml[f'A_rain_{waa_method}'] = cml[f'A_rain_{waa_method}'].where(cml[f'A_rain_{waa_method}'] >= 0, 0)#zkontrolovani A ?
    cml[f'R_{waa_method}'] = pycml.processing.k_R_relation.calc_R_from_A(
        A=cml[f'A_rain_{waa_method}'], L_km=float(cml.length), f_GHz=cml.frequency/1e9, pol=cml.polarization#vypocitani R pro jednotlive metody z A
    )
cml['R'] = pycml.processing.k_R_relation.calc_R_from_A(A=cml.trsl - cml.baseline, L_km=float(cml.length), f_GHz=cml.frequency/1e9, pol=cml.polarization)#vypocitani R z A jako samostatny parametr ktery muzeme dal zavolat




#vypisovani grafu
fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
plt.sca(axs[0])
cml.trsl.isel(channel_id=0).plot.line(x='time', label='TRSL', color='k', zorder=10)
cml.baseline.isel(channel_id=0).plot.line(x='time', label='baseline', color='C0')
(cml.baseline + cml.waa_leijnse).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_leijnse', color='C1')
(cml.baseline + cml.waa_pastorek).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_pastorek', color='C2')
(cml.baseline + cml.waa_schleiss).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_schleiss', color='C3')#predchozich 5 radku jsou podobne vypsani jednotlivych car/casti grafu nejsem si jisty 2 vecmi channel_id (pravdepodobne id kanalu ale jak poznat ?) a proc .isel (.plot.line dava smysl ale nevim co znamena .isel mozna neco s daty ?)
plt.ylabel('total path attenuation in dB')
plt.title(f'cml_id = {cml.cml_id}   length = {cml.length:2.2f} km   frequency = {cml.frequency.isel(channel_id=0)/1e9} GHz')#nazev celeho grafu
plt.legend()

plt.sca(axs[1])
(radar_along_cml.sel(cml_id=cml.cml_id.values).rainfall_amount * 12).plot.line(color='k', linewidth=3.0, label='RADKLIM-YW', alpha=0.3)#radar co meril stejne jako waa *12 je protoze radarove mereni je zaznamenano 1/hod a waa je 1/5min
cml.R.isel(channel_id=0).plot.line(x='time', label='no WAA', color='C0')
cml.R_leijnse.isel(channel_id=0).plot.line(x='time', label='with WAA_leijnse', color='C1')
cml.R_pastorek.isel(channel_id=0).plot.line(x='time', label='with WAA_pastorek', color='C2')
cml.R_schleiss.isel(channel_id=0).plot.line(x='time', label='with WAA_schleiss', color='C3')#podobne jako predtim akorat pro jine jednotky
plt.ylabel('Rain-rate in mm/h')
plt.title('')
plt.legend()

plt.sca(axs[2])
radar_along_cml.sel(cml_id=cml.cml_id.values).rainfall_amount.cumsum(dim='time').plot.line(color='k', linewidth=3.0, label='RADKLIM-YW', alpha=0.3)
(cml.R.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='no WAA', color='C0')
(cml.R_leijnse.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_leijnse', color='C1')
(cml.R_pastorek.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_pastorek', color='C2')
(cml.R_schleiss.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_schleiss', color='C3')#podobne jako predtim akorat pro jine jednotky, ale vypocet sum je zvlastni proc /60 ?
plt.ylabel('Rainfall sum in mm')
plt.title('')
plt.legend();
plt.show()












#interpolacni metody jen kopie z pcml jen pridano pro lespi znazorneni pro me, posledni komentar
ds_cmls = xr.concat(cml_list, dim='cml_id')

rainsum_5min = ds_cmls.sel(channel_id="channel_1").R.resample(time="5min").sum() / 60

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

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))

bounds = [0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20] 
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')
cmap = plt.get_cmap('turbo').copy()
cmap.set_under('w')

for i, axi in enumerate(ax.flat):
    R_grid = idw_interpolator(
        x=cmls_R_1h.lon_center, 
        y=cmls_R_1h.lat_center, 
        z=cmls_R_1h.R.isel(channel_id=1).isel(time=i + 88).where(ds_cmls.wet_fraction < 0.3), 
        resolution=0.01,
    )
    pc = axi.pcolormesh(
        idw_interpolator.xgrid, 
        idw_interpolator.ygrid, 
        R_grid, 
        shading='nearest', 
        cmap=cmap,
        norm=norm,
    )
    axi.set_title(str(cmls_R_1h.time.values[i + 88])[:19])
    
    plot_cml_lines(cmls_R_1h, ax=axi)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
cb = fig.colorbar(pc, cax=cbar_ax, label='Hourly rainfall sum in mm', );
plt.show()
