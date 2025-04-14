
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycomlink as pycml
import xarray as xr
import pandas as pd



def load_data(data_path, start, end): #nacteni dat
    cmls = xr.open_dataset(data_path + '/example_cml_data.nc')
    return cmls

def load_ref(data_path): #pouziva se potom ve vykreslovani interpolace
    path_ref = xr.open_dataset(data_path + '/example_path_averaged_reference_data.nc')
    return path_ref


def load_radar(): #nacteni pomocnych dat z radaru
    radar_along_cml = xr.open_dataset(pycml.io.examples.get_example_data_path() + '/example_path_averaged_reference_data.nc')
    return radar_along_cml

def preprocess_data(cmls, start, end): #zpracovani dat

    cml_list = [cmls.isel(cml_id=i) for i in range(len(cmls.cml_id))]
    cml = cml_list[0]

    cml['tsl'] = cml.tsl.where(cml.tsl != 255.0)
    cml['rsl'] = cml.rsl.where(cml.rsl != -99.9)
    cml['trsl'] = cml.tsl - cml.rsl
    cml['trsl'] = cml.trsl.interpolate_na(dim='time', method='linear', max_gap='5min')

    cml['wet'] = cml.trsl.rolling(time=60, center=True).std(skipna=False) > 0.8
    cml['wet_fraction'] = (cml.wet==1).sum() / len(cml.time)

    cml['baseline'] = pycml.processing.baseline.baseline_constant(trsl=cml.trsl, wet=cml.wet, n_average_last_dry=5,)

    cml['A'] = cml.trsl - cml.baseline
    cml['A'] = cml.A.where(cml.A >= 0, 0)   
    return cml #vraceni cml, snad staci takhle jako vraceni zpracovanych dat


def waamethods(cml, i, start, end):#nastaveni jednotlivych metod, k cemu tu slouzi i ? mozna lepsi pouzit u cml = cml_list[i] ?
    cml['waa_schleiss'] = pycml.processing.wet_antenna.waa_schleiss_2013(rsl=cml.trsl,baseline=cml.baseline,wet=cml.wet,waa_max=2.2,delta_t=1,tau=15,)
    cml['waa_leijnse'] = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(A_obs=cml.A,f_Hz=cml.frequency,pol=cml.polarization,L_km=cml.length,)
    cml['waa_pastorek'] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(A_obs=cml.A,f_Hz=cml.frequency,pol=cml.polarization,L_km=cml.length,)

    for waa_method in ['leijnse', 'pastorek', 'schleiss']:
        cml[f'A_rain_{waa_method}'] = cml.trsl - cml.baseline - cml[f'waa_{waa_method}']
        cml[f'A_rain_{waa_method}'] = cml[f'A_rain_{waa_method}'].where(cml[f'A_rain_{waa_method}'] >= 0, 0)
        cml[f'R_{waa_method}'] = pycml.processing.k_R_relation.calc_R_from_A(
        A=cml[f'A_rain_{waa_method}'], L_km=float(cml.length), f_GHz=cml.frequency/1e9, pol=cml.polarization
        )
        
    cml['R'] = pycml.processing.k_R_relation.calc_R_from_A(A=cml.trsl - cml.baseline, L_km=float(cml.length), f_GHz=cml.frequency/1e9, pol=cml.polarization)
    return cml #opet vraceni cml, snad se ukladaji zmenena data takto


def plot_waa_results(cml, radar_along_cml): #vykresleni grafu (nefunkcni)
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True) #pise fig is not accessed
    plt.sca(axs[0])
    cml.trsl.isel(channel_id=0).plot.line(x='time', label='TRSL', color='k', zorder=10)
    cml.baseline.isel(channel_id=0).plot.line(x='time', label='baseline', color='C0')
    (cml.baseline + cml.waa_leijnse).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_leijnse', color='C1')
    (cml.baseline + cml.waa_pastorek).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_pastorek', color='C2')
    (cml.baseline + cml.waa_schleiss).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_schleiss', color='C3')
    plt.ylabel('total path attenuation in dB')
    plt.title(f'cml_id = {cml.cml_id}   length = {cml.length:2.2f} km   frequency = {cml.frequency.isel(channel_id=0)/1e9} GHz')
    plt.legend()

    plt.sca(axs[1])
    (radar_along_cml.sel(cml_id=cml.cml_id.values).rainfall_amount * 12).plot.line(color='k', linewidth=3.0, label='RADKLIM-YW', alpha=0.3)
    cml.R.isel(channel_id=0).plot.line(x='time', label='no WAA', color='C0')
    cml.R_leijnse.isel(channel_id=0).plot.line(x='time', label='with WAA_leijnse', color='C1')
    cml.R_pastorek.isel(channel_id=0).plot.line(x='time', label='with WAA_pastorek', color='C2')
    cml.R_schleiss.isel(channel_id=0).plot.line(x='time', label='with WAA_schleiss', color='C3')
    plt.ylabel('Rain-rate in mm/h')
    plt.title('')
    plt.legend()

    plt.sca(axs[2])
    radar_along_cml.sel(cml_id=cml.cml_id.values).rainfall_amount.cumsum(dim='time').plot.line(color='k', linewidth=3.0, label='RADKLIM-YW', alpha=0.3)
    (cml.R.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='no WAA', color='C0')
    (cml.R_leijnse.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_leijnse', color='C1')
    (cml.R_pastorek.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_pastorek', color='C2')
    (cml.R_schleiss.isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_schleiss', color='C3')
    plt.ylabel('Rainfall sum in mm')
    plt.title('')
    plt.legend();
    plt.show()



def main():
    waa_methods=['waa_schleiss','waa_leijnse','waa_pastorek'] #jen seznam metod ?

    start = '2018-05-01'
    end = '2018-05-30'

    data_path = pycml.io.examples.get_example_data_path()

    cmls = load_data(data_path, start, end)

    ref = load_ref(data_path)

    radar = load_radar()

    cml = preprocess_data(cmls, start, end)

    rains = {}
    for i in waa_methods:
        result_waa = waamethods(cml, i, start, end)
        rains[i] = result_waa #k cemu je zde tato cast jen okopirovane z emailu - mozna nejaka mnozina kam se ukladaji vysledky?
        plot_waa_results(result_waa, radar)



main