import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
import glob, warnings, gzip, logging, os, sys, csv
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from math import sin, pi
from scipy.special import erf                               # Error function 
from lmfit import Parameters, minimize
from astropy.stats import sigma_clip
from math import ceil, sqrt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

def sort_spectra(spectra_files):
    spectra_with_dates = []
    for spectrum_file in spectra_files:
        obs_date = fits.getheader(spectrum_file)['MJD-OBS']
        if obs_date:
            spectra_with_dates.append((spectrum_file, obs_date))
    spectra_with_dates.sort(key=lambda x: x[1])
    sorted_spectra_files = [s[0] for s in spectra_with_dates]
    sorted_obs_dates = [s[1] for s in spectra_with_dates]
    return sorted_spectra_files, sorted_obs_dates

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave, flux = data[0], data[1]
    return wave, flux

def telluric_correction(wv, flx):
    window_size = 100
    cleaned_flx = np.copy(flx)

    for i in range(0, len(wv), window_size):
        flx_window = flx[i:i + window_size]
        clipped_flux = sigma_clip(flx_window, sigma=2, maxiters=10, masked=True)
        cleaned_flx[i:i + window_size] = np.where(clipped_flux.mask, np.nan, flx_window)

    cleaned_flx = np.interp(wv, wv[~np.isnan(cleaned_flx)], cleaned_flx[~np.isnan(cleaned_flx)])
    return wv, cleaned_flx

#reading from linelist to fit

def read_line_list(filename):
    line_centers = []
    line_widths = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  #removing whitespaces
            if not line:
                continue  #skipping empty lines
            parts = line.split()
            center = float(parts[0])
            if len(parts) > 1:
                width = float(parts[1])
            else:
                width = 10.0  #default
            line_centers.append(center)
            line_widths.append(width)

    return line_centers, line_widths


def generate_data(wave, flux, line_centers, line_widths, wavelength_slices):
    interp_func = interp1d(wave, flux, kind='linear')
    wave_slices = []
    flux_slices = []
    for center, width in zip(line_centers, line_widths):
        new_wave = np.linspace(center - width, center + width, wavelength_slices)
        new_flux = interp_func(new_wave)
        wave_slices.append(new_wave)
        flux_slices.append(new_flux)
    return np.concatenate(wave_slices), np.concatenate(flux_slices)

def gauss(x,a,center,R, gamma):
  sigma = sigma = 4471/ (2.0 * R * np.sqrt(2.0 * np.log(2))) 
  return a*np.exp(-(x-center)**2/(2*sigma**2)) + gamma

class Model_broad:
    def __init__(self, wave, flux):
        self.x = wave
        self.y = flux


def Broaden(model, vsini, epsilon=0.5, linear=False, findcont=False):
    # Remove NaN values from the flux array and corresponding wavelength values
    non_nan_idx = ~np.isnan(model.y)
    wvl = model.x[non_nan_idx]
    flx = model.y[non_nan_idx]
    
    dwl = wvl[1] - wvl[0]
    binnu = int(np.floor((((vsini/10)/ 299792.458) * max(wvl)) / dwl)) + 1 #adding extra bins for error handling
    #validIndices = np.arange(len(flx)) + binnu => this was used in rotbroad as a user cond ==> this is always on here
    front_fl = np.ones(binnu) * flx[0]
    end_fl = np.ones(binnu) * flx[-1]
    flux = np.concatenate((front_fl, flx, end_fl))

    front_wv = (wvl[0] - (np.arange(binnu) + 1) * dwl)[::-1]
    end_wv = wvl[-1] + (np.arange(binnu) + 1) * dwl
    wave = np.concatenate((front_wv, wvl, end_wv))

    if not linear:
        x = np.logspace(np.log10(wave[0]), np.log10(wave[-1]), len(wave))
    else:
        x = wave
        
    if findcont:
        # Find the continuum
        model.cont = np.ones_like(flux)  # Placeholder for continuum finding
        
    # Make the broadening kernel
    dx = np.log(x[1] / x[0])
    c = 299792458  # Speed of light in m/s
    lim = vsini / c
    if lim < dx:
        warnings.warn("vsini too small ({}). Not broadening!".format(vsini))
        return Model_broad(wave.copy(), flux.copy())  # Create a copy of the Model object
    
    d_logx = np.arange(0.0, lim, dx)
    d_logx = np.concatenate((-d_logx[::-1][:-1], d_logx))
    alpha = 1.0 - (d_logx / lim) ** 2
    B = (1.0 - epsilon) * np.sqrt(alpha) + epsilon * np.pi * alpha / 4.0  # Broadening kernel
    B /= np.sum(B)  # Normalize

    # Do the convolution
    broadened = Model_broad(wave.copy(), flux.copy())  # Create a copy of the Model object
    broadened.y = fftconvolve(flux, B, mode='same')
    
    return broadened


def macro_broaden(xdata, ydata, vmacro):
    c = 299792458 #~constants.c.cgs.value * units.cm.to(units.km)
    sq_pi = np.sqrt(np.pi)
    lambda0 = np.median(xdata)
    xspacing = xdata[1] - xdata[0]
    mr = vmacro * lambda0 / c
    ccr = 2 / (sq_pi * mr)

    px = np.arange(-len(xdata) / 2, len(xdata) / 2 + 1) * xspacing
    pxmr = abs(px) / mr
    profile = ccr * (np.exp(-pxmr ** 2) + sq_pi * pxmr * (erf(pxmr) - 1.0))

    before = ydata[int(-profile.size / 2 + 1):]
    after = ydata[:int(profile.size / 2 +1)] #add one to fix size mismatch
    extended = np.r_[before, ydata, after]

    first = xdata[0] - float(int(profile.size / 2.0 + 0.5)) * xspacing
    last = xdata[-1] + float(int(profile.size / 2.0 + 0.5)) * xspacing
    
    x2 = np.linspace(first, last, extended.size)  #newdata x array ==> handles edge effects

    conv_mode = "valid"

    newydata = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)

    return newydata

def generate_broaden(params, line_centers, line_widths, wavelength_slices, vsini = 400000):
    model_slices = []
    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        wave = np.linspace(center - width, center + width, wavelength_slices)
        
        instrum = gauss(wave, params[f'a{i}'], params[f'center{i}'], 180000, params[f'gamma{i}']) #resolution is still hardcoded R=20000 change accordingly
        broad_rot = Broaden(Model_broad(wave, instrum), vsini)
        
        broad_macro = macro_broaden(broad_rot.x, broad_rot.y, params[f'vmacro{i}']) #macro broad restores the same wave array as input  
        
        interp = interp1d(broad_rot.x, broad_macro, kind= 'linear')
        broad_flux = interp(wave)
        model_slices.append(broad_flux)
        
    return  np.concatenate(model_slices)

def objective(params, wave, flux, line_centers, line_widths, wavelength_slices, vrot = 400000):
    wave_data, flux_data = generate_data(wave, flux, line_centers, line_widths, wavelength_slices)
    model = generate_broaden(params, line_centers, line_widths, wavelength_slices, vsini = vrot)
    return flux_data - model

def fit_lines(wave, flux, line_centers, line_widths, wavelength_slices, vrot):
    params = Parameters()
    wave_data, flux_data = generate_data(wave, flux, line_centers, line_widths, wavelength_slices)
    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        params.add(f'a{i}', value=-1)   # Initial guess for amplitude
        params.add(f'center{i}', value=center, min = center-1, max = center+1)  # Initial guess for center
        params.add(f'gamma{i}', value=1, min = 0.5, max = 1.2)
        params.add(f'vmacro{i}', value=200000, min = 0, max = 500000)

    result = minimize(objective, params=params, args=(wave_data, flux_data, line_centers, line_widths, wavelength_slices, vrot))
    return result

def save_fit_results_to_csv(result, line_centers, spectrum_name, output_csv):
    spectrum_data = [spectrum_name]
    headers = ["Spectrum"]
    
    for i, center in enumerate(line_centers):
        vmacro_key = f"vmacro_{center:.2f} Å"
        error_key = f"vmacro_error_{center:.2f} Å"
        model_params = result.params
        vmacro_value = model_params[f'vmacro{i}'].value
        vmacro_error = model_params[f'vmacro{i}'].stderr
        spectrum_data.extend([vmacro_value, vmacro_error])
        headers.extend([vmacro_key, error_key])

    file_exists = os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(spectrum_data)

    print(f"Saved results for {spectrum_name} to {output_csv}")


def process_spectra(vft_file, linelist = 'line_list.txt', output_csv = 'results_fixed_vsini.csv', base_path='/home/c4011027/PhD_stuff/ESO_proposals/manual_normalization/norm/'):
    
    logging.basicConfig(filename='error_log_fixed_rotation.txt', level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    line_centers, line_widths = read_line_list(linelist)
    
    df = pd.read_csv(vft_file)
    spectrum_names = np.array(df['Spectrum'])
    vsini_columns = df.columns[1:]  # assuming all columns after first are vsini s
    vsini_medians = np.array(df[vsini_columns].median(axis=1)) * 1000 # note: vsini values are in ms-1

    for spectrum, vsini_median in zip(spectrum_names, vsini_medians):

        try:
                
            wave, flux = read_spectrum(f"{base_path}{spectrum}.fits")
            _, flux_tc = telluric_correction(wave, flux)
            result = fit_lines(wave, flux_tc, line_centers, line_widths, wavelength_slices=1000, vrot = vsini_median) 

            #save_plot function goes here

            save_fit_results_to_csv(result, line_centers, spectrum, output_csv) 
        except Exception as e:
            logging.error(f"Failed to process spectrum {spectrum}.fits: {e}")
            continue