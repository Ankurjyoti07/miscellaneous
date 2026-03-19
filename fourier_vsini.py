import numpy as np
from scipy.signal import find_peaks
from astropy import constants, units
import sys, glob, os, logging, warnings, csv
from astropy.io import fits
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as svg
from astropy.constants import c
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks, fftconvolve
from astropy.stats import sigma_clip
from tqdm import tqdm

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave = data[0]
    flux = data[1]
    return wave, flux

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

def read_fourier_list(filename):
    line_list = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            clam = float(parts[0])  # Convert central wavelength to float            
            # Manually parse the window values without eval
            window_str = parts[1].strip("[]")  # Remove the brackets
            window = [float(val) for val in window_str.split(',')]  # Convert to list of floats            
            line_list.append({'clam': clam, 'window': window})
    return line_list

def telluric_correction(wv, flx):
    window_size = 100
    cleaned_flx = np.copy(flx)

    for i in range(0, len(wv), window_size):
        flx_window = flx[i:i + window_size]
        clipped_flux = sigma_clip(flx_window, sigma=2, maxiters=10, masked=True)
        cleaned_flx[i:i + window_size] = np.where(clipped_flux.mask, np.nan, flx_window)

    cleaned_flx = np.interp(wv, wv[~np.isnan(cleaned_flx)], cleaned_flx[~np.isnan(cleaned_flx)])
    return wv, cleaned_flx


def vsini(wave, flux, epsilon=0.0, clam=None, window=None):
    cc = constants.c.to(units.AA / units.s).value

    if window is not None:
        keep = (window[0] <= wave) & (wave <= window[1])
        wave, flux = wave[keep], flux[keep]

    clam = clam or np.mean(wave)

    q1 = 0.610 + 0.062 * epsilon + 0.027 * epsilon ** 2 + 0.012 * epsilon ** 3 + 0.004 * epsilon ** 4

    # Using Lomb-Scargle periodogram instead of FFT
    frequency, power = LombScargle(wave, flux).autopower(samples_per_peak=50)

    power /= max(power)  # Normalize power for comparison with ampls

    # Find minima in the power spectrum
    peaks, _ = find_peaks(-power)
    minima = frequency[peaks][:-1]
    minvals = power[peaks][:-1]

    frequency = frequency * clam / q1 / cc
    vsini_values = cc / clam * q1 / minima

    # Estimate the error based on the wavelength range and scaling factor
    error = np.ptp(wave) * clam / q1 / cc

    return (frequency, power), (minima, minvals), vsini_values / 10**13, error

def plot_ft(wave, flux, result, epsilon=0.6, clam=6678, window=[6670, 6686], save_path = '/home/c4011027/PhD_stuff/ESO_proposals/trash_stuff/FT.png'):
    
    cc = constants.c.to(units.AA / units.s).value
    q1 = 0.610 + 0.062 * epsilon + 0.027 * epsilon ** 2 + 0.012 * epsilon ** 3 + 0.004 * epsilon ** 4
    freqs = result[0][0] /( clam / q1 / cc)
    ampls = result[0][1]
    velocities = (cc / clam * q1 / freqs) / 10**13
    log_ampls = np.log(ampls)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), sharex=False)
    ax1.plot(velocities, log_ampls, color='darkorange', linewidth=2, label='FT Amplitude')
    ax1.axvline(max(result[2]), color='blue', linestyle='--', label=f'vsini: {max(result[2]):.2f} km/s')
    ax1.set_ylabel('Log Amplitude')
    ax1.set_xlabel('Velocity (km/s)')
    ax1.set_title('Fourier Transform: Velocity vs Log Amplitude')
    ax1.grid(True)
    ax1.set_xlim(0, 500)
    ax1.legend()
    
    buffer_region = [window[0] - 10, window[1] + 10]
    buffer_keep = (buffer_region[0] <= wave) & (wave <= buffer_region[1])
    window_keep = (window[0] <= wave) & (wave <= window[1])
    ax2.plot(wave[buffer_keep], flux[buffer_keep], color='grey', linewidth=1.5, label='Spectrum with Buffer')
    ax2.plot(wave[window_keep], flux[window_keep], color='darkorange', linewidth=1.5, label='Windowed Spectrum')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    #plt.close()

def initialize_csv(result_file, line_list):
    headers = ["Spectrum"] + [f"Line_{line['clam']:.2f}" for line in line_list]
    if not os.path.exists(result_file):
        with open(result_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

def append_to_csv(result_file, row_data):
    with open(result_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        writer.writerow(row_data)

def process_spectra(specdir, line_list, epsilon=0.6, 
                    result_file='/home/c4011027/PhD_stuff/ESO_proposals/checks/ftresults/results.csv',
                    plot_dir='/home/c4011027/PhD_stuff/ESO_proposals/checks/ftplots/'):
    plots_dir = plot_dir
    os.makedirs(plots_dir, exist_ok=True)

    logging.basicConfig(
        filename='error_log_timeseriesFT.txt', 
        level=logging.ERROR, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    initialize_csv(result_file, line_list)
    specfiles, _ = sort_spectra(glob.glob(specdir)[0:500])    
    for files in tqdm(specfiles):
        try:
            spectrum_name = files.split('/')[-1].strip('.fits')
            row_data = {"Spectrum": spectrum_name}

            for line in line_list:
                clam, window = line['clam'], line['window']
                wv, flx = read_spectrum(files)
                wave, flux = telluric_correction(wv, flx)
                result = vsini(wave, flux, clam=clam, window=window)
                vsini_max = max(result[2])
                row_data[f"Line_{clam:.2f}"] = vsini_max
                plot_filename = os.path.join(plots_dir, f"{spectrum_name}_{clam:.2f}.png")
                plot_ft(wave, flux, result, epsilon=epsilon, clam=clam, window=window, save_path=plot_filename)

            append_to_csv(result_file, row_data)
        except Exception as e:
            logging.error(f"Error processing {files}: {str(e)}")
            continue


linelist = read_fourier_list('/home/c4011027/PhD_stuff/ESO_proposals/prologs/fourier_linelist.txt')
specdir = '/home/c4011027/PhD_stuff/ESO_proposals/manual_normalization/norm/*.fits'
#result_folder = '/home/c4011027/PhD_stuff/ESO_proposals/checks/ftresults/'
results_file = '/home/c4011027/PhD_stuff/ESO_proposals/checks/ftresults/results.csv'
plots_folder = "/home/c4011027/PhD_stuff/ESO_proposals/checks/ftplots/"

process_spectra(specdir, linelist, epsilon=0.6, result_file=results_file, plot_dir=plots_folder)