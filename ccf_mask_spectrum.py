import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import os, csv, glob
from astropy.io import fits
from tqdm import tqdm
from astropy.stats import sigma_clip


def gaussian(x, center, height, std, yoffset):
    return height * np.exp(-1 * (x - center)**2 / (2*std**2)) + yoffset

def double_gaussian(x, center1, height1, std1, center2, height2, std2, yoffset):
    return gaussian(x, center1, height1, std1, 0) + gaussian(x, center2, height2, std2, 0) + yoffset

def quadratic(x, q, a, c):
    b = -2*q*a
    return a * x**2 + b*x + c

def dopler_shift(w, rv):
    w = np.array(w)
    c = 299792.458
    return w * c / (c - rv)

def read_region_list(filename=None, region_list=None):
    regions = []
    if filename:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                start, end = map(float, line.split())
                regions.append((start, end))
    elif region_list:
        regions = region_list
    return regions

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

def mask_spectrum(wave, flux, regions, base_continuum=1.0, variation=0.004):

    line_mask = np.zeros_like(wave, dtype=bool)
    for start, end in regions:
        line_mask |= (wave >= start) & (wave <= end)
    masked_flux = flux.copy()
    random_variations = np.random.normal(loc=base_continuum, scale=variation, size=masked_flux.shape)
    masked_flux[~line_mask] = random_variations[~line_mask]

    return wave, masked_flux


def compute_mean_flux_for_lines(spectra_files):

    all_flux = []
    for spectrum_file in spectra_files:
        _, flux = read_spectrum(spectrum_file)
        all_flux.append(flux)    
    all_flux = np.array(all_flux)
    mean_flux = np.nanmedian(all_flux, axis=0)

    return mean_flux


def calc_ccf_template(wavelength, flux1, flux2, velocity):
    ccf = []
    w_low = dopler_shift(wavelength[0], max(velocity))
    w_high = dopler_shift(wavelength[-1], min(velocity))
    delta = wavelength[1] - wavelength[0]
    w_new = np.arange(w_low, w_high, delta/3.)
    for v in velocity:
        w2 = dopler_shift(wavelength, v)
        f1 = np.abs(1 - np.interp(w_new, wavelength, flux1))
        f2 = np.abs(1 - np.interp(w_new, w2, flux2))
        ccf.append(np.sum(f1*f2))
    return ccf

def rv_determination(w, f1, f2, velocity, center, infile, method = 'gaussian', plot = False, ycutoff=-99999, output_dir='ccf_plots'):     
    ccf = np.array(calc_ccf_template(w, f1, f2, velocity))
    inds = (ccf >= ycutoff)
    if method == 'gaussian':
        popt, pcov = so.curve_fit(gaussian, velocity[inds], ccf[inds], p0=[velocity[np.argmax(ccf)], np.max(ccf)-np.min(ccf), 20, np.min(ccf)], bounds=([-np.inf, 0, 10, -np.inf], np.inf))
    elif method == 'double_gaussian':
        popt, pcov = so.curve_fit(double_gaussian, velocity[inds], ccf[inds], p0=[velocity[np.argmax(ccf)], np.max(ccf)-np.min(ccf), 20, 1 - velocity[np.argmax(ccf)], (np.max(ccf)-np.min(ccf))/2, 20, np.min(ccf)], bounds=([-np.inf, 0, 10, -np.inf, 0, 10, -np.inf], np.inf))
    elif method == 'quadratic':
        popt, pcov = so.curve_fit(quadratic, velocity[inds], ccf[inds])
    
    rv = popt[0]
    if method == 'gaussian':
        rv_error = np.sqrt(np.diag(pcov))[0]
    elif method == 'double_gaussian':
        rv1 = popt[0]
        rv2 = popt[3]
        rv1_error = np.sqrt(np.diag(pcov))[0]
        rv2_error = np.sqrt(np.diag(pcov))[3]
        rv = [rv1, rv2]
        rv_error = [rv1_error, rv2_error]
    elif method == 'quadratic':
        rv_error = np.sqrt(np.diag(pcov))[0]

    if plot:
        base_filename = os.path.splitext(os.path.basename(infile))[0]
        plot_filename = f"{base_filename}_{center:.2f}_ccf.png" #centre is set outside function in ccf loop
        plot_path = os.path.join(output_dir, plot_filename)
        os.makedirs(output_dir, exist_ok=True)  # ensure directory exists

        plt.figure()
        plt.plot(velocity, ccf, label='CCF')
        if method == 'gaussian':
            plt.plot(velocity[inds], gaussian(velocity[inds], *popt), label='Gaussian Fit')
        elif method == 'double_gaussian':
            plt.plot(velocity[inds], double_gaussian(velocity[inds], *popt), label='Double Gaussian Fit')
        elif method == 'quadratic':
            plt.plot(velocity[inds], quadratic(velocity[inds], *popt), label='Quadratic Fit')
        
        plt.axvline(popt[0], color='r', linestyle='--', label=f'RV: {popt[0]:.2f} km/s')
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('CCF')
        plt.title(f'Line Center: {center} Å') #centre is set outside function in ccf loop
        plt.text(0.1, 0.9, f'RV Error: {np.sqrt(np.diag(pcov))[0]:.2f} km/s', transform=plt.gca().transAxes)
        plt.legend()

        # Save the plot
        plt.savefig(plot_path)
        plt.close()

    return rv, rv_error

def process_dir(spec_dir, output_csv = 'rv_full_spectrum.csv', out_dir = 'ccf_plots', velocity = np.linspace(-100, 100, 200), region = [(4400, 4500), (5500, 5600)]):

    all_flux, mjd_list, utc_time_list, snr_list, file_names = [], [], [], [], []

    spectra_files = glob.glob(spec_dir)
    for spectrum_file in spectra_files:
        wave, flux = read_spectrum(spectrum_file)
        with fits.open(spectrum_file) as hdul:
            header = hdul[0].header
            mjd = header.get('MJD-OBS', np.nan)  
            utc_time = header.get('DATE-OBS', 'Unknown') 
            snr = header.get('SNR', np.nan)
        _, flux_tc = telluric_correction(wave, flux)    
        _, masked_flux = mask_spectrum(wave, flux_tc, regions = region, base_continuum=1.0, variation=0.004)

        all_flux.append(masked_flux)
        mjd_list.append(mjd)
        utc_time_list.append(utc_time)
        snr_list.append(snr)
        file_names.append(spectrum_file)

    all_flux = np.array(all_flux) 
    mean_flux = np.nanmedian(all_flux, axis=0)
    
    rv_list, rv_err_list = [], []
    for flux_epoch in all_flux:
        rv, rv_err = rv_determination(wave, mean_flux, flux_epoch, velocity, method='gaussian', plot=False, output_dir= out_dir)
        rv_list.append(rv)
        rv_err_list.append(rv_err)

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['file_name', 'mjd', 'utc_time', 'snr', 'rv', 'rv_err'])
        
        for i in range(len(file_names)):
            writer.writerow([file_names[i], mjd_list[i], utc_time_list[i], snr_list[i], rv_list[i], rv_err_list[i]])
    
    print('Done Processing!!!')


spectra_dir = '/path/to/spectra/files/*.fits'
output_csv = 'rv_results.csv'
velocity = np.linspace(-100, 100, 200)
process_dir(spectra_dir, output_csv = output_csv,  out_dir = 'ccf_plots', region = [(4400, 4500), (5500, 5600)], velocity = velocity)

