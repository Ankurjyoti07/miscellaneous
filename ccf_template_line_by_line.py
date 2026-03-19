import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import scipy.optimize as so
import os, csv, glob
from astropy.io import fits
from tqdm import tqdm

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


def read_line_list(filename):
    line_centers = []
    line_widths = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            center = float(parts[0])
            if len(parts) > 1:
                width = float(parts[1])
            else:
                width = 14.0
            line_centers.append(center)
            line_widths.append(width)

    return line_centers, line_widths

def compute_mean_flux_for_lines(spectra_files, line_centers, line_widths):
    all_flux = []
    for spectrum_file in spectra_files:
        wavelengths, flux = read_spectrum(spectrum_file)
        all_flux.append(flux)    
    all_flux = np.array(all_flux)
    mean_flux = np.nanmedian(all_flux, axis=0) #ignore nan values for any corrupted file

    mean_flux_for_lines = []
    for i, center in enumerate(line_centers):
        lower_bound = center - line_widths[i]
        upper_bound = center + line_widths[i]
        mask = (wavelengths >= lower_bound) & (wavelengths <= upper_bound)
        mean_flux_for_lines.append(mean_flux[mask])
    
    return mean_flux_for_lines

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

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave, flux = data[0], data[1]
    return wave, flux


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

def process_time_series(spectra_files, line_list_file, velocity, output_csv='rv_results.csv', log_file='error_log.txt'):
    line_centers, line_widths = read_line_list(line_list_file)
    mean_flux = compute_mean_flux_for_lines(spectra_files, line_centers, line_widths)
    
    if not os.path.exists(output_csv):
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Filename', 'UTC', 'MJD-OBS', 'Airmass'] + [f'RV_{center}_Å' for center in line_centers] + [f'RV_error_{center}_Å' for center in line_centers]
            writer.writerow(headers)
    
    with open(log_file, 'w') as log:
        log.write("Error Log\n")

    for infile in tqdm(spectra_files, desc = 'processing:'):

        try:
            wavelength, flux = read_spectrum(infile)
            header = fits.getheader(infile)
            utc_time = header['DATE-OBS']
            mjd_obs = header['MJD-OBS']
            airmass = header['HIERARCH ESO QC AIRM AVG']

            rv_values, rv_errors = [], []
            for i, center in enumerate(line_centers):
                mask = (wavelength > (center - line_widths[i])) & (wavelength < (center + line_widths[i]))  ## changes
                flux_line = flux[mask]
                wavelength_line = wavelength[mask]
                if len(flux_line) == 0:
                    rv_values.append(None)
                    rv_errors.append(None)
                    continue

                ccf = calc_ccf_template(wavelength_line, mean_flux[i], flux_line, velocity)
                try:
                    rv, rv_error = rv_determination(wavelength_line, mean_flux[i], flux_line, velocity, center, infile, plot=True, output_dir='../ccf_plots') # saves ccf plots/ harcoded for working with ccf loop
                    rv_values.append(rv)
                    rv_errors.append(rv_error)
                except RuntimeError:
                    rv_values.append(None)
                    rv_errors.append(None)

            with open(output_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                row = [infile, utc_time, mjd_obs, airmass] + rv_values + rv_errors
                writer.writerow(row)
        except Exception as e:

            with open(log_file, 'a') as log:
                log.write(f"Error with file {infile}: {str(e)}\n")
                print('error logged for:', infile)
            continue

    print(f"RV results saved to {output_csv}")


spectra_files = glob.glob('../normalized_spectra/norm/*.fits')
line_list_file = '/home/c4011027/PhD_stuff/ESO_proposals/prologs/line_list.txt'
velocity = np.linspace(-100, 100, 200)
process_time_series(spectra_files, line_list_file, velocity)
