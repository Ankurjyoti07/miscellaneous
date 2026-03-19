import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import glob, warnings, gzip, logging, os, sys, csv, re
import matplotlib.gridspec as gridspec
import sys, glob, os, logging, warnings, csv
from tqdm import tqdm

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave, flux = data[0], data[1]
    return wave, flux

# Compute weighted mean flux
def get_mean_flux_weighted_efficient(glob_specdir):
    weighted_flux_sum = None
    weight_sum = None
    wavelengths = None

    for spectrum_file in glob_specdir:
        wl, flux = read_spectrum(spectrum_file)
        snr = fits.getheader(spectrum_file)['SNR']
        if wavelengths is None:
            wavelengths = wl
            weighted_flux_sum = np.zeros_like(flux, dtype=np.float64)
            weight_sum = np.zeros_like(flux, dtype=np.float64)

        valid_mask = ~np.isnan(flux)
        weighted_flux_sum[valid_mask] += flux[valid_mask] * snr
        weight_sum[valid_mask] += snr

    mean_flux = np.divide(weighted_flux_sum, weight_sum, 
                          out=np.full_like(weighted_flux_sum, np.nan), 
                          where=weight_sum != 0)

    return wavelengths, mean_flux

# Load spectrum
spec_files = glob.glob('/home/c4011027/PhD_stuff/ESO_proposals/normalized_corrected/*.fits')
wavelength, flux = get_mean_flux_weighted_efficient(spec_files)

output_file = "selected_wavelengths_new.txt"

# Load existing selections
if os.path.exists(output_file):
    selected_wavelengths = np.loadtxt(output_file).tolist()
    if isinstance(selected_wavelengths, float):
        selected_wavelengths = [selected_wavelengths]
else:
    selected_wavelengths = []

# Setup plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wavelength, flux, label='Spectrum')
ax.set_xlabel('Wavelength')
ax.set_ylabel('Flux')
ax.set_title('Click to mark regions; press "d" to delete nearest line')
lines = []
toolbar = plt.get_current_fig_manager().toolbar  # Get the Matplotlib toolbar

# Draw existing lines
for wl in selected_wavelengths:
    line = ax.axvline(wl, color='red', linestyle='--')
    lines.append((line, wl))
plt.legend()

# Find nearest wavelength
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Save to file
def save_wavelengths():
    with open(output_file, 'w') as f:
        for wl in selected_wavelengths:
            f.write(f"{wl:.6f}\n")

# Track mouse position
mouse_x = None

def onmove(event):
    global mouse_x
    if event.inaxes == ax:
        mouse_x = event.xdata

# Mouse click handler (only add if not zooming/panning)
def onclick(event):
    if toolbar.mode:  # If zoom or pan is active, ignore clicks
        return
    if event.inaxes != ax or event.button != 1:  # Only left-click
        return
    x_click = event.xdata
    nearest_wl = find_nearest(wavelength, x_click)
    line = ax.axvline(nearest_wl, color='red', linestyle='--')
    lines.append((line, nearest_wl))
    selected_wavelengths.append(nearest_wl)
    save_wavelengths()
    plt.draw()
    print(f"Selected: {nearest_wl:.4f}")

# Press 'd' to delete nearest line
def onkeypress(event):
    global mouse_x
    if event.key == 'd' and lines and mouse_x is not None:
        distances = [abs(wl - mouse_x) for _, wl in lines]
        min_idx = np.argmin(distances)
        line_obj, wl = lines.pop(min_idx)
        line_obj.remove()
        selected_wavelengths.remove(wl)
        save_wavelengths()
        plt.draw()
        print(f"Deleted: {wl:.4f}")

# Connect events
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmove)
fig.canvas.mpl_connect('key_press_event', onkeypress)

plt.show()