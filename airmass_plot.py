import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import glob
from astropy.io import fits
from astropy.time import Time
import math

def read_linelist(filename):
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

def calculate_flux_and_flag(wavelength, flux, line_center, line_width):
    line_mask = (wavelength >= line_center - line_width) & (wavelength <= line_center + line_width)
    
    if not np.any(line_mask):
        return 'absorption'
    
    mean_flux = np.mean(flux[line_mask])
    
    left_wing_mask = (wavelength >= line_center - line_width - 0.25) & (wavelength < line_center - line_width)
    right_wing_mask = (wavelength > line_center + line_width) & (wavelength <= line_center + line_width + 0.25)
    
    left_wing_flux = flux[left_wing_mask]
    right_wing_flux = flux[right_wing_mask]
    
    if len(left_wing_flux) == 0 or len(right_wing_flux) == 0:
        continuum_flux = np.mean(flux)
    else:
        continuum_flux = np.median(np.concatenate([left_wing_flux, right_wing_flux]))

    if mean_flux >= 1.05 * continuum_flux:
        return 'emission'
    else:
        return 'absorption'

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

def process_spectra(spectra_dir, linelist_file):
    spectra_files = glob.glob(f"{spectra_dir}/*.fits")
    sorted_spectra_files, sorted_obs_dates = sort_spectra(spectra_files)  ##change this line to access chuncks of full data 
    line_centers, line_widths = read_linelist(linelist_file)
    time_series_data = {center: [] for center in line_centers}
    
    for file in sorted_spectra_files:
        header = fits.getheader(file)
        airmass = 1 / np.cos(np.radians(90 - header['HIERARCH ESO TEL1 ALT']))
        mjd_time = header['MJD-OBS']  #use only MJD-OBS
        
        data = fits.getdata(file)
        wavelength = data[0]
        flux = data[1]
        for line_center, line_width in zip(line_centers, line_widths):
            flag = calculate_flux_and_flag(wavelength, flux, line_center, line_width)
            time_series_data[line_center].append((mjd_time, airmass, flag)) #add other info/quality_flag/utc_times here
    
    return time_series_data, sorted_spectra_files, sorted_obs_dates

def plot_time_series(time_series_data, sorted_spectra_files, sorted_obs_dates):
    num_lines = len(time_series_data)
    grid_size = math.ceil(math.sqrt(num_lines))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 8))
    axes = axes.flatten()    
    scatter_plots = []
    time_series_positions = {}
    
    for i, (line_center, data) in enumerate(time_series_data.items()):
        if not data:  #skip if no data
            continue
        
        mjd_times, airmasses, flags = zip(*data)
        colors = ['darkorange' if flag == 'emission' else 'slateblue' for flag in flags]
        ax = axes[i]
        scatter = ax.scatter(mjd_times, airmasses, c=colors, marker='*', alpha=0.5)
        scatter_plots.append(scatter)
        time_series_positions[i] = np.array(list(zip(mjd_times, airmasses)))
        ax.set_xlabel('MJD')
        ax.set_ylabel('Airmass')
        ax.set_title(f'Line Center: {line_center:.1f} Å')
        ax.legend(*scatter.legend_elements(), title="Line Status", loc='upper right')

        ax2 = ax.twiny()
        utc_times = Time(mjd_times, format='mjd').datetime
        utc_labels = [utc.strftime('%H:%M:%S') for utc in utc_times]
        tick_step = max(len(mjd_times) // 8, 1)
        reduced_mjd_ticks = mjd_times[::tick_step]
        reduced_utc_labels = utc_labels[::tick_step]
        ax2.set_xlim(ax.get_xlim())  #align MJD and UTC axes
        ax2.set_xticks(reduced_mjd_ticks)  #fewer ticks for UTC
        ax2.set_xticklabels(reduced_utc_labels, rotation=45, ha='left')
        ax2.set_xlabel('UTC Time')
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    #create cursor
    cursor = mplcursors.cursor(scatter_plots, hover=True)    
    @cursor.connect("add")
    def on_add(sel):
        mjd = sel.target[0]
        airmass = sel.target[1]
        ax = sel.artist.axes
        line_center = ax.get_title().split(': ')[1]
        color = sel.artist.get_facecolors()[sel.index]
        flag = 'Emission' if np.allclose(color, [1.0, 0.549, 0.0, 0.5]) else 'Absorption'
        sel.annotation.set(text=f"MJD: {mjd:.2f}\nAirmass: {airmass:.2f}\nLine: {line_center} Å\nFlag: {flag}")
        sel.annotation.get_bbox_patch().set(alpha=0.8)

    def onclick(event):
        for ax_idx, ax in enumerate(axes):
            clicked_pos = (event.xdata, event.ydata)  # Use mjd time for x-axis
            distances = np.linalg.norm(time_series_positions[ax_idx] - clicked_pos, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_mjd_time = time_series_positions[ax_idx][nearest_idx][0]
            mjd_differences = np.abs(np.array(sorted_obs_dates) - nearest_mjd_time)
            closest_mjd_idx = np.argmin(mjd_differences)
            #debug stuff
            spectrum_file = sorted_spectra_files[closest_mjd_idx]
            show_spectrum(spectrum_file)
            break    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


def show_spectrum(spectrum_file):
    data = fits.getdata(spectrum_file)
    wavelength = data[0]
    flux = data[1]
    plt.figure(figsize=(10, 8))
    plt.plot(wavelength, flux, label=f'Spectrum: {spectrum_file}', color = 'darkorange')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'Original Spectrum from {spectrum_file}')
    plt.legend()
    plt.show()

spectra_dir = '../man_norm/norm/'
linelist_file = 'line_list.txt'
time_series_data, sorted_spectra_files, sorted_obs_dates = process_spectra(spectra_dir, linelist_file)
plot_time_series(time_series_data, sorted_spectra_files, sorted_obs_dates)
