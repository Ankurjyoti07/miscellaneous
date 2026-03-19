import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from astropy.io import fits

def read_spectrum(infile):
    data = fits.getdata(infile)
    wave, flux = data[0], data[1]
    return wave, flux

def get_mean_flux(glob_specdir):
    all_flux = []
    for spectrum_file in glob_specdir:
        wavelengths, flux = read_spectrum(spectrum_file)
        all_flux.append(flux)
    all_flux = np.array(all_flux)
    mean_flux = np.nanmedian(all_flux, axis=0)
    return wavelengths, mean_flux

def initialize_shift_file(filename, n_segments):
    with open(filename, 'w') as f:
        for i in range(n_segments):
            f.write(f"Segment_{i+1}: 0.0\n")

def read_shift_file(filename):
    shift_values = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                shift = float(line.strip().split(":")[1])
            except (IndexError, ValueError):
                shift = 0.0
            shift_values.append(shift)
    return shift_values

def update_shift_file(filename, segment_index, new_value):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines[segment_index] = f"Segment_{segment_index+1}: {new_value:.12f}\n"
    with open(filename, 'w') as f:
        f.writelines(lines)

# --- Setup ---
specdir = glob.glob('/home/c4011027/PhD_stuff/ESO_proposals/normalized_corrected/*.fits')
wave, flux = get_mean_flux(specdir)

txt_file = "selected_wavelengths_new.txt"
if os.path.exists(txt_file):
    order_merging_points = np.loadtxt(txt_file).tolist()
    if isinstance(order_merging_points, float):
        order_merging_points = [order_merging_points]
else:
    order_merging_points = []

order_merging_points = [wave[0]] + sorted(order_merging_points) + [wave[-1]]

segments = []
for i in range(len(order_merging_points) - 1):
    mask = (wave >= order_merging_points[i]) & (wave < order_merging_points[i + 1])
    segments.append((wave[mask], flux[mask]))

# --- Segment shift file logic ---
shift_file = "segment_shifts_new.txt"
n_segments = len(segments)

if os.path.exists(shift_file):
    shift_values = read_shift_file(shift_file)
    if len(shift_values) != n_segments:
        raise ValueError(f"Mismatch: {len(shift_values)} shifts found, but {n_segments} segments exist.")
else:
    initialize_shift_file(shift_file, n_segments)
    shift_values = np.zeros(n_segments)

shift_values = np.array(shift_values)

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(12, 6))
lines = [ax.plot(seg[0], seg[1] + shift_values[i], label=f"Segment {i+1}")[0]
         for i, seg in enumerate(segments)]

ax.set_xlabel("Wavelength")
ax.set_ylabel("Flux")
ax.set_title("Click a Segment & Drag to Shift")

for wl in order_merging_points[1:-1]:
    ax.axvline(wl, color='red', linestyle='--', label="Order Merging" if 'Order Merging' not in ax.get_legend_handles_labels()[1] else "")

selected_segment = None
click_y = None

def find_segment(x):
    for i, (wave_seg, _) in enumerate(segments):
        if wave_seg[0] <= x <= wave_seg[-1]:
            return i
    return None

def on_click(event):
    global selected_segment, click_y
    if event.inaxes != ax or event.button != 1:
        return
    selected_segment = find_segment(event.xdata)
    if selected_segment is not None:
        click_y = event.ydata

def on_motion(event):
    global selected_segment, click_y
    if selected_segment is None or event.inaxes != ax or click_y is None:
        return
    shift_amount = (event.ydata - click_y)
    shift_values[selected_segment] += shift_amount
    lines[selected_segment].set_ydata(segments[selected_segment][1] + shift_values[selected_segment])
    click_y = event.ydata
    update_shift_file(shift_file, selected_segment, shift_values[selected_segment])
    plt.draw()

def on_release(event):
    global selected_segment, click_y
    selected_segment = None
    click_y = None

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
plt.show()
