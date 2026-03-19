import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits

def read_unnorm(infile_adp):
    data = fits.getdata(infile_adp)
    wave, flux = data[0][0], data[0][1]
    return wave, flux

def read_spectrum(infile_norm):
    data = fits.getdata(infile_norm)
    wave, flux = data[0], data[1]
    return wave, flux

unnorm_file = "/home/c4011027/PhD_stuff/ESO_proposals/data_zeta_oph3n/ADP.2024-07-05T12:10:57.849.fits"
norm_file = "/home/c4011027/PhD_stuff/ESO_proposals/renorm_o2/norm/ADP.2024-07-05T12:10:57.849_norm.fits"
txt_file = "selected_wavelengths.txt"
wave, flux = read_unnorm(unnorm_file)
wave_norm, flux_norm = read_spectrum(norm_file)

### this part is a temporary fix
shift_array = wave - wave_norm
wave_corrected = wave - shift_array

if os.path.exists(txt_file):
    order_merging_points = np.loadtxt(txt_file).tolist()
    if isinstance(order_merging_points, float):
        order_merging_points = [order_merging_points]

order_merging_points = [wave_corrected[0]] + sorted(order_merging_points) + [wave_corrected[-1]]
segments = []
for i in range(len(order_merging_points) - 1):
    mask = (wave_corrected >= order_merging_points[i]) & (wave_corrected < order_merging_points[i + 1])
    segments.append((wave_corrected[mask], flux[mask]))
shift_values = np.zeros(len(segments))

fig, ax = plt.subplots(figsize=(12, 6))
lines = [ax.plot(seg[0], seg[1], label=f"Segment {i+1}")[0] for i, seg in enumerate(segments)]
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
    plt.draw()

def on_release(event):
    global selected_segment, click_y
    selected_segment = None
    click_y = None

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
plt.show()
