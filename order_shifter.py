import os
import re
import glob
import numpy as np
from astropy.io import fits

# --------------------------------------------------
# Paths
# --------------------------------------------------
shift_file = "/home/c4011027/PhD_stuff/spectra_ts/prologs_temp/segment_shifts_new.txt"
wave_file = "/home/c4011027/PhD_stuff/spectra_ts/prologs_temp/selected_wavelengths_new.txt"

input_pattern = "/home/c4011027/PhD_stuff/spectra_ts/results/zeta_oph_results/validation_test/fresh_validation_tests/espresso_norm/*.fits"
output_dir = "/home/c4011027/PhD_stuff/spectra_ts/results/zeta_oph_results/validation_test/fresh_validation_tests/espresso_order_fixed"

os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Read segment shifts
# --------------------------------------------------
def read_segment_shifts(filename):
    shifts = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = re.match(
                r"Segment_(\d+):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
                line
            )
            if m:
                seg_num = int(m.group(1))
                shift_val = float(m.group(2))
                shifts[seg_num] = shift_val

    if not shifts:
        raise ValueError(f"No valid segment shifts found in {filename}")

    max_seg = max(shifts.keys())
    shift_array = np.array([shifts[i] for i in range(1, max_seg + 1)], dtype=float)
    return shift_array


# --------------------------------------------------
# Read wavelength boundaries
# --------------------------------------------------
def read_wavelength_boundaries(filename):
    bounds = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                bounds.append(float(line))
            except ValueError:
                continue

    if len(bounds) < 1:
        raise ValueError(f"Need at least 1 wavelength boundary in {filename}")

    bounds = np.array(bounds, dtype=float)

    if not np.all(np.diff(bounds) > 0):
        print("WARNING: wavelength boundaries are not sorted in file. Sorting them now.")

    return np.sort(bounds)


# --------------------------------------------------
# Apply constant offsets segment by segment
# --------------------------------------------------
def apply_segment_offsets(wave, flux, boundaries, shifts):
    """
    Segment 1: wave < boundaries[0]
    Segment 2: boundaries[0] <= wave < boundaries[1]
    Segment 3: boundaries[1] <= wave < boundaries[2]
    ...
    Last segment: wave >= boundaries[-1]
    """

    n_expected_segments = len(boundaries) + 1
    n_shift_segments = len(shifts)

    if n_shift_segments != n_expected_segments:
        raise ValueError(
            f"Mismatch: wavelength file implies {n_expected_segments} segments, "
            f"but shift file contains {n_shift_segments} segment values."
        )

    flux_new = flux.copy()

    # Segment 1
    mask = wave < boundaries[0]
    flux_new[mask] += shifts[0]

    # Middle segments
    for i in range(1, len(boundaries)):
        mask = (wave >= boundaries[i - 1]) & (wave < boundaries[i])
        flux_new[mask] += shifts[i]

    # Last segment
    mask = wave >= boundaries[-1]
    flux_new[mask] += shifts[-1]

    return flux_new


# --------------------------------------------------
# Main
# --------------------------------------------------
segment_shifts = read_segment_shifts(shift_file)
boundaries = read_wavelength_boundaries(wave_file)

fits_files = sorted(glob.glob(input_pattern))
if not fits_files:
    raise FileNotFoundError(f"No FITS files found in pattern:\n{input_pattern}")

print(f"Found {len(fits_files)} FITS files")
print(f"Found {len(segment_shifts)} segment offsets")
print(f"Found {len(boundaries)} wavelength boundaries")

for i, infile in enumerate(fits_files, 1):
    base = os.path.basename(infile)
    outfile = os.path.join(output_dir, base)

    data = fits.getdata(infile)
    header = fits.getheader(infile)

    wave = np.array(data[0], dtype=float)
    flux = np.array(data[1], dtype=float)

    flux_fixed = apply_segment_offsets(wave, flux, boundaries, segment_shifts)

    new_data = np.array(data, copy=True)
    new_data[1] = flux_fixed

    fits.writeto(outfile, new_data, header=header, overwrite=True)

    print(f"[{i:4d}/{len(fits_files):4d}] saved -> {outfile}")

print("Done.")