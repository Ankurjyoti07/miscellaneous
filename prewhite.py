import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/home/c4011027/PhD_stuff/ESO_proposals/prologs/rv_results')
base_dir = "/home/c4011027/PhD_stuff/ESO_proposals/manual_normalization/norm/"

#list of corrupted filenames (ending parts)
corrupted_files = [
    "ADP.2024-07-05T12:11:00.363_norm.fits", "ADP.2024-07-05T12:11:00.351_norm.fits",
    "ADP.2024-07-05T12:11:00.372_norm.fits", "ADP.2024-07-05T13:02:51.727_norm.fits",
    "ADP.2024-07-05T13:02:51.916_norm.fits", "ADP.2024-07-05T13:02:51.928_norm.fits",
    "ADP.2024-07-05T14:00:36.226_norm.fits", "ADP.2024-07-05T14:00:36.244_norm.fits",
    "ADP.2024-07-05T14:00:36.277_norm.fits", "ADP.2024-07-05T14:00:36.310_norm.fits",
    "ADP.2024-07-05T15:00:39.032_norm.fits", "ADP.2024-07-05T15:00:39.041_norm.fits",
    "ADP.2024-07-05T15:00:39.050_norm.fits"
]

corrupted_files_full_path = [base_dir + file for file in corrupted_files]
data = data[~data['Filename'].isin(corrupted_files_full_path)]

mjd = data['MJD-OBS']
rv_4471 = data['RV_4471.48_Å']
rv_error_4471 = data['RV_error_4471.48_Å']

fap_level = 0.0001
snr_threshold = 5
oversampling_factor = 40 #resolution in power spectrum
current_rv = rv_4471
nyquist_frequency = 625
nyquist_period = 1 / nyquist_frequency
max_period_window = 3.8  #observation window

print(f"Nyquist Frequency: {nyquist_frequency:.4f} cycles/day")
print(f"Nyquist Period: {nyquist_period:.4f} days")
print(f"Period Search Range: {nyquist_period:.4f} to {max_period_window:.4f} days")

extracted_periods = []

def sine_wave_multi_period(t, *params):
    """Sine wave model with multiple fixed periods. Amplitude and phase are optimized for each period."""
    result = np.zeros_like(t)
    for i, period in enumerate(extracted_periods):
        amplitude = params[2 * i]
        phase = params[2 * i + 1]
        result += amplitude * np.sin(2 * np.pi * t / period + phase)
    return result

while True:
    frequency, power = LombScargle(mjd, current_rv, rv_error_4471, fit_mean=True).autopower(samples_per_peak=oversampling_factor)
    fap_threshold_power = LombScargle(mjd, current_rv, rv_error_4471).false_alarm_level(fap_level)
    period = 1 / frequency

    #peaks in power spectrum with scipy find_peaks
    peaks, _ = find_peaks(power)
    peak_periods = period[peaks]
    peak_powers = power[peaks]
    significant_peaks = peak_powers > fap_threshold_power
    significant_peak_periods = peak_periods[significant_peaks]
    significant_peak_powers = peak_powers[significant_peaks]
    restricted_indices = (significant_peak_periods >= nyquist_period) & (significant_peak_periods <= max_period_window)
    restricted_peak_periods = significant_peak_periods[restricted_indices]
    restricted_peak_powers = significant_peak_powers[restricted_indices]

    #break if no peaks found
    if len(restricted_peak_powers) == 0:
        print("No significant peaks in the restricted range. Stopping.")
        break

    max_power_index = np.argmax(restricted_peak_powers)
    max_power = restricted_peak_powers[max_power_index]
    max_period = restricted_peak_periods[max_power_index]
    snr = max_power / fap_threshold_power

    if snr < snr_threshold:
        print("No significant peaks remaining. Stopping.")
        break

    print(f"Detected period: {max_period:.4f} days with SNR: {snr:.2f}")

    extracted_periods.append(max_period)

    #create initial guess for detrending
    initial_guess = []
    for periods in extracted_periods:
        initial_guess.extend([np.std(current_rv), 0])  #[amplitude, phase] for each period

    params, _ = curve_fit(sine_wave_multi_period, mjd, current_rv, p0=initial_guess)
    interpolated_rv = sine_wave_multi_period(mjd, *params)
    detrended_rv = current_rv - interpolated_rv
    current_rv = detrended_rv     #set current RV to detrended RV for next itertation

    plt.figure(figsize=(10, 6))
    plt.plot(period, power, label="Lomb-Scargle Power")
    plt.plot(restricted_peak_periods, restricted_peak_powers, "x", color="orange", label="Significant Peaks")
    plt.axhline(y=fap_threshold_power, color='r', linestyle='--', label="FAP = 0.0001")
    plt.xlabel("Period (days)")
    plt.ylabel("Power")
    plt.title("Lomb-Scargle Periodogram with Oversampling (Iteration)")
    plt.legend()
    plt.xscale('log')
    plt.show()

