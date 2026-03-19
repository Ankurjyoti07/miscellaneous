import numpy as np
import sys, glob, gzip, os
from astropy.io import fits
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from math import sin, pi


model_files = glob.glob('/home/c4011027/PhD_stuff/ESO_proposals/tlusty_Omodels/Gvispec/*.gz')
vis_7_files = {}
vis_17_files = {}

for file_name in model_files:
    name_without_extension = file_name.split('/')[-1].split('.')[0] # Extracting the file name without extension
    
    if 'vis.7' in file_name:
        vis_7_files[name_without_extension] = file_name
    elif 'vis.17' in file_name:
        vis_17_files[name_without_extension] = file_name
    else: continue
    
vis_7_files_list = list(vis_7_files.values())
vis_17_files_sorted = {key: vis_17_files[key] for key in vis_7_files.keys()}
vis_17_files_list = list(vis_17_files_sorted.values())


def read_data(file_path):
    with gzip.open(file_path, 'rt') as file:
        df = pd.read_csv(file, delim_whitespace=True, names=['wave', 'flux'])
    return df

def normalize_flux(df_17, df_7):
    normalized_flux = df_7['flux'] / df_17['flux']
    return normalized_flux

def interpolate_flux(df_17, df_7):
    interp_func = interp1d(df_17['wave'], df_17['flux'], kind='linear', fill_value='extrapolate')
    interpolated_flux = interp_func(df_7['wave'])
    return interpolated_flux

input_directory = "/home/c4011027/PhD_stuff/ESO_proposals/tlusty_Omodels/Gvispec/"
output_directory = "/home/c4011027/PhD_stuff/ESO_proposals/tlusty_Omodels/normalized_data/"
os.makedirs(output_directory, exist_ok=True)

for file_17, file_7 in zip(vis_17_files_list, vis_7_files_list):
    df_17 = read_data(file_17)
    df_7 = read_data(file_7)
    
    #debug stuff
    #print("Columns of df_7:", df_7.columns)
    #print("Columns of interpolated_df_17:", interpolated_df_17.columns)

    interpolated_flux = interpolate_flux(df_17, df_7)
    interpolated_df_17 = pd.DataFrame({'wave': df_7['wave'], 'flux': interpolated_flux})
    normalized_flux = normalize_flux(interpolated_df_17, df_7)
    normalized_df = pd.DataFrame({'wave': df_7['wave'], 'normalized_flux': normalized_flux})
    file_name_without_extension = os.path.splitext(os.path.basename(file_17))[0]
    output_file_path = os.path.join(output_directory, f"{file_name_without_extension}_normalized.csv")
    normalized_df.to_csv(output_file_path, index=False)