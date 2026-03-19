from suppnet import get_suppnet
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob, os, shutil
from tqdm import tqdm

suppnet = get_suppnet(norm_only=False)

def read_ADP(specfile):
    '''
    takes eso espresso spectrum as input [change it to your specfile format]
    returns wave[in air], flux[in units of erg/cm^2/s/AA]
    '''
    data = fits.getdata(specfile)
    return data[0][5], data[0][1]

def write_espresso(infile, norm_flux, continuum_error, outfile):
    hdul_infile = fits.open(infile)
    hdul_new = fits.HDUList()
    prim_header = hdul_infile[0].header.copy()    
    wave = hdul_infile[1].data[0][5]
    flux = hdul_infile[1].data[0][1]    
    hdul_new.append(fits.PrimaryHDU(data=np.vstack((wave, flux, norm_flux, continuum_error)), header=prim_header))
    hdul_new.writeto(outfile, overwrite=True)

fits_files = glob.glob('/home/c4011027/PhD_stuff/ESO_proposals/data_zeta_oph3n/*.fits')
for fits_file in tqdm(fits_files):
    wave, flux = read_ADP(fits_file)
    continuum, continuum_error, segmentation, segmentation_error = suppnet.normalize(wave, flux)
    output_file = '/home/c4011027/PhD_stuff/ESO_proposals/data_zeta_oph3n/norm/'+f'{fits_file.split("/")[-1].replace('.fits', '_supnorm.fits')}'
    write_espresso(fits_file, flux/continuum, continuum_error, output_file)