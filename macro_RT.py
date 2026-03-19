import numpy as np
from scipy.special import erf                               # Error function
from scipy.signal import fftconvolve 

def macro_RT(xdata, ydata, zeta_r, zeta_t, aat):
    c = 299792458 #in meters, set vmacro in meters too
    sq_pi = np.sqrt(np.pi)
    lambda0 = np.median(xdata)
    xspacing = xdata[1] - xdata[0]
    
    aar = 1- aat
    
    mr = zeta_r * lambda0 / c
    mt = zeta_t * lambda0 / c 

    ccr = 2 * aar / (sq_pi * mr)
    cct = 2 * aat / (sq_pi * mt)

    px = np.arange(-len(xdata) / 2, len(xdata) / 2 + 1) * xspacing
    pxmr = abs(px) / mr
    pxmt = abs(px) / mt
    
    profile_r = ccr * (np.exp(-pxmr ** 2) + sq_pi * pxmr * (erf(pxmr) - 1.0))
    profile_t = cct * (np.exp(-pxmt ** 2) + sq_pi * pxmt * (erf(pxmt) - 1.0))

    profile = profile_r + profile_t
    
    before = ydata[int(-profile.size / 2 + 1):]
    after = ydata[:int(profile.size / 2 +1)] #add one to fix size mismatch
    extended = np.r_[before, ydata, after]

    first = xdata[0] - float(int(profile.size / 2.0 + 0.5)) * xspacing
    last = xdata[-1] + float(int(profile.size / 2.0 + 0.5)) * xspacing
    
    x2 = np.linspace(first, last, extended.size)  #newdata x array ==> handles edge effects

    conv_mode = "valid"

    newydata = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)

    return newydata
    
    
    
    
