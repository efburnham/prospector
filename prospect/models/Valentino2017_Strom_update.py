#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the code I received from Sean Johnson on 3/16
# doublet line ratios added

import numpy as np
import extinction
import math, random

__all__ = ['predict_L_Hemis', 
           'predict_L_OII_tot', 'predict_L_OIII5007', 
           'predict_L_NeIII3870', 'predict_L_NII6585', 
           'predict_L_SII_tot', 'find_nearest']

# ---------------------------------------------------------------- #
def find_nearest(array,value):
    """
    - much faster than argmin
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

# ---------------------------------------------------------------- #
def predict_L_Hemis(logSFR, Av_stellar):
    """
    Predict the Halpha to Hzeta luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.005 dex
    and standard deviation of 0.007 dex.
    return: log of the Halpha luminosity in erg/s
    """
    out = {}
    out['wave'] = np.array([6564.61, 4862.68, 4341.68, 4102.89, 3971.19, 3890.17])

    f = 0.76 # stellar to nebular Av ratio.
    Av_nebular = Av_stellar/f
    ext_mags = extinction.calzetti00(out['wave'], Av_nebular, 4.05)
    
    # -- Halpha -- #
    logL_Ha_int = logSFR - np.log10(7.9e-42) # K98
    logL_Ha = logL_Ha_int - np.log10(2.512**ext_mags[0])
    
    # -- Hbeta -- #
    logL_Hb_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) # K98 and Case B
    logL_Hb = logL_Hb_int - np.log10(2.512**ext_mags[1])   
    
    # -- Hgamma -- K98 and Case B #
    logL_Hg_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.466) 
    logL_Hg = logL_Hg_int - np.log10(2.512**ext_mags[2])
    
    # -- Hdelta -- K98 and Case B #
    logL_Hd_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.256)   
    logL_Hd = logL_Hd_int - np.log10(2.512**ext_mags[3])
    
    # -- Hepsilon luminosity given logSFR/(Msun/yr) and stellar Av.
    #return: log of the Hdelta luminosity in erg/s
    # K98 and Case B
    logL_He_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.158)   
    logL_He = logL_He_int - np.log10(2.512**ext_mags[4])
   
    # -- Predict the Hzeta luminosity given logSFR/(Msun/yr) and stellar Av.
    # -- return: log of the Hdelta luminosity in erg/s   
    # K98 and Case B
    logL_Hz_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.105)
    logL_Hz = logL_Hz_int - np.log10(2.512**ext_mags[5])
    
    out['logL'] = np.array([logL_Ha, logL_Hb, logL_Hg, logL_Hd, logL_He, logL_Hz])
    return out['logL']
   
    
def predict_L_OII_tot(logSFR, Av_stellar):
    """
    Predict the [O II] luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.058 dex
    and standard deviation of 0.043 dex.
    return: log of the [O II] luminosity in erg/s
    """
    f = 0.76 # stellar to nebular Av ratio.
    logL_OII_int = logSFR - np.log10(7.9e-42) # K04 actually just K98
    Av_nebular = Av_stellar/f
    ext_mags = extinction.calzetti00(np.array([3728.0]), Av_nebular, 4.05)
   
    logL_OII = logL_OII_int - np.log10(2.512**ext_mags[0])
   
    # -- 3729/3726=1.5(low density limit), 0.35(high density limit)
    # -- https://www.ucolick.org/~simard/phd/root/node21.html
    #lineratio = np.random.uniform(0.35, 1.5)
    # -- Sanders et al. 2016
    lineratio = random.uniform(0.3839, 1.4558)
    logL_OII_3729 = np.log10(10**logL_OII*(lineratio/(1.+lineratio)))
    logL_OII_3726 = np.log10(10**logL_OII*(1./(1.+lineratio)))
   
    return logL_OII, logL_OII_3729, logL_OII_3726
   

    
def predict_L_OIII5007(logSFR, Av_stellar, logMstar):
    """
    Predict the [O III] luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.025 dex
    and standard deviation of 0.019 dex.
    return: log of the [O III] luminosity in erg/s
    """
    f = 0.76 # stellar to nebular Av ratio.
    logL_Hb = predict_L_Hb(logSFR, Av_stellar)
    logOIII_Hbeta = 0.3 + 0.48*math.atan(-(logMstar - 10.28))
   
    logL_OIII = logL_Hb + logOIII_Hbeta
   
    # -- OIII5007/OIII4959 = 2.98 (Storey and Zeippen 2000)
    # -- https://academic.oup.com/mnras/article/312/4/813/1017382
    lineratio = 2.98
    logL_OIII_4959 = np.log10(10**logL_OIII/lineratio)  
    
    return logL_OIII, logL_OIII_4959
   

def predict_L_NeIII3870(logSFR, Av_stellar, logMstar):
    """
    Predict the [Ne III] 3870 luminosity given logSFR/(Msun/yr) and stellar Av and stellar mass.
    return: log of the [Ne III] luminosity in erg/s
    """
    f = 0.76 # stellar to nebular Av ratio.
    Av_nebular = Av_stellar/f
   
    # Get Hbeta assuming no dust
    logL_Hb_int = predict_L_Hb(logSFR, 0.0)
   
    # Get [O III] assuming no dust
    logOIII_Hbeta = 0.3 + 0.48*math.atan(-(logMstar - 10.28))
    logL_OIII_int = logL_Hb_int + logOIII_Hbeta
   
    # Assuming [Ne III] is 10x weaker than [O III]
    logL_NeIII_int = logL_OIII_int - 1.0
   
    # Apply dust
    ext_mags = extinction.calzetti00(np.array([3869.85]), Av_nebular, 4.05)
    logL_NeIII = logL_NeIII_int - np.log10(2.512**ext_mags[0])
   
    return logL_NeIII
   
   
   
def predict_L_NII6585(logSFR, Av_stellar, logMstar):
    """Best fit relation from Strom+2017"""

    logL_Ha = predict_L_Ha(logSFR, Av_stellar)
    logL_Hb = predict_L_Hb(logSFR, Av_stellar)
    logL_OIII = predict_L_OIII5007(logSFR, Av_stellar, logMstar)[0]
   
    logL_NII  = 0.61/((logL_OIII - logL_Hb) - 1.12) + 0.22 + logL_Ha
   
    # -- NII6583/NII6548 = 3.05 (Storey and Zeippen 2000)
    # -- https://academic.oup.com/mnras/article/312/4/813/1017382
    lineratio = 3.05
    logL_NII_6548 = np.log10(10**logL_NII/lineratio)   
   
    return logL_NII, logL_NII_6548
   

def predict_L_SII_tot(logSFR, Av_stellar, logMstar):
    """Best fit relation from Strom+2017"""

    logL_Ha = predict_L_Ha(logSFR, Av_stellar)
    logL_Hb = predict_L_Hb(logSFR, Av_stellar)
    logL_OIII = predict_L_OIII5007(logSFR, Av_stellar, logMstar)[0]
   
    logL_SII  = 0.72/((logL_OIII - logL_Hb) - 1.15) + 0.53 + logL_Ha
   
    # -- Sanders et al. 2016, Table 1
    lineratio = random.uniform( 0.4375, 1.4484)
    logL_SII_6716 = np.log10(10**logL_SII*(lineratio/(1.+lineratio)))
    logL_SII_6731 = np.log10(10**logL_SII*(1./(1.+lineratio)))

    return logL_SII, logL_SII_6716, logL_SII_6731


def predict_L_Ha(logSFR, Av_stellar):
    """
    Predict the Halpha luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.005 dex
    and standard deviation of 0.007 dex.
    return: log of the Halpha luminosity in erg/s
    """
    # Set stellar vs nebular extinction f
    f = 0.76 # stellar to nebular Av ratio.
      
    logL_Ha_int = logSFR - np.log10(7.9e-42) # K98
    Av_nebular = Av_stellar/f
    ext_mags = extinction.calzetti00(np.array([6564.61]), Av_nebular, 4.05)
   
    logL_Ha = logL_Ha_int - np.log10(2.512**ext_mags[0])
   
    return logL_Ha
   
def predict_L_Hb(logSFR, Av_stellar):
    """
    Predict the Hbeta luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.024 dex
    and standard deviation of 0.026 dex.
    return: log of the Hbeta luminosity in erg/s
    """
   
    logL_Hb_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) # K98 and Case B
    f = 0.76
    Av_nebular = Av_stellar/f
    ext_mags = extinction.calzetti00(np.array([4862.68]), Av_nebular, 4.05)
   
    logL_Hb = logL_Hb_int - np.log10(2.512**ext_mags[0])
   
    return logL_Hb
   
   
def predict_L_Hg(logSFR, Av_stellar):
    """
    Predict the Hgamma luminosity given logSFR/(Msun/yr) and stellar Av.
    return: log of the Hgamma luminosity in erg/s
    """
   
    # K98 and Case B
    f = 0.76
    logL_Hg_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.466)
    Av_nebular = Av_stellar/f
    ext_mags = extinction.calzetti00(np.array([4341.68]), Av_nebular, 4.05)
   
    logL_Hg = logL_Hg_int - np.log10(2.512**ext_mags[0])
   
    return logL_Hg
   

def predict_L_Hd(logSFR, Av_stellar):
    """
    Predict the Hdelta luminosity given logSFR/(Msun/yr) and stellar Av.
    return: log of the Hdelta luminosity in erg/s
    """
   
    # K98 and Case B
    f = 0.76
    logL_Hd_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.256)
    Av_nebular = Av_stellar/f
    ext_mags = extinction.calzetti00(np.array([4102.89]), Av_nebular, 4.05)
   
    logL_Hd = logL_Hd_int - np.log10(2.512**ext_mags[0])
   
    return logL_Hd
   

def predict_L_He(logSFR, Av_stellar):
    """
    Predict the Hepsilon luminosity given logSFR/(Msun/yr) and stellar Av.
    return: log of the Hdelta luminosity in erg/s
    """
    # K98 and Case B
    f = 0.76
    logL_He_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.158)
    Av_nebular = Av_stellar/f
    ext_mags = extinction.calzetti00(np.array([3971.19]), Av_nebular, 4.05)
   
    logL_He = logL_He_int - np.log10(2.512**ext_mags[0])
   
    return logL_He
   
    
def predict_L_Hz(logSFR, Av_stellar):
   """
   Predict the Hzeta luminosity given logSFR/(Msun/yr) and stellar Av.
   return: log of the Hdelta luminosity in erg/s
   """
   
   # K98 and Case B
   f = 0.76
   logL_Hz_int = logSFR - np.log10(7.9e-42) - np.log10(2.86) + np.log10(0.105)
   Av_nebular = Av_stellar/f
   ext_mags = extinction.calzetti00(np.array([3890.17]), Av_nebular, 4.05)
   
   logL_Hz = logL_Hz_int - np.log10(2.512**ext_mags[0])
   
   return logL_Hz
   
