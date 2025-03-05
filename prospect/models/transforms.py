#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""transforms.py -- This module contains parameter transformations that may be
useful to transform from parameters that are easier to _sample_ in to the
parameters required for building SED models.

They can be used as ``"depends_on"`` entries in parameter specifications.
"""

import numpy as np
from ..sources.constants import cosmo
import warnings
#from gp_sfh import *
#import gp_sfh_kernels


__all__ = ["stellar_logzsol", "delogify_mass",
           "tburst_from_fage", "tage_from_tuniv", "zred_to_agebins",
           "dustratio_to_dust1",
           "logsfr_ratios_to_masses", "logsfr_ratios_to_sfrs",
           "logsfr_ratios_to_masses_flex", "logsfr_ratios_to_agebins",
           "zfrac_to_masses", "zfrac_to_sfrac", "zfrac_to_sfr", "masses_to_zfrac",
           "sfratio_to_sfr", "sfratio_to_mass", 
           #"get_sfr_covar", "sfr_covar_to_sfr_ratio_covar",
           "zred_to_agebins_pbeta",
           "zredmassmet_to_zred", "zredmassmet_to_logmass", "zredmassmet_to_mass", "zredmassmet_to_logzsol",
           "nzsfh_to_zred", "nzsfh_to_logmass", "nzsfh_to_mass", "nzsfh_to_logzsol", "nzsfh_to_logsfr_ratios",
           "create_masses_ssSFH", "create_agebins_ssSFH",
          ]


# --------------------------------------
# --- Basic Convenience Transforms ---
# --------------------------------------

def stellar_logzsol(logzsol=0.0, **extras):
    """Simple function that takes an argument list and returns the value of the
    `logzsol` argument (i.e. the stellar metallicity)

    Parameters
    ----------
    logzsol : float
        FSPS stellar metaliicity parameter.

    Returns
    -------
    logzsol: float
        The same.
    """
    return logzsol


def delogify_mass(logmass=0.0, **extras):
    """Simple function that takes an argument list including a `logmass`
    parameter and returns the corresponding linear mass.

    Parameters
    ----------
    logmass : float
        The log10(mass)

    Returns
    -------
    mass : float
        The mass in linear units
    """
    return 10**logmass


def total_mass(mass=0.0, **extras):
    """Simple function that takes an argument list uncluding a `mass`
    parameter and returns the corresponding total mass.

    Parameters
    ----------
    mass : ndarray of shape ``(N_bins,)``
        Vector of masses in bins

    Returns
    -------
    total_mass : float
        Total mass in linear units
    """
    return mass.sum()

# --------------------------------------
# Fancier transforms
# --------------------------------------

def tburst_from_fage(tage=0.0, fage_burst=0.0, **extras):
    """This function transfroms from a fractional age of a burst to an absolute
    age.  With this transformation one can sample in ``fage_burst`` without
    worry about the case ``tburst`` > ``tage``.

    Parameters
    ----------
    tage : float, Gyr
        The age of the host galaxy.

    fage_burst : float between 0 and 1
        The fraction of the host age at which the burst occurred.

    Returns
    -------
    tburst : float, Gyr
        The age of the host when the burst occurred (i.e. the FSPS ``tburst``
        parameter)
    """
    return tage * fage_burst


def tage_from_tuniv(zred=0.0, tage_tuniv=1.0, **extras):
    """This function calculates a galaxy age from the age of the universe at
    ``zred`` and the age given as a fraction of the age of the universe.  This
    allows for both ``zred`` and ``tage`` parameters without ``tage`` exceeding
    the age of the universe.

    Parameters
    ----------
    zred : float
        Cosmological redshift.

    tage_tuniv : float between 0 and 1
        The ratio of ``tage`` to the age of the universe at ``zred``.

    Returns
    -------
    tage : float
        The stellar population age, in Gyr
    """
    tuniv = cosmo.age(zred).value
    tage = tage_tuniv * tuniv
    return tage


def zred_to_agebins(zred=0.0, agebins=[], **extras):
    """Set the nonparameteric SFH age bins depending on the age of the universe
    at ``zred``. The first bin is not altered and the last bin is always 15% of
    the upper edge of the oldest bin, but the intervening bins are evenly
    spaced in log(age).

    Parameters
    ----------
    zred : float
        Cosmological redshift.  This sets the age of the universe.

    agebins :  ndarray of shape ``(nbin, 2)``
        The SFH bin edges in log10(years).

    Returns
    -------
    agebins : ndarray of shape ``(nbin, 2)``
        The new SFH bin edges.
    """
    tuniv = cosmo.age(zred).value * 1e9
    tbinmax = tuniv * 0.85
    ncomp = len(agebins)
    agelims = list(agebins[0]) + np.linspace(agebins[1][1], np.log10(tbinmax), ncomp-2).tolist() + [np.log10(tuniv)]
    return np.array([agelims[:-1], agelims[1:]]).T


def dustratio_to_dust1(dust2=0.0, dust_ratio=0.0, **extras):
    """Set the value of dust1 from the value of dust2 and dust_ratio

    Parameters
    ----------
    dust2 : float
        The diffuse dust V-band optical depth (the FSPS ``dust2`` parameter.)

    dust_ratio : float
        The ratio of the extra optical depth towards young stars to the diffuse
        optical depth affecting all stars.

    Returns
    -------
    dust1 : float
        The extra optical depth towards young stars (the FSPS ``dust1``
        parameter.)
    """
    return dust2 * dust_ratio

# --------------------------------------
# --- Transforms for the continuity non-parametric SFHs used in (Leja et al. 2018) ---
# --------------------------------------

def logsfr_ratios_to_masses(logmass=None, logsfr_ratios=None, agebins=None,
                            **extras):
    """This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
    log10(\Sum_i M_i) to values of M_i.  j=0 is the most recent bin in lookback
    time.
    """
    nbins = agebins.shape[0]
    sratios = 10**np.clip(logsfr_ratios, -10, 10)  # numerical issues...
    #sratios = 10**np.clip(logsfr_ratios, -100, 100)  # numerical issues...
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    coeffs = np.array([ (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
                        for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()

    return m1 * coeffs


def logsfr_ratios_to_sfrs(logmass=None, logsfr_ratios=None, agebins=None, **extras):
    """Convenience function
    """
    masses = logsfr_ratios_to_masses(logmass=logmass, logsfr_ratios=logsfr_ratios,
                                     agebins=agebins)
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])

    return masses / dt

# --------------------------------------
# --- Transforms for the flexible agebin continuity non-parametric SFHs used in (Leja et al. 2018) ---
# --------------------------------------

def logsfr_ratios_to_masses_flex(logmass=None, logsfr_ratios=None,
                                 logsfr_ratio_young=None, logsfr_ratio_old=None,
                                 **extras):
    logsfr_ratio_young = np.clip(logsfr_ratio_young, -10, 10)
    logsfr_ratio_old = np.clip(logsfr_ratio_old, -10, 10)
    #logsfr_ratio_young = np.clip(logsfr_ratio_young, -100, 100)
    #logsfr_ratio_old = np.clip(logsfr_ratio_old, -100, 100)

    abins = logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios, **extras)

    nbins = abins.shape[0] - 2
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    dtyoung, dt1 = (10**abins[:2, 1] - 10**abins[:2, 0])
    dtn, dtold = (10**abins[-2:, 1] - 10**abins[-2:, 0])
    mbin = (10**logmass) / (syoung*dtyoung/dt1 + sold*dtold/dtn + nbins)
    myoung = syoung * mbin * dtyoung / dt1
    mold = sold * mbin * dtold/dtn
    n_masses = np.full(nbins, mbin)

    return np.array(myoung.tolist() + n_masses.tolist() + mold.tolist())


def logsfr_ratios_to_agebins(logsfr_ratios=None, agebins=None, **extras):
    """This transforms from SFR ratios to agebins by assuming a constant amount
    of mass forms in each bin agebins = np.array([NBINS,2])

    use equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin
    """

    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios, -10, 10)
    #logsfr_ratios = np.clip(logsfr_ratios, -100, 100)

    # calculate delta(t) for oldest, youngest bins (fixed)
    lower_time = (10**agebins[0, 1] - 10**agebins[0, 0])
    upper_time = (10**agebins[-1, 1] - 10**agebins[-1, 0])
    tflex = (10**agebins[-1,-1] - upper_time - lower_time)

    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tflex / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))

    # translate into agelims vector (time bin edges)
    agelims = [1, lower_time, dt1+lower_time]
    for i in range(n_ratio):
        agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
    #agelims += [tuniv[0]]
    agelims += [10**agebins[-1, 1]]
    agebins = np.log10([agelims[:-1], agelims[1:]]).T

    return agebins

# --------------------------------------
# -- Transforms for the fixed+flexible non-parametric SFHs used in (Suess et al. 2021) --
# --------------------------------------
def logsfr_ratios_to_masses_psb(logmass=None, logsfr_ratios=None,
                                 logsfr_ratio_young=None, logsfr_ratio_old=None,
                                 tlast=None, tflex=None, nflex=None, nfixed=None,
                                 agebins=None, **extras):
    """This is a modified version of logsfr_ratios_to_masses_flex above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.

    The major difference between this and the transform above is that
    logsfr_ratio_old is a vector.
    """

    # clip for numerical stability
    nflex = nflex[0]; nfixed = nfixed[0]
    logsfr_ratio_young = np.clip(logsfr_ratio_young[0], -7, 7)
    logsfr_ratio_old = np.clip(logsfr_ratio_old, -7, 7)
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    sratios = 10.**np.clip(logsfr_ratios, -7, 7) # numerical issues...

    # get agebins
    abins = psb_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios,
            agebins=agebins, tlast=tlast, tflex=tflex, nflex=nflex, nfixed=nfixed, **extras)

    # get find mass in each bin
    dtyoung, dt1 = (10**abins[:2, 1] - 10**abins[:2, 0])
    dtold = 10**abins[-nfixed-1:, 1] - 10**abins[-nfixed-1:, 0]
    old_factor = np.zeros(nfixed)
    for i in range(nfixed):
        old_factor[i] = (1. / np.prod(sold[:i+1]) * np.prod(dtold[1:i+2]) / np.prod(dtold[:i+1]))
    mbin = 10**logmass / (syoung*dtyoung/dt1 + np.sum(old_factor) + nflex)
    myoung = syoung * mbin * dtyoung / dt1
    mold = mbin * old_factor
    n_masses = np.full(nflex, mbin)

    return np.array(myoung.tolist() + n_masses.tolist() + mold.tolist())


def psb_logsfr_ratios_to_agebins(logsfr_ratios=None, agebins=None,
                                 tlast=None, tflex=None, nflex=None, nfixed=None, **extras):
    """This is a modified version of logsfr_ratios_to_agebins above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.

    For the flexible bins, we again use the equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin

    """

    # dumb way to de-arrayify values...
    tlast = tlast[0]; tflex = tflex[0]
    try: nflex = nflex[0]
    except IndexError: pass
    try: nfixed = nfixed[0]
    except IndexError: pass

    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios, -7, 7)

    # flexible time is t_flex - youngest bin (= tlast, which we fit for)
    # this is also equal to tuniv - upper_time - lower_time
    tf = (tflex - tlast) * 1e9

    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tf / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))

    # translate into agelims vector (time bin edges)
    agelims = [1, (tlast*1e9), dt1+(tlast*1e9)]
    for i in range(n_ratio):
        agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
    agelims += list(10**agebins[-nfixed:,1])
    abins = np.log10([agelims[:-1], agelims[1:]]).T

    return abins

# --------------------------------------
# --- Transforms for Dirichlet non-parametric SFH used in (Leja et al. 2017) ---
# --------------------------------------

def zfrac_to_sfrac(z_fraction=None, **extras):
    """This transforms from independent dimensionless `z` variables to sfr
    fractions. The transformation is such that sfr fractions are drawn from a
    Dirichlet prior.  See Betancourt et al. 2010 and Leja et al. 2017

    Parameters
    ----------
    z_fraction : ndarray of shape ``(Nbins-1,)``
        latent variables drawn from a specific set of Beta distributions. (see
        Betancourt 2010)

    Returns
    -------
    sfrac : ndarray of shape ``(Nbins,)``
        The star formation fractions (See Leja et al. 2017 for definition).
    """
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    if (sfr_fraction < 0).any():
        idx = sfr_fraction < 0
        if np.isclose(sfr_fraction[idx],0,rtol=1e-8):
            sfr_fraction[idx] = 0.0
        else:
            raise ValueError('The input z_fractions are returning negative masses!')


    return sfr_fraction


def zfrac_to_masses(total_mass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to sfr
    fractions and then to bin mass fractions. The transformation is such that
    sfr fractions are drawn from a Dirichlet prior.  See Betancourt et al. 2010
    and Leja et al. 2017

    Parameters
    ----------
    total_mass : float
        The total mass formed over all bins in the SFH.

    z_fraction : ndarray of shape ``(Nbins-1,)``
        latent variables drawn from a specific set of Beta distributions. (see
        Betancourt 2010)

    Returns
    -------
    masses : ndarray of shape ``(Nbins,)``
        The stellar mass formed in each age bin.
    """
    # sfr fractions
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    if (mass_fraction < 0).any():
        idx = mass_fraction < 0
        if np.isclose(mass_fraction[idx],0,rtol=1e-8):
            mass_fraction[idx] = 0.0
        else:
            raise ValueError('The input z_fractions are returning negative masses!')

    masses = total_mass * mass_fraction
    return masses

    # -- version of above for arrays of fractions --
    #zf = np.atleast_2d(z_fraction)
    #shape = list(zf.shape)
    #shape[-1] += 1
    #sfr_fraction = np.zeros(shape)
    #sfr_fraction[..., 0] = 1.0 - z_fraction[..., 0]
    #for i in range(1, shape[-1]-1):
    #   sfr_fraction[..., i] = (np.prod(z_fraction[..., :i], axis=-1) *
    #                            (1.0 - z_fraction[...,i]))
    #sfr_fraction[..., -1] = 1 - np.sum(sfr_fraction[..., :-1], axis=-1)
    #sfr_fraction = np.squeeze(sfr_fraction)
    #
    # convert to mass fractions
    #time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
    #sfr_fraction *= np.array(time_per_bin)
    #mtot = np.atleast_1d(sfr_fraction.sum(axis=-1))
    #mass_fraction = sfr_fraction / mtot[:, None]
    #
    #masses = np.atleast_2d(total_mass) * mass_fraction.T
    #return masses.T


def zfrac_to_sfr(total_mass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to SFRs.

    :returns sfrs:
        The SFR in each age bin (msun/yr).
    """
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    masses = zfrac_to_masses(total_mass, z_fraction, agebins)
    return masses / time_per_bin


def masses_to_zfrac(mass=None, agebins=None, **extras):
    """The inverse of :py:func:`zfrac_to_masses`, for setting mock parameters
    based on mock bin masses.

    Returns
    -------
    total_mass : float
        The total mass

    zfrac : ndarray of shape ``(Nbins-1,)``
        latent variables drawn from a specific set of Beta distributions. (see
        Betancourt 2010) related to the fraction of mass formed in each bin.
    """
    total_mass = mass.sum()
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    sfr_fraction = mass / time_per_bin
    sfr_fraction /= sfr_fraction.sum()

    z_fraction = np.zeros(len(sfr_fraction) - 1)
    z_fraction[0] = 1 - sfr_fraction[0]
    for i in range(1, len(z_fraction)):
        z_fraction[i] = 1.0 - sfr_fraction[i] / np.prod(z_fraction[:i])

    return total_mass, z_fraction


# --------------------------------------
# --- Transforms for SFR ratio based nonparameteric SFH ---
# --------------------------------------

def sfratio_to_sfr(sfr_ratio=None, sfr0=None, **extras):
    raise(NotImplementedError)


def sfratio_to_mass(sfr_ratio=None, sfr0=None, agebins=None, **extras):
    raise(NotImplementedError)


# --------------------------------------
# --- Transforms for prospector-beta ---
# --------------------------------------

def zred_to_agebins_pbeta(zred=None, agebins=[], **extras):
    """New agebin scheme, refined so that none of the bins is overly wide when the universe is young.
    
    Parameters
    ----------
    zred : float
        Cosmological redshift.  This sets the age of the universe.
    agebins :  ndarray of shape ``(nbin, 2)``
        The SFH bin edges in log10(years).
    
    Returns
    -------
    agebins : ndarray of shape ``(nbin, 2)``
        The new SFH bin edges.
    """
    amin = 7.1295
    nbins_sfh = len(agebins)
    tuniv = cosmo.age(zred)[0].value*1e9 # because input zred is atleast_1d
    tbinmax = (tuniv*0.9)
    if (zred <= 3.):
        agelims = [0.0,7.47712] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    else:
        agelims = np.linspace(amin,np.log10(tbinmax),nbins_sfh).tolist() + [np.log10(tuniv)]
        agelims[0] = 0
        
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

# separates a theta vector of [zred, mass, met] into individual parameters
# can be used with PhiMet & ZredMassMet
def zredmassmet_to_zred(zredmassmet=None, **extras):
    return zredmassmet[0]

def zredmassmet_to_logmass(zredmassmet=None, **extras):
    return zredmassmet[1]

def zredmassmet_to_mass(zredmassmet=None, **extras):
    return 10**zredmassmet[1]

def zredmassmet_to_logzsol(zredmassmet=None, **extras):
    return zredmassmet[2]

# separates a theta vector of [zred, mass, met, logsfr_ratios] into individual parameters
# can be used with PhiSFH & NzSFH
def nzsfh_to_zred(nzsfh=None, **extras):
    return nzsfh[0]

def nzsfh_to_logmass(nzsfh=None, **extras):
    return nzsfh[1]

def nzsfh_to_mass(nzsfh=None, **extras):
    return 10**nzsfh[1]

def nzsfh_to_logzsol(nzsfh=None, **extras):
    return nzsfh[2]

def nzsfh_to_logsfr_ratios(nzsfh=None, **extras):
    return nzsfh[3:]


# --------------------------------------------
# --- Transforms for simple sinusoidal sfh ---
# --------------------------------------------
    
    
def create_nbins_ssSFH(dt, zred):
    """
    Calculate the number of bins based on the age of the universe and a given time step.

    Args:
        - dt: float, the time step in Myr.
        - zred: float, redshift.

    Returns:
        - nbins: int, the total number of bins calculated based on the age of the universe and the time step.
    """
    t_univ = cosmo.age(zred).value * 1e9 # age of the universe in years
    nbins = 0.85 * t_univ / (dt * 1e6)
    nbins = np.around(nbins, decimals=0).astype(int)
    return nbins + 1

def create_agebins_ssSFH(dt=40, phi=0.0, zred=0, **extras):
    """
    Create age bins based on specified parameters, including bin size and redshift.

    Args:
        - dt: float or array, the size of the age bins in Myrs.
        - phi: float or array, the phase.
        - zred: float or array, the redshift.

    Returns:
        - agebins: np.array or list of np.arrays, the calculated age bins in log space.

    Warnings:
        - If the first bin size is less than 1 Myr, a warning will be issued (fsps may crash if you continue!!!).
    """

    # Ensure inputs are iterable if they are scalars
    if np.isscalar(dt):
        dt = np.array([dt])
    if np.isscalar(phi):
        phi = np.array([phi])
    if np.isscalar(zred):
        zred = np.array([zred])

    agebins_list = []

    for d, p, z in zip(dt, phi, zred):
        dt_yrs = d * 1e6
        nbins = create_nbins_ssSFH(d, z) 
        t_univ = cosmo.age(z).value * 1e9  # Age of the universe in years

        agelims = np.ones((nbins+1,))

        # Determine bin size adjustment based on phi
        perc_size = 1 - p/np.pi if p/np.pi < 1 else 2 - p/np.pi
        first_bin_size = dt_yrs * perc_size  # first bin size in log years

        if first_bin_size < 1e6:
            warnings.warn("Warning... The first bin size for this phi and dt is less than 1 Myr -- `prospector` will throw an error.")
    
        agelims[1] = first_bin_size
        agelims[2:] = first_bin_size + dt_yrs * np.arange(1, nbins)
        agelims[-1] = t_univ  # fixing last age to be the age of the universe at the specified redshift
        agelims[-2] = 0.85 * t_univ  # fixing second to last age to be 85% of the age of the universe at the specified redshift

        log_agelims = np.log10(agelims)
        agebins = np.array([log_agelims[:-1], log_agelims[1:]]).T
        
        agebins_list.append(agebins)    

    return agebins_list[0] if len(agebins_list) == 1 else agebins_list

def apply_powerlaw_to_masses_ssSFH(agebins, masses, alpha, phi,t_crit=500):
        """
        Applies power-law modification to one or more mass arrays using a vectorized approach.
    
        Args:
            - agebins: (N_bins, 2) array, age bin boundaries (log(yrs))
            - masses: (N_bins) array, mass formed in each age bin (Msun)
            - t_crit: critical time in Myr for the power-law application (default: 500 Myr)
        
        Returns:
            - masses: (N_bins) or (N_mass_arrays, N_bins), mass arrays with power-law modification applied
        """
        times = 10 ** np.unique(agebins.flatten())[:-1]  # Times in years on front edge of age bins
        times_fine = np.linspace(np.min(times), np.max(times), int(1e6))  # Fine-grained time array (yrs)
        times_fine_powerlaw = times_fine[times_fine < t_crit * 1e6]  # Times below the critical time in yrs
        
        # interpolation of SFR over fine-grained time steps
        masses_fine = np.interp(times_fine, times, masses)            
        # Apply power-law modification for times < t_crit across all mass arrays
        masses_fine[1:len(times_fine_powerlaw)] *= times_fine_powerlaw[1:] ** alpha # leaving the first bin alone to avoid infinity
        masses_fine[len(times_fine_powerlaw):] *= t_crit ** alpha 
    
        # Interpolate the modified SFRs back to the original sparse time steps for all mass arrays
        masses = np.interp(times, times_fine, masses_fine)

        return masses 


def create_masses_ssSFH(dt=40, sigma=0.3, alpha=0.0, phi=0.0, logmass=10, zred=0.0, SFMS=-8.8, **extras):
    """
    Create mass vector from the ssSFH hyperparameters.

    Args:
        - zred (float, optional): Redshift of the object.
        - logmass (float): log mass of object.
        - dt (float): Size of age bins in Myr.
        - sigma (float): Amplitude of SFH around star forming MS in dex.
        - alpha (float): Power-law index for rising SFH.
        - log_sSFR (float): Log specific star formation rate for the star forming MS in log(yr^-1).
        
    Returns:
        - masses: np.array, the calculated mass values for each age bin, adjusted based on parameters.
    """
    
    nbins = create_nbins_ssSFH(dt, zred)  # Ensure this returns an integer
    agebins = create_agebins_ssSFH(dt, phi, zred)
    
    max_SFR = 10 ** (SFMS + sigma + logmass)
    min_SFR = 10 ** (SFMS - sigma + logmass)
    
    dt_last_bin = np.diff(10 ** agebins)[-1][0]
    max_mass = max_SFR * dt * 1e6
    min_mass = min_SFR * dt * 1e6
    
    masses = np.zeros(nbins)
    
    if phi < np.pi:  # star forming phi
        masses[::2] = max_mass
        masses[1::2] = min_mass
        masses[-1] = min_SFR * dt_last_bin if nbins % 2 == 0 else max_SFR * dt_last_bin
    else:  # quiescent phi
        masses[::2] = min_mass
        masses[1::2] = max_mass
        masses[-1] = max_SFR * dt_last_bin if nbins % 2 == 0 else min_SFR * dt_last_bin
    
    mass_adjust = 1 - phi/np.pi if phi/np.pi < 1 else 2 - phi/np.pi
    masses[0] *= mass_adjust # Adjust the first bin mass by phi
    
    if alpha != 0.0:

        masses = apply_powerlaw_to_masses_ssSFH(agebins,masses,alpha,phi)

    # Ensure masses are non-negative
    masses = np.clip(masses, a_min=0, a_max=None)
    
    return masses