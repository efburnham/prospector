#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""priors.py -- This module contains various objects to be used as priors.
When called these return the ln-prior-probability, and they can also be used to
construct prior transforms (for nested sampling) and can be sampled from.
"""

import numpy as np
import scipy.stats
from scipy.special import erf, erfinv
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, trapezoid


__all__ = ["Prior", "Uniform", "TopHat", "Normal", "MultiVariateNormal", "ClippedNormal",
           "LogNormal", "LogUniform", "Beta",
           "StudentT", "SkewNormal",
           "FastUniform", "FastTruncatedNormal",
           "FastTruncatedEvenStudentTFreeDeg2",
           "FastTruncatedEvenStudentTFreeDeg2Scalar"]


class Prior(object):
    """Encapsulate the priors in an object.  Each prior should have a
    distribution name and optional parameters specifying scale and location
    (e.g. min/max or mean/sigma).  These can be aliased at instantiation using
    the ``parnames`` keyword. When called, the argument should be a variable
    and the object should return the ln-prior-probability of that value.

    .. code-block:: python

        ln_prior_prob = Prior(param=par)(value)

    Should be able to sample from the prior, and to get the gradient of the
    prior at any variable value.  Methods should also be avilable to give a
    useful plotting range and, if there are bounds, to return them.

    Parameters
    ----------
    parnames : sequence of strings
        A list of names of the parameters, used to alias the intrinsic
        parameter names.  This way different instances of the same Prior can
        have different parameter names, in case they are being fit for....
    
    Attributes
    ----------
    params : dictionary
        The values of the parameters describing the prior distribution.
    """

    def __init__(self, parnames=[], name='', **kwargs):
        """Constructor.

        Parameters
        ----------
        parnames : sequence of strings
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

    def __repr__(self):
        argstring = ['{}={}'.format(k, v) for k, v in list(self.params.items())]
        return '{}({})'.format(self.__class__, ",".join(argstring))

    def update(self, **kwargs):
        """Update ``self.params`` values using alias.
        """
        for k in self.prior_params:
            try:
                self.params[k] = kwargs[self.alias[k]]
            except(KeyError):
                pass
        # FIXME: Should add a check for unexpected kwargs.

    def __len__(self):
        """The length is set by the maximum size of any of the prior_params.
        Note that the prior params must therefore be scalar of same length as
        the maximum size of any of the parameters.  This is not checked.
        """
        return max([np.size(self.params.get(k, 1)) for k in self.prior_params])

    def __call__(self, x, **kwargs):
        """Compute the value of the probability desnity function at x and
        return the ln of that.

        Parameters
        ----------
        x : float or sequqnce of float
            Value of the parameter, scalar or iterable of same length as the
            Prior object.

        kwargs : optional
            All extra keyword arguments are used to update the `prior_params`.

        Returns
        -------
        lnp : float or sequqnce of float, same shape as ``x``
            The natural log of the prior probability at ``x``, scalar or ndarray
            of same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
            
        pdf = self.distribution.pdf
        
        try:
            p = pdf(x, *self.args, loc=self.loc, scale=self.scale)
            
        except(ValueError):#, TypeError):
            # Deal with `x` vectors of shape (nsamples, len(prior))
            # for pdfs that don't broadcast nicely.
            p = [pdf(_x, *self.args, loc=self.loc, scale=self.scale)
                 for _x in x]           
            p = np.array(p)

        with np.errstate(invalid='ignore'):
            lnp = np.log(p)
            
        return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.rvs(*self.args, size=len(self),
                                     loc=self.loc, scale=self.scale)

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.ppf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.cdf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def gradient(self, theta):
        raise(NotImplementedError)

    @property
    def loc(self):
        """This should be overridden.
        """
        return 0

    @property
    def scale(self):
        """This should be overridden.
        """
        return 1

    @property
    def args(self):
        return []

    @property
    def range(self):
        raise(NotImplementedError)

    @property
    def bounds(self):
        raise(NotImplementedError)

    def serialize(self):
        raise(NotImplementedError)


class DummyPrior():
    """ A prior used purely from sampling from arbitrary PDF given some data set."""
    def __init__(self,data,parnames=[], name='', **kwargs):
        from scipy.stats import gaussian_kde
        self.kde = gaussian_kde(data)

    def __len__(self):
        return 1

    def sample(self,nsample=1,**kwargs):
        return self.kde.resample(size=nsample)


class Uniform(Prior):
    """A simple uniform prior, described by two parameters

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.uniform

    @property
    def scale(self):
        return self.params['maxi'] - self.params['mini']

    @property
    def loc(self):
        return self.params['mini']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class TopHat(Uniform):
    """Uniform distribution between two bounds, renamed for backwards compatibility
    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """


class Normal(Prior):
    """A simple gaussian prior.


    :param mean:
        Mean of the distribution

    :param sigma:
        Standard deviation of the distribution
    """
    prior_params = ['mean', 'sigma']
    distribution = scipy.stats.norm

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        nsig = 4
        return (self.params['mean'] - nsig * self.params['sigma'],
                self.params['mean'] + nsig * self.params['sigma'])

    def bounds(self, **kwargs):
        #if len(kwargs) > 0:
        #    self.update(**kwargs)
        return (-np.inf, np.inf)


class MultiVariateNormal(Prior):
    prior_params = ["mean", 'Sigma']
    distribution = scipy.stats.norm

    @property
    def scale(self):
        return self.params['Sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        nsig = 4
        return (self.params['mean'] - nsig * self.params['Sigma'],
                self.params['mean'] + nsig * self.params['Sigma'])

    def bounds(self, **kwargs):
        #if len(kwargs) > 0:
        #    self.update(**kwargs)
        return (-np.inf, np.inf)
    
    def sample(self, nsample=None, **kwargs):
        prior = scipy.stats.multivariate_normal(mean=self.loc, cov=self.scale)
        return prior.rvs(size = nsample)

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        z = self.distribution.ppf(x, *self.args, loc=0, scale=1)
        #print(z)
        sqrtS = np.linalg.cholesky(self.params['Sigma']) #, lower=True)
        #print(sqrtS)
        #theta = np.matmul(z, sqrtS)
        theta = np.matmul(sqrtS, z)
        return theta
    
    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return scipy.stats.multivariate_normal.cdf(x, *self.args,
                                                   mean=0., cov=self.params['Sigma'])


class ClippedNormal(Prior):
    """A Gaussian prior clipped to some range.

    :param mean:
        Mean of the normal distribution

    :param sigma:
        Standard deviation of the normal distribution

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mean', 'sigma', 'mini', 'maxi']
    distribution = scipy.stats.truncnorm

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    @property
    def args(self):
        a = (self.params['mini'] - self.params['mean']) / self.params['sigma']
        b = (self.params['maxi'] - self.params['mean']) / self.params['sigma']
        return [a, b]

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogUniform(Prior):
    """Like log-normal, but the distribution of natural log of the variable is
    distributed uniformly instead of normally.

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.reciprocal

    @property
    def args(self):
        a = self.params['mini']
        b = self.params['maxi']
        return [a, b]

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class Beta(Prior):
    """A Beta distribution.

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution

    :param alpha:

    :param beta:
    """
    prior_params = ['mini', 'maxi', 'alpha', 'beta']
    distribution = scipy.stats.beta

    @property
    def scale(self):
        return self.params.get('maxi', 1) - self.params.get('mini', 0)

    @property
    def loc(self):
        return self.params.get('mini', 0)

    @property
    def args(self):
        a = self.params['alpha']
        b = self.params['beta']
        return [a, b]

    @property
    def range(self):
        return (self.params.get('mini',0), self.params.get('maxi',1))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogNormal(Prior):
    """A log-normal prior, where the natural log of the variable is distributed
    normally.  Useful for parameters that cannot be less than zero.

    Note that ``LogNormal(np.exp(mode) / f) == LogNormal(np.exp(mode) * f)``
    and ``f = np.exp(sigma)`` corresponds to "one sigma" from the peak.

    :param mode:
        Natural log of the variable value at which the probability density is
        highest.

    :param sigma:
        Standard deviation of the distribution of the natural log of the
        variable.
    """
    prior_params = ['mode', 'sigma']
    distribution = scipy.stats.lognorm

    @property
    def args(self):
        return [self.params["sigma"]]

    @property
    def scale(self):
        return  np.exp(self.params["mode"] + self.params["sigma"]**2)

    @property
    def loc(self):
        return 0

    @property
    def range(self):
        nsig = 4
        return (np.exp(self.params['mode'] + (nsig * self.params['sigma'])),
                np.exp(self.params['mode'] - (nsig * self.params['sigma'])))

    def bounds(self, **kwargs):
        return (0, np.inf)


class LogNormalLinpar(Prior):
    """A log-normal prior, where the natural log of the variable is distributed
    normally.  Useful for parameters that cannot be less than zero.

    LogNormal(mode=x, sigma=y) is equivalent to
    LogNormalLinpar(mode=np.exp(x), sigma_factor=np.exp(y))

    :param mode:
        The (linear) value of the variable where the probability density is
        highest. Must be > 0.

    :param sigma_factor:
        The (linear) factor describing the dispersion of the log of the
        variable.  Must be > 0
    """
    prior_params = ['mode', 'sigma_factor']
    distribution = scipy.stats.lognorm

    @property
    def args(self):
        return [np.log(self.params["sigma_factor"])]

    @property
    def scale(self):
        k = self.params["sigma_factor"]**np.log(self.params["sigma_factor"])
        return  self.params["mode"] * k

    @property
    def loc(self):
        return 0

    @property
    def range(self):
        nsig = 4
        return (self.params['mode'] * (nsig * self.params['sigma_factor']),
                self.params['mode'] /  (nsig * self.params['sigma_factor']))

    def bounds(self, **kwargs):
        return (0, np.inf)

class SkewNormal(Prior):
    """A normal distribution including a skew parameter

    :param location:
        Center (*not* mean, mode, or median) of the distribution.
        The center will approach the mean as skew approaches zero.

    :param sigma:
        Standard deviation of the distribution

    :param skew:
        Skewness of the distribution
    """
    prior_params = ['location', 'sigma', 'skew']
    distribution = scipy.stats.skewnorm

    @property
    def args(self):
        return [self.params['skew']]

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['location']

    @property
    def range(self):
        nsig = 4
        return (self.params['location'] - nsig * self.params['sigma'],
                self.params['location'] + nsig * self.params['sigma'])

    def bounds(self, **kwargs):
        return (-np.inf, np.inf)


class StudentT(Prior):
    """A Student's T distribution

    :param mean:
        Mean of the distribution

    :param scale:
        Size of the distribution, analogous to the standard deviation

    :param df:
        Number of degrees of freedom
    """
    prior_params = ['mean', 'scale', 'df']
    distribution = scipy.stats.t

    @property
    def args(self):
        return [self.params['df']]

    @property
    def scale(self):
        return self.params['scale']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return scipy.stats.t.interval(0.995, self.params['df'], self.params['mean'], self.params['scale'])

    def bounds(self, **kwargs):
        return (-np.inf, np.inf)


# fast versions to the above priors
# essentially rewriting the numpy/scipy functions

# A faster uniform distribution. Give it a lower bound `a` and
# an upper bound `b`.
class FastUniform(Prior):

    prior_params = ['a', 'b']

    def __init__(self, a=0.0, b=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.a, self.b = a, b

        if self.b <= self.a:
            raise ValueError('b must be greater than a')

        self.diffthing = b - a
        self.pdfval = 1.0 / (b - a)
        self.logpdfval = np.log(self.pdfval)

    def __len__(self):
        return 1

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            if self.a <= x <= self.b:
                return self.logpdfval
            else:
                return np.NINF
        else:
            return [self.logpdfval if (self.a <= xi <= self.b) else np.NINF for xi in x]

    def scale(self):
        return 0.5 * self.diffthing

    def loc(self):
        return 0.5 * (self.a + self.b)

    def unit_transform(self, x):
        return (x * self.diffthing) + self.a

    def sample(self):
        return self.unit_transform(np.random.rand())


# A faster truncated normal distribution. Give it a lower bound `a`,
# a upper bound `b`, a mean `mu`, and a standard deviation `sig`.
class FastTruncatedNormal(Prior):

    prior_params = ['a', 'b', 'mu', 'sig']

    def __init__(self, a=-1.0, b=1.0, mu=0.0, sig=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.a, self.b, self.mu, self.sig = a, b, mu, sig

        if self.b <= self.a:
            raise ValueError('b must be greater than a')

        self.alpha = (self.a - self.mu) / self.sig
        self.beta = (self.b - self.mu) / self.sig

        self.A = erf(self.alpha / np.sqrt(2.0))
        self.B = erf(self.beta / np.sqrt(2.0))

    def xi(self, x):
        return (x - self.mu) / self.sig

    def phi(self, x):
        return np.sqrt(2.0 / (self.sig**2.0 * np.pi)) * np.exp(-0.5 * self.xi(x)**2.0)

    def __len__(self):
        return 1

    def __call__(self, x):
        # if self.a <= x <= self.b:
        #     return np.log(self.phi(x) / (self.B - self.A))
        # else:
        #     return np.NINF
        if not hasattr(x, "__len__"):
            if self.a <= x <= self.b:
                return np.log(self.phi(x) / (self.B - self.A))
            else:
                np.NINF
        else:
            return [np.log(self.phi(xi) / (self.B - self.A)) if (self.a <= xi <= self.b) else np.NINF for xi in x]

    def scale(self):
        return self.sig

    def loc(self):
        return self.mu

    def unit_transform(self, x):
        return self.sig * np.sqrt(2.0) * erfinv((self.B - self.A) * x + self.A) + self.mu

    def sample(self):
        return self.unit_transform(np.random.rand())


# Okay. This is a sort of Student's t-distribution that allows
# for truncation and rescaling, but it requires nu = 2 and mu = 0
# and for the truncation limits to be equidistant from mu. Give it
# the half-width of truncation (i.e. if you want it truncated to the
# domain (-5, 5), give it `hw = 5`) and the rescaled standard
# devation `sig`.
class FastTruncatedEvenStudentTFreeDeg2(Prior):

    prior_params = ['hw', 'sig']

    def __init__(self, hw=0.0, sig=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.hw, self.sig = hw, sig

        if np.any(self.hw <= 0.0):
            raise ValueError('hw must be greater than 0.0')

        if np.any(self.sig <= 0.0):
            raise ValueError('sig must be greater than 0.0')

        self.const1 = np.sqrt(1.0 + 0.5*(self.hw**2.0))
        self.const2 = 2.0 * self.sig * self.hw
        self.const3 = self.const2**2.0
        self.const4 = 2.0 * (self.hw**2.0)

    def __len__(self):
        return len(self.hw)

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            if np.abs(x) <= self.hw:
                return np.log(self.const1 / (self.const2 * (1 + 0.5*(x / self.sig)**2.0)**1.5))
            else:
                return np.NINF
        else:
            ret = np.log(self.const1 / (self.const2 * (1 + 0.5*(x / self.sig)**2.0)**1.5))
            bad = np.abs(x) > self.hw
            ret[bad] = np.NINF
            return ret

    def scale(self):
        return self.sig

    def loc(self):
        return 0.0

    def invcdf_numerator(self, x):
        return -1.0 * (self.const3 * x**2.0 - self.const3 * x + (self.sig * self.hw)**2.0)

    def invcdf_denominator(self, x):
        return self.const4 * x**2.0 - self.const4 * x - self.sig**2.0

    def unit_transform(self, x):
        f = (((x > 0.5) & (x <= 1.0)) * np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x)) -
             ((x >= 0.0) & (x <= 0.5)) * np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x)))
        return f

    def sample(self):
        return self.unit_transform(np.random.rand())


# Okay. This is a sort of Student's t-distribution that allows
# for truncation and rescaling, but it requires nu = 2 and mu = 0
# and for the truncation limits to be equidistant from mu. Give it
# the half-width of truncation (i.e. if you want it truncated to the
# domain (-5, 5), give it `hw = 5`) and the rescaled standard
# devation `sig`.
class FastTruncatedEvenStudentTFreeDeg2Scalar(Prior):

    prior_params = ['hw', 'sig']

    def __init__(self, hw=0.0, sig=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.hw, self.sig = hw, sig

        if self.hw <= 0.0:
            raise ValueError('hw must be greater than 0.0')

        if self.sig <= 0.0:
            raise ValueError('sig must be greater than 0.0')

        self.const1 = np.sqrt(1.0 + 0.5*(self.hw**2.0))
        self.const2 = 2.0 * self.sig * self.hw
        self.const3 = self.const2**2.0
        self.const4 = 2.0 * (self.hw**2.0)

    def __len__(self):
        return 1

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            if np.abs(x) <= self.hw:
                return np.log(self.const1 / (self.const2 * (1 + 0.5*(x / self.sig)**2.0)**1.5))
            else:
                return np.NINF
        else:
            return [np.log(self.const1 / (self.const2 * (1 + 0.5*(xi / self.sig)**2.0)**1.5)) if np.abs(xi) <= self.hw else np.NINF for xi in x]

    def scale(self):
        return self.sig

    def loc(self):
        return 0.0

    def invcdf_numerator(self, x):
        return -1.0 * (self.const3 * x**2.0 - self.const3 * x + (self.sig * self.hw)**2.0)

    def invcdf_denominator(self, x):
        return self.const4 * x**2.0 - self.const4 * x - self.sig**2.0

    def unit_transform(self, x):
        if 0.0 <= x <= 0.5:
            return -1.0 * np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x))
        elif 0.5 < x <= 1.0:
            return np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x))

    def sample(self):
        return self.unit_transform(np.random.rand())


# class SMFz8:
#     """
#     z~8 Stellar Mass Function sampler.

#     Parameters
#     ----------
#     marginalize : bool
#         If True, draw a new stochastic realization of the SMF
#         each time the PDF is rebuilt using the reported
#         uncertainties.

#         If False, use the central SMF values directly.

#     random_state : int or None
#         Random seed for reproducibility.
#     """

#     def __init__(self, marginalize=False, random_state=None):

#         self.rng = np.random.default_rng(random_state)

#         self.marginalize = marginalize

#         table = np.array([
#             [7.25,  np.inf,                     np.nan,     np.nan],
#             [7.75,  np.inf,                     np.nan,     np.nan],
#             [8.25,  np.inf,                     np.nan,     np.nan],
#             [8.75, -2.9993859455738083,         0.27995724053633797, 0.17794937722098814],
#             [9.25, -3.642684086117914,          0.32570088749668447, 0.18793129122939778],
#             [9.75, -4.0887181254382226,         0.5462860677567649,  0.24216890460220508],
#             [10.25,-4.330978943786744,          1.388940513089568,   0.3040527885191393],
#             [10.75,-4.783708976480842,          99.0,                0.4471051103722452],
#             [11.25,-5.5436922065944305,         99.0,                0.5724495783106169],
#             [11.75,-5.658360812893412,          99.0,                0.5591245877898237],
#         ])

#         logM          = table[:, 0]
#         logPhi        = table[:, 1]
#         logPhi_lerr   = table[:, 2]
#         logPhi_uerr   = table[:, 3]

#         valid = np.isfinite(logPhi)

#         self.logM = logM[valid]
#         self.logPhi = logPhi[valid]
#         self.logPhi_lerr = logPhi_lerr[valid]
#         self.logPhi_uerr = logPhi_uerr[valid]

#         bad = (
#             ~np.isfinite(self.logPhi_lerr)
#             | (self.logPhi_lerr > 10)
#         )

#         self.logPhi_lerr[bad] = self.logPhi_uerr[bad]

#         # Build initial PDF
#         self._build_pdf()

#     def __len__(self):
#         return 1


#     def _build_pdf(self):

#         if self.marginalize:

#             z = self.rng.standard_normal(len(self.logPhi))

#             sigma = np.where(
#                 z < 0,
#                 self.logPhi_lerr,
#                 self.logPhi_uerr
#             )

#             logPhi_draw = self.logPhi + z * sigma

#         else:

#             logPhi_draw = self.logPhi.copy()

#         phi = 10**logPhi_draw

#         norm = trapezoid(phi, self.logM)

#         self.pdf = phi / norm

#         self.cdf = cumulative_trapezoid(
#             self.pdf,
#             self.logM,
#             initial=0
#         )

#         self.cdf /= self.cdf[-1]

#         self.inverse_cdf = interp1d(
#             self.cdf,
#             self.logM,
#             bounds_error=False,
#             fill_value=(self.logM[0], self.logM[-1])
#         )

#     def sample(self, n_samples=1, rebuild_pdf=None):
#         """
#         Sample stellar masses from the SMF.

#         Parameters
#         ----------
#         n_samples : int
#             Number of samples.

#         rebuild_pdf : bool or None
#             If True, redraw a new marginalized SMF realization
#             before sampling.

#             If False, reuse the current PDF.

#             If None:
#                 - rebuild automatically when marginalize=True
#                 - otherwise reuse the fixed PDF

#         Returns
#         -------
#         samples_logM : ndarray
#             Samples in log10(Mstar/Msun).
#         """

#         if rebuild_pdf is None:
#             rebuild_pdf = self.marginalize

#         if rebuild_pdf:
#             self._build_pdf()

#         u = self.rng.random(n_samples)
#         samples_logM = self.inverse_cdf(u)
#         return samples_logM

# class MassMetPrior:


class SchechterSMF:
    """
    Samples stellar masses from a Schechter Function.
    
    Parameters
    ----------
    logM_star : float
        The "knee" of the mass function. Typical z~8 value is ~9.8.
    alpha : float
        The low-mass power-law slope. Typical z~8 value is ~-2.0 to -2.2.
        (Note: alpha < -1 means the number of low-mass galaxies diverges,
        so a low-mass cutoff `logM_min` is physically required).
    logM_min : float
        Minimum mass to sample. 7.0 is a reasonable limit for JWST/HST.
    logM_max : float
        Maximum mass to sample. The exponential tail will naturally suppress 
        values past logM_star, but this bounds the grid.
    n_pts : int
        Resolution of the internal PDF/CDF grid.
    random_state : int or None
        Random seed for reproducibility.
    """
    
    def __init__(self, logM_star=9.8, alpha=-2.0, 
                 logM_min=7.0, logM_max=11.5, 
                 n_pts=2000, random_state=None):
        
        self.logM_star = logM_star
        self.alpha = alpha
        self.logM_min = logM_min
        self.logM_max = logM_max
        self.rng = np.random.default_rng(random_state)
        
        # Build the PDF and CDF once on initialization
        self.logM_grid = np.linspace(logM_min, logM_max, n_pts)
        self._build_interpolators()

    def __len__(self):
        return 1

    def _schechter_pdf(self, logM):
        """
        Unnormalized Schechter function for d(logM).
        Formula: phi(M) dlogM ~ (M/M*)^(alpha + 1) * exp(-M/M*)
        """
        mass_ratio = 10**(logM - self.logM_star)
        
        # We calculate the unnormalized shape. 
        # Constants like phi* and ln(10) drop out when we normalize the PDF later.
        density = (mass_ratio)**(self.alpha + 1) * np.exp(-mass_ratio)
        return density

    def _build_interpolators(self):
        # 1. Evaluate the unnormalized PDF
        pdf_unnorm = self._schechter_pdf(self.logM_grid)
        
        # 2. Normalize to create a true PDF (area under curve = 1)
        norm = np.trapz(pdf_unnorm, self.logM_grid)
        self.pdf = pdf_unnorm / norm
        
        # 3. Build the CDF
        self.cdf = cumulative_trapezoid(self.pdf, self.logM_grid, initial=0)
        self.cdf /= self.cdf[-1] # Force the last value to be exactly 1.0
        
        # 4. Create an interpolator to map [0, 1] random numbers back to logM
        self.inverse_cdf = interp1d(
            self.cdf, 
            self.logM_grid, 
            bounds_error=False, 
            fill_value=(self.logM_min, self.logM_max)
        )

    def sample(self, n_samples=1):
        """
        Draw random galaxy masses based on the Schechter probability distribution.
        """
        u = self.rng.random(n_samples)
        return self.inverse_cdf(u)
    
