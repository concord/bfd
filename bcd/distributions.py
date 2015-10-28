""" Defines prior distributions for bayesian changepoint detection

Formulas adapted from
https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
"""

from __future__ import division

from scipy import stats
import numpy as np


class Distribution(object):
    """Base class for probability distributions used in BCD.

    Note that each instance represents _multiple_ different pdfs, each
    assuming a different run length. So, after calling `update` t times,
    there are (t + 1) different possible values of r (0 <= r <= t).
    `pdf` should thus return an array of length (t + 1).

    We thus store (t + 1) copies of each hyperparameter.
    hyperparameter[0] is our prior distribution, hyperparameter[1] is
    the distribution fitted on 1 observation, hyperparameter[2] is the
    distribution fitted on 2 observations, and so on.
    """

    def pdf(self, observation):
        """ Get the prob density of the observation conditioned on run length.

        Note that calling pdf _does not_ automatically call update. You
        must do this manually.

        Args:
            observation: Latest observtion
        Returns:
            A 1-D array of floats, where pdf[t] is the probability
            density of observation assuming that the run length is t and given
            our fitted hyperparameters.
            pdf[t] = p(observation | r=t, hyperparameters)
        """
        raise NotImplementedError

    def update(self, observation):
        """ Update the hyperparameters given the current observation
        Args:
            observation: The latest observation
        Returns:
            None
        """

        raise NotImplementedError


class Gaussian(Distribution):
    """ Assume underlying model is Gaussian

    The mean is estimated from kappa observations with sample mean mu.
    The variance is estimated from 2 * alpha observations with sum of squared
        deviations 2 * beta

    Args:
        kappa: The prior for kappa
        mu:    The prior for mu
        alpha: The prior for alpha
        beta:  The prior for beta
    """

    def __init__(self, kappa, mu, alpha, beta):
        """ Initialize the prior distribution """
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])

    def pdf(self, observation):
        """ Calculate probability density at observation at all possible run lengths.

        This does _not_ automatically call update.

        Args:
            observation: float that is the latest observtion
        Returns:
            A 1-D array of floats, whose length is equal to the number
            of times update is called
        """
        variance = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        return stats.t.pdf(x=observation, df=2*self.alpha,
                           loc=self.mu, scale=np.sqrt(variance))

    def update(self, observation):
        """ Update the hyperparameters

        Args:
            observation: float that is the latest observation
        Returns:
            None
        """

        new_kappa = self.kappa + 1
        new_mu = (self.kappa * self.mu + observation) / (self.kappa + 1)
        new_alpha = (self.alpha + 0.5)
        new_beta = self.beta + ((self.kappa * (observation - self.mu) ** 2) /
                                (2 * self.kappa + 2))

        self.kappa = np.concatenate([self.kappa0, new_kappa])
        self.mu = np.concatenate([self.mu0, new_mu])
        self.alpha = np.concatenate([self.alpha0, new_alpha])
        self.beta = np.concatenate([self.beta0, new_beta])
