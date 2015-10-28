from __future__ import print_function, division
import pytest
import numpy as np

from bcd import offline_changepoint_detection, BayesianChangepointDetection
from bcd.distributions import Gaussian


@pytest.fixture(scope="module")
def hazard():
    def hazard_func(t):
        return 0.5 ** t
    return hazard_func


def test_bcd_against_reference(hazard):
    data = [0] * 100 + [2] * 100
    detector = BayesianChangepointDetection(hazard, Gaussian(0.1, 0.1, 1, 1))

    reference = offline_changepoint_detection(data,
                                              hazard, Gaussian(0.1, 0.1, 1, 1))

    for i, observation in enumerate(data):
        pr = detector.step(observation)
        # we want 0th to ith elements inclusive, so + 2
        assert (pr == reference[:i + 2, i]).all()


def test_gaussian_update():
    g = Gaussian(0, 0, 0, 0)
    g.update(2)
    g.update(-1)
    g.update(1)
    g.update(0)

    # hand-calculated parameters:
    # kappa = [0, 1,   2,   3,   4]
    # alpha = [0, 1/2, 1,   3/2, 2]
    # mu =    [0, 0,   1/2, 0,   1/2]
    # beta =  [0, 0,   1/4, 1,   5/2]

    assert np.isclose(g.kappa, [0, 1, 2, 3, 4]).all()
    assert np.isclose(g.alpha, [0, 0.5, 1, 1.5, 2]).all()
    assert np.isclose(g.mu, [0, 0, 0.5, 0, 0.5]).all()
    assert np.isclose(g.beta, [0, 0, 0.25, 1, 2.5]).all()


def test_gaussian_pdf():
    g = Gaussian(1, 0, 1, 1)
    # g.pdf ~ t-distribution w/ np.sqrt(2) degrees of freedom
    assert np.isclose(g.pdf(0), 0.25)
    assert np.isclose(g.pdf(1), 0.17888543819998315)
