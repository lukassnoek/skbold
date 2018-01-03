from __future__ import absolute_import
import os.path as op
import pytest
import numpy as np
from ...postproc import PrevalenceInference

@pytest.mark.prevalence
@pytest.mark.parametrize("K", [1, (9, 10, 9)])
def test_prevalence(K, N=20, P1=100, P2=20000, gamma0=0.5, alpha=0.05):

    obs = np.random.normal(loc=0.55, scale=0.05, size=(N, np.prod(K)))
    perms = np.random.normal(loc=0.5, scale=0.05, size=(N, np.prod(K), P1))
    pvi = PrevalenceInference(obs=obs, perms=perms, P2=P2, alpha=alpha,
                              gamma0=gamma0)
    pvi.run()
