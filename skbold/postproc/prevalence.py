from __future__ import division, print_function, absolute_import

from builtins import range
import os.path as op
import numpy as np
import nibabel as nib
from sklearn.externals import joblib
from scipy import stats
from tqdm import trange
from warnings import filterwarnings

filterwarnings(action='ignore', category=RuntimeWarning)

class PrevalenceInference(object):
    """
    Class that performs PrevalenceInference based on the paper by Allefeld,
    Gorgen, & Haynes (2016), NeuroImage.

    Parameters
    ----------
    obs : numpy ndarray
        A 2D array of shape [N (subjects) x K (voxels)], or a 1D array of shape
        [N, 1].
    perms : numpy ndarray
        A 3D array of shape [N (subjects) x K (voxels) x P1 (first level
        permutations)], or a 2D array of shape [N x P1]
    P2 : int
        Number of second level permutations to run
    gamma0 : float
        What prevalence inference null (gamma < gamma0) to test
    alpha : float
        Significance level for hypothesis testing

    Examples
    --------
    >>> from skbold.postproc import PrevalenceInference
    >>> import numpy as np
    >>> N, K, P1 = 20, (40, 40, 38), 15
    >>> obs = np.random.normal(loc=0.55, scale=0.05, size=(N, np.prod(K)))
    >>> perms = np.random.normal(loc=0.5, scale=0.05, size=(N, np.prod(K), P1))
    >>> pvi = PrevalenceInference(obs=obs, perms=perms, P2=100000, gamma0=05,
                                  alpha=0.05)
    >>> pvi.run()
    Running with parameters:
	N = 20
	K = 60800
	P1 = 15
	P2 = 100000
    """

    def __init__(self, obs, perms, P2=100000, gamma0=0.5, alpha=0.05):
        """ Initializes PrevalenceInference object."""

        self.obs = obs
        self.perms = perms
        self.P2 = P2
        self.gamma0 = gamma0
        self.alpha = alpha
        self.N = None
        self.K = None
        self.P1 = None

        print("This is experimental functionality! (i.e., not yet fully tested"
              " through)")

    def _check_inputs(self):
        """ Checks and validates inputs data and extracts parameters. """

        # Remove singleton dimensions
        self.obs = self.obs.squeeze()
        self.perms = self.perms.squeeze()

        if self.perms.ndim > 3:
            raise ValueError("Your array should be 2D or 3D!")

        if self.obs.ndim == 2:  # Assume we're dealing with multiple voxels
            self.K = self.obs.shape[1]
            self._create_mask()  # To remove NaNs and such
        else:  # We just have a single score
            self.K = 1
            # Add singleton axis to make calculations more parsimonious
            self.perms = self.perms[:, np.newaxis, :]

        if self.obs.ndim > 2:
            msg = "Observed values (obs) should be 1 or 2 dimensional!"
            raise ValueError(msg)

        self.N = self.obs.shape[0]
        self.P1 = self.perms.shape[-1]

        print("Running with parameters:\n\tN = %i\n\tK = %i\n\tP1 = %i\n\tP2 = %i"
              % (self.N, self.K, self.P1, self.P2))

    def _create_mask(self):
        """ Removes all zero or NaN voxels. """
        # Stack all data together
        all_data = np.dstack((self.obs[:, :, np.newaxis], self.perms))

        # Voxels should be neither zero nor NaN
        mask = np.logical_and(~(np.isnan(all_data).sum(axis=(0, 2)) > 0),
                              ~((all_data == 0.0).sum(axis=(0, 2)) > 0))
        all_data_nonzero = all_data[:, mask, :]  # index with mask

        # Unstack obs and perms
        self.obs = all_data_nonzero[:, :, 0]
        self.perms = all_data_nonzero[:, :, 1:]
        print("Found %i non-zero voxels" % mask.sum())

    def run(self):
        """ Runs actual prevalence inference algorithm. """

        self._check_inputs()

        # Shorten parameters for clarity
        N, K, P1, P2, alpha, gamma0 = (self.N, self.K, self.P1, self.P2,
                                       self.alpha, self.gamma0)

        m = np.min(self.obs, axis=0)
        u_rank = np.zeros(K)

        if K > 1:
            c_rank = np.zeros(K)

        for j in trange(P2):  # Loop for second level permutations
            these_perms = np.vstack([self.perms[k, :, np.random.choice(np.arange(P1))]
                                     for k in range(N)])
            min_vals = these_perms.min(axis=0)
            u_rank += m <= min_vals  # Update uncorrected values

            if K > 1:  # Update corrected values
                c_rank += m <= min_vals.max()

        # Calculate statistics!
        # - pu_GN = pvalue uncorrected Global Null,
        # - pu_MN = pvalue uncorrected Majority Null,
        # - pc_MN = pvalue corrected Majority Null,
        # - gamma0_u = largest gamma0 given alpha (uncorrected)
        # - gamma0_max_u = largest possible gamma0 given N, P2, and alpha (uncorrected)
        pu_GN = (1 + u_rank) / (P2 + 1)
        pu_MN = ((1 - gamma0) * pu_GN ** (1 / N) + gamma0) ** N
        gamma0_u = (alpha ** (1 / N) - pu_GN ** (1 / N)) / (1 - pu_GN ** (1 / N))
        gamma0_u[alpha < pu_GN] = 0
        gamma0_max_u = (alpha ** (1 / N) - 1 / P2 ** (1 / N)) / (1 - 1 / P2 ** (1 / N))

        self.pu_GN = pu_GN
        self.pu_MN = pu_MN
        self.gamma0_u = gamma0_u
        self.gamma0_max_u = gamma0_max_u

        # Some extra corrected statistics
        # - pc_GN = pvalue corrected Global Null
        # - pc_GN = pvalue corrected Global Null
        # - gamma0_c = largest gamma0 given alpha (corrected)
        # - gamma0_max_c = largest possible gamma0 given N, P2, and alpha (corrected)
        if K > 1:
            pc_GN = (1 + c_rank) / (P2 + 1)
            pc_MN = pc_GN + (1 - pc_GN) * pu_MN
            alpha_c = (alpha - pc_GN) / (1 - pc_GN)
            alpha_c[np.isinf(alpha_c)] = 0
            alpha_c[alpha_c <= 0.0] = 0
            gamma0_c = (alpha_c ** (1 / N) - pu_GN ** (1 / N)) / (1 - pu_GN ** (1 / N))
            gamma0_c[alpha_c < pu_GN] = 0
            alpha_max_c = (alpha - 1 / P2) / (1 - 1 / P2)
            gamma0_max_c = (alpha_max_c ** (1 / N) - 1 / P2 ** (1 / N)) / (1 - 1 / P2 ** (1 / N))
            self.pc_GN = pc_GN
            self.pc_MN = pc_MN
            self.gamma0_c = gamma0_c
            self.gamma0_max_c = gamma0_max_c

        # Median scores of observed values
        self.score_typical = np.median(self.obs, axis=0)

    def write(self, path):
        """ Writes results from Prevalence Inference procedure to disk.

        Parameters
        ----------
        path : str
            Where to write the results to disk
        """

        pass
