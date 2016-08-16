from __future__ import division, print_function

import os
import os.path as op
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, r2_score, mean_squared_error,
                             f1_score)
import nibabel as nib
from fnmatch import fnmatch
from itertools import combinations
from scipy.misc import comb
import pandas as pd
import joblib
from scipy import stats


class MvpResults(object):
    """
    .. _ReadTheDocs: http://skbold.readthedocs.io
    Class to keep track of model evaluation metrics and feature scores.
    See the ReadTheDocs_ homepage for more information on its API and use.

    Parameters
    ----------
    mvp : mvp-object
        Necessary to extract some metadata from.
    n_iter : int
        Number of folds that will be kept track of.
    out_path : str
        Path to save results to.
    feature_scoring : str
        Which method to use to calculate feature-scores with. Can be:
        1) 'fwm': feature weight mapping [1]_ - keep track of
        raw voxel-weights (coefficients)
        2) 'forward': transform raw voxel-weights to corresponding forward-
        model [2]_.
    verbose : bool
        Whether to print extra output.

    References
    ----------
    .. [1] Stelzer, J., Buschmann, T., Lohmann, G., Margulies, D.S., Trampel,
     R., and Turner, R. (2014). Prioritizing spatial accuracy in high-resolution
     fMRI data using multivariate feature weight mapping. Front. Neurosci.,
     http://dx.doi.org/10.3389/fnins.2014.00066.

    .. [2] Haufe, S., Meineck, F., Gorger, K., Dahne, S., Haynes, J-D.,
    Blankertz, B., and Biessmann, F. et al. (2014). On the interpretation of
    weight vectors of linear models in multivariate neuroimaging. Neuroimage,
    87, 96-110.
    """

    def __init__(self, mvp, n_iter, out_path=None, feature_scoring='',
                 verbose=False):

        self.mvp = mvp
        self.n_iter = n_iter
        self.fs = feature_scoring
        self.verbose = verbose
        self.X = mvp.X
        self.y = mvp.y
        self.data_shape = mvp.data_shape
        self.data_name = mvp.data_name
        self.affine = mvp.affine
        self.voxel_idx = mvp.voxel_idx
        self.featureset_id = mvp.featureset_id
        self.iter = 0

        if out_path is None:
            out_path = os.getcwd()

        if not op.exists(out_path):
            os.makedirs(out_path)

        self.out_path = out_path

    def save_model(self, model):
        """ Method to serialize model(s) to disk.

        Parameters
        ----------
        model : pipeline or scikit-learn object.
            Model to be saved.
        """

        # Can also be a pipeline!
        if model.__class__.__name__ == 'Pipeline':
            model = model.steps

        for step in model:
            fn = op.join(self.out_path, step[0] + '.jl')
            joblib.dump(step[1], fn, compress=3)

    def load_model(self, path, param=None):
        """ Load model or pipeline from disk.

        Parameters
        ----------
        path : str
            Absolute path to model.
        param : str
            Which, if any, specific param needs to be loaded.
        """
        model = joblib.load(path)

        if param is None:
            return model
        else:
            if not isinstance(param, list):
                param = [param]
            return {p: getattr(model, p) for p in param}

    def write(self, feature_viz=True, confmat=True, to_tstat=True,
              multiclass='ovr'):
        """ Writes results to disk.

        Parameters
        ----------
        to_tstat : bool
            Whether to convert averaged coefficients to t-tstats (by dividing
            them by sqrt(coefs.std(axis=0)).
        """
        self._check_mvp_attributes()
        values = self.voxel_values

        self.df.to_csv(op.join(self.out_path, 'results.tsv'), sep='\t', index=False)

        if hasattr(self, 'confmat') and confmat:
            np.save(op.join(self.out_path, 'confmat'), self.confmat)

        if not feature_viz:
            return None

        if multiclass == 'ovo':
            # in scikit-learn 'ovo', Positive labels are reversed
            values = values * -1
            n_class = len(np.unique(self.mvp.y))
            n_models = comb(n_class, 2, exact=True)
            cmb = list(combinations(range(n_models), 2))

            scores = np.zeros((values.shape[0], values.shape[1], n_class))

            for number in range(n_models):

                for i, c in enumerate(cmb):

                    if number in c:

                        if c.index(number) == 1:
                            val = values[:, :, i] * -1
                        else:
                            val = values[:, :, i]

                        scores[:, :, number] += val

            values = scores / 3

        if to_tstat:
            n = values.shape[0]
            values = values.mean(axis=0) / ((values.std(axis=0)) / np.sqrt(n))
        else:
            values = values.mean(axis=0)

        for i in np.unique(self.featureset_id):

            img = np.zeros(self.data_shape[i]).ravel()
            subset = values[self.featureset_id == i]

            if subset.ndim > 1:
                for ii in range(subset.ndim + 1):
                    img[self.voxel_idx[self.featureset_id == i]] = subset[:, ii]
                    img = nib.Nifti1Image(img.reshape(self.data_shape[i]),
                                          affine=self.affine[i])
                    img.to_filename(op.join(self.out_path, self.data_name[i] + '_%i.nii.gz' % ii))
                    img = np.zeros(self.data_shape[i]).ravel()

            else:
                img[self.voxel_idx[self.featureset_id == i]] = subset
                img = nib.Nifti1Image(img.reshape(self.data_shape[i]),
                                      affine=self.affine[i])
                img.to_filename(op.join(self.out_path,
                                        self.data_name[i] + '.nii.gz'))

    def _check_mvp_attributes(self):

        if not isinstance(self.affine, list):
            self.affine = [self.affine]

        if not isinstance(self.data_shape, list):
            self.data_shape = [self.data_shape]

        if not isinstance(self.data_name, list):
            self.data_name = [self.data_name]

    def _extract_values_from_pipeline(self, pipe):

        if pipe.__class__.__name__ == 'GridSearchCV':
            pipe = pipe.best_estimator_

        match = 'coef_' if self.fs in ['fwm', 'forward'] else 'scores_'
        val = [getattr(step, match) for step in pipe.named_steps.values()
               if hasattr(step, match)]

        ensemble = [step for step in pipe.named_steps.values()
                    if hasattr(step, 'estimators_')]

        if len(val) == 1:
            val = val[0]
        elif len(val) == 0 and len(ensemble) == 1:
            val = np.concatenate([ens.coef_ for ens in ensemble[0]]).mean(
                axis=0)
        elif len(val) == 0:
            raise ValueError('Found no %s attribute anywhere in the ' \
                             'pipeline!' % match)
        else:
            raise ValueError('Found more than one %s attribute in the ' \
                             'pipeline!' % match)

        idx = [step.get_support() for step in pipe.named_steps.values()
               if callable(getattr(step, "get_support", None))]

        if len(idx) == 0:
            idx = [getattr(step, 'idx_') for step in pipe.named_steps.values()
                   if hasattr(step, 'idx_')]

        if len(idx) == 1:
            idx = idx[0]
        elif len(idx) > 1:
            msg = 'Found more than one index in pipeline!'
            raise ValueError(msg)
        else:
            msg = 'Found no index in pipeline!'
            raise ValueError(msg)

        val = np.squeeze(val)
        if val.shape[0] != idx.sum():
            val = val.T

        return val, idx

    def _update_voxel_values(self, pipe):

        val, idx = self._extract_values_from_pipeline(pipe)
        self.n_vox[self.iter] = val.shape[0]

        if self.fs == 'fwm':
            self.voxel_values[self.iter, idx] = val
        elif self.fs == 'ufs':
            self.voxel_values[self.iter, :] = val
        elif self.fs == 'forward':
            A = self._calculate_forward_mapping(val, idx)
            self.voxel_values[self.iter, idx] = A
        else:
            msg = "Please specify either 'ufs', 'fwm', or 'forward'."
            raise ValueError(msg)

    def _calculate_forward_mapping(self, val, idx):

        # Haufe et al. (2014). On the interpretation of weight vectors of
        # linear models in multivariate neuroimaging. Neuroimage, 87, 96-110.

        W = val
        X = self.X[:, idx]
        s = W.dot(X.T)

        if len(np.unique(self.y)) < 3:
            A = np.cov(X.T).dot(W)
        else:
            X_cov = np.cov(X.T)
            A = X_cov.dot(W.T).dot(np.linalg.pinv(np.cov(s)))

        return A


class MvpResultsRegression(MvpResults):
    """
    MvpResults class specifically for Regression analyses.

    Parameters
    ----------
    mvp : mvp-object
        Necessary to extract some metadata from.
    n_iter : int
        Number of folds that will be kept track of.
    out_path : str
        Path to save results to.
    feature_scoring : str
        Which method to use to calculate feature-scores with. Can be:
        1) 'coef': keep track of raw voxel-weights (coefficients)
        2) 'forward': transform raw voxel-weights to corresponding forward-
        model (see Haufe et al. (2014). On the interpretation of weight vectors
        of linear models in multivariate neuroimaging. Neuroimage, 87, 96-110.)
    verbose : bool
        Whether to print extra output.

    .. warning:: Has not been tested with MvpWithin!

    """
    def __init__(self, mvp, n_iter, feature_scoring='', verbose=False,
                 out_path=None):

        super(MvpResultsRegression, self).__init__(mvp=mvp, n_iter=n_iter,
                                                   feature_scoring=feature_scoring,
                                                   verbose=verbose,
                                                   out_path=out_path)

        self.R2 = np.zeros(self.n_iter)
        self.mse = np.zeros(self.n_iter)
        self.n_vox = np.zeros(self.n_iter)
        self.voxel_values = np.zeros((self.n_iter, mvp.X.shape[1]))

    def update(self, test_idx, y_pred, pipeline=None):
        """ Updates with information from current fold.

        Parameters
        ----------
        test_idx : ndarray
            Indices of current test-trials.
        y_pred : ndarray
            Predictions of current test-trials.
        values : ndarray
            Values of features for model in the current fold. This can be the
            entire pipeline (in this case, it is extracted automaticlly). When
            a pipeline is passed, the idx-parameter does not have to be passed.
        idx : ndarray
            Index mapping the 'values' back to whole-brain space.
        """
        i = self.iter
        y_true = self.y[test_idx]

        self.R2[i] = r2_score(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)

        if self.verbose:
            print('R2: %f' % self.R2[i])

        if pipeline is not None:
            self._update_voxel_values(pipeline)

        self.iter += 1

    def compute_scores(self):
        """ Computes scores across folds. """
        df = pd.DataFrame({'R2': self.R2,
                           'MSE': self.mse,
                           'n_voxels': self.n_vox})
        self.df = df
        print(df.describe().loc[['mean', 'std']])


class MvpResultsClassification(MvpResults):
    """
    MvpResults class specifically for classification analyses.

    Parameters
    ----------
    mvp : mvp-object
        Necessary to extract some metadata from.
    n_iter : int
        Number of folds that will be kept track of.
    out_path : str
        Path to save results to.
    feature_scoring : str
        Which method to use to calculate feature-scores with. Can be:
        1) 'coef': keep track of raw voxel-weights (coefficients)
        2) 'forward': transform raw voxel-weights to corresponding forward-
        model (see Haufe et al. (2014). On the interpretation of weight vectors
        of linear models in multivariate neuroimaging. Neuroimage, 87, 96-110.)
    verbose : bool
        Whether to print extra output.
    """

    def __init__(self, mvp, n_iter, feature_scoring='', verbose=False,
                 out_path=None):

        super(MvpResultsClassification, self).__init__(mvp=mvp, n_iter=n_iter,
                                                       feature_scoring=feature_scoring,
                                                       verbose=verbose,
                                                       out_path=out_path)

        self.accuracy = np.zeros(self.n_iter)
        self.recall = np.zeros(self.n_iter)
        self.precision = np.zeros(self.n_iter)
        self.f1 = np.zeros(self.n_iter)
        self.n_class = np.unique(self.mvp.y).size
        self.confmat = np.zeros((self.n_iter, self.n_class, self.n_class))
        self.n_vox = np.zeros(self.n_iter)
        self.n_class = len(np.unique(mvp.y))
        if self.n_class < 3 or self.fs == 'ufs':
            self.voxel_values = np.zeros((self.n_iter, mvp.X.shape[1]))
        else:
            self.voxel_values = np.zeros((self.n_iter, mvp.X.shape[1], self.n_class))

    def update(self, test_idx, y_pred, pipeline=None):
        """ Updates with information from current fold.

        Parameters
        ----------
        test_idx : ndarray
            Indices of current test-trials.
        y_pred : ndarray
            Predictions of current test-trials.
        values : ndarray
            Values of features for model in the current fold. This can be the
            entire pipeline (in this case, it is extracted automaticlly). When
            a pipeline is passed, the idx-parameter does not have to be passed.
        idx : ndarray
            Index mapping the 'values' back to whole-brain space.
        """

        i = self.iter
        y_true = self.y[test_idx]

        self.accuracy[i] = accuracy_score(y_true, y_pred)
        self.precision[i] = precision_score(y_true, y_pred, average='macro')
        self.recall[i] = recall_score(y_true, y_pred, average='macro')
        self.f1[i] = f1_score(y_true, y_pred, average='macro')
        self.confmat[i, :, :] = confusion_matrix(y_true, y_pred)

        if self.verbose:
            print('Accuracy: %f' % self.accuracy[i])

        if pipeline is not None:
            self._update_voxel_values(pipeline)

        self.iter += 1

    def compute_scores(self):
        """ Computes scores across folds. """
        df = pd.DataFrame({'Accuracy': self.accuracy,
                           'Precision': self.precision,
                           'Recall': self.recall,
                           'F1': self.f1,
                           'n_voxels': self.n_vox})

        self.df = df

        print('\n')
        print(df.describe().loc[['mean', 'std']])
        print('\nConfusion matrix:')
        print(self.confmat.sum(axis=0))
        print('\n')


class MvpAverageResults(object):
    """
    Averages results from MVPA analyses on, for example, different subjects
    or different ROIs.

    Parameters
    ----------
    out_dir : str
        Absolute path to directory where the results will be saved.
    """

    def __init__(self, out_dir, type='classification'):

        self.out_dir = out_dir
        self.type = type

    def compute(self, mvp_list, identifiers, metric='f1', h0=0.5):

        identifiers = [op.basename(p).split('.')[0] for p in identifiers]
        scores = np.array([getattr(mvp, metric) for mvp in mvp_list])
        n = scores.shape[1]
        df = {}
        df['mean'] = scores.mean(axis=1)
        df['std'] = scores.std(axis=1)
        df['t'] = (df['mean'] - h0) / (df['std'] / np.sqrt(n - 1))
        df['p'] = [stats.t.sf(abs(tt), n - 1) for tt in df['t']]
        df['n_vox'] = np.array([mvp.n_vox for mvp in mvp_list]).mean(axis=1)
        df = pd.DataFrame(df, index=identifiers)
        df = df.sort_values(by='t', ascending=False)
        self.df = df
        print(self.df)

    def write(self, path, name='average_results'):

        fn = op.join(path, name + '.tsv')
        self.df.to_csv(fn, sep ='\t')
