from __future__ import division, print_function, absolute_import

from builtins import range
import os
import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.externals import joblib
from scipy import stats
from itertools import combinations
from scipy.misc import comb
from copy import copy
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             cohen_kappa_score, roc_auc_score,
                             confusion_matrix, r2_score, mean_squared_error,
                             f1_score)


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
    type_model : str
        Either 'classification' or 'regression'
    feature_scoring : str
        Which method to use to calculate feature-scores with. Can be:
        1) 'fwm': feature weight mapping [1]_ - keep track of
        raw voxel-weights (coefficients)
        2) 'forward': transform raw voxel-weights to corresponding forward-
        model [2]_.
    confmat : bool
        Whether to keep track of the confusion-matrix across folds (only
        relevant for `type_model='classification'`)
    verbose : bool
        Whether to print extra output.
    **metrics : keyword-arguments
        Keyword arguments of the form: `name_metric: metric_function`;
        any metric from scikit-learn works (or other metrics, as long as
        they have two input args, `y_true` and `y_pred`).

    References
    ----------
    .. [1] Stelzer, J., Buschmann, T., Lohmann, G., Margulies, D.S., \
       Trampel, R., and Turner, R. (2014). Prioritizing spatial accuracy in \
       high-resolution fMRI data using multivariate feature weight mapping. \
       Front. Neurosci., http://dx.doi.org/10.3389/fnins.2014.00066.
    .. [2] Haufe, S., Meineck, F., Gorger, K., Dahne, S., Haynes, J-D., \
       Blankertz, B., and Biessmann, F. et al. (2014). On the interpretation \
       of weight vectors of linear models in multivariate neuroimaging. \
       Neuroimage, 87, 96-110.
    """

    def __init__(self, mvp, n_iter, type_model='classification', feature_scoring=None,
                 confmat=False, verbose=False, **metrics):

        for name, metric in metrics.items():
            setattr(self, name, np.zeros(n_iter))

        if type_model != 'classification':
            confmat = False

        self.n_class = len(np.unique(mvp.y))
        self.n_iter = n_iter
        self.n_vox = np.zeros(self.n_iter)
        self.mvp = mvp
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
        self.voxel_values = None
        self.df = None
        self.metrics = metrics

        if type_model == 'classification':
            if self.n_class < 3 or self.fs == 'ufs':
                self.voxel_values = np.zeros((self.n_iter, mvp.X.shape[1]))
            else:
                self.voxel_values = np.zeros((self.n_iter, mvp.X.shape[1],
                                              self.n_class))
        else:
            self.voxel_values = np.zeros((self.n_iter, mvp.X.shape[1]))

        if confmat:
            self.metrics['confmat'] = confusion_matrix
            self.confmat = np.zeros((self.n_iter, self.n_class, self.n_class))        

    def save_model(self, model, out_path):
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
            fn = op.join(out_path, step[0] + '.jl')
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

    def update(self, test_idx, y_pred, pipeline=None):
        """ Updates with information from current fold.

        Parameters
        ----------
        test_idx : ndarray
            Indices of current test-trials.
        y_pred : ndarray
            Predictions of current test-trials.
        pipeline : scikit-learn Pipeline object
            pipeline from which relevant scores/coefficients will be
            extracted.
        """
        i = self.iter
        y_true = self.y[test_idx]

        for name, metric in self.metrics.items():
            tmp = getattr(self, name)
            tmp[i] = metric(y_true, y_pred)
            setattr(self, name, tmp)
            if self.verbose:
                print("%s: %.3f" % (name, tmp[i]))

        if pipeline is not None and self.fs is not None:
            self._update_voxel_values(pipeline)

        self.iter += 1

    def compute_scores(self, multiclass='ovr', maps_to_tstat=True):
        """ Computes scores across folds. """

        self._check_mvp_attributes()

        df = {name: getattr(self, name, values) for name, values in self.metrics.items()
              if name != 'confmat'}
        df['n_voxels'] = self.n_vox
        self.df = pd.DataFrame(df)
        print(self.df.describe().loc[['mean', 'std']])

        if self.fs is not None:
            feature_scores = self._calculate_feature_scores(multiclass, maps_to_tstat)
            if len(feature_scores) == 1:
                feature_scores = feature_scores[0]
                self.feature_scores = feature_scores
            return(self.df, feature_scores)
        else:    
            return(self.df)

    def write(self, out_path, confmat=True, to_tstat=True,
              multiclass='ovr'):
        """ Writes results to disk.

        Parameters
        ----------
        out_path : str
            Where to save the results to
        feature_viz : bool
            Whether to write out (and optionally return) feature-visualization
            information
        confmat : bool
            Whether to write out (and optionally return) the confusion-matrix
            (across folds).
        to_tstat : bool
            Whether to convert averaged feature-scores to t-tstats (by dividing
            them by sqrt(score.std(axis=0)).
        """

        if self.df is None:
            raise ValueError("Cannot write out results; "
                             "call compute_scores() first!")

        self.df.loc[len(self.df)] = [np.nan] * self.df.shape[1]
        self.df.loc[len(self.df)] = self.df.mean()

        self.df.to_csv(op.join(out_path, 'results.tsv'),
                       sep='\t', index=False)

        if hasattr(self, 'confmat'):
            np.save(op.join(out_path, 'confmat'), self.confmat)

        if self.fs is not None:

            if not isinstance(self.feature_scores, list):
                fscores = [self.feature_scores]
            else:
                fscore = self.feature_scores

            for i, fscore in enumerate(fscores):
                pos_idx = np.where(i == self.featureset_id)[0][0]

                if len(fscore.shape) > 3:

                    for ii in range(subset.ndim + 1):

                        fscore.to_filename(op.join(out_path,
                                           self.data_name[pos_idx] +
                                           '_%i.nii.gz' % ii))
                else:
                    fscore.to_filename(op.join(out_path,
                                       self.data_name[pos_idx] +
                                       '.nii.gz'))

    def _calculate_feature_scores(self, multiclass, to_tstat):

        values = self.voxel_values

        if multiclass == 'ovo':
            # in scikit-learn 'ovo', Positive labels are reversed
            values *= -1
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

            values = scores / n_class

        if to_tstat:
            n = values.shape[0]
            values = values.mean(axis=0) / ((values.std(axis=0)) / np.sqrt(n))
        else:
            values = values.mean(axis=0)
        values[np.isnan(values)] = 0
        fids = np.unique(self.featureset_id)

        to_return = []
        for i in fids:

            pos_idx = np.where(i == fids)[0][0]
            img = np.zeros(self.data_shape[pos_idx]).ravel()
            subset = values[self.featureset_id == i]

            if subset.ndim > 1:
                for ii in range(subset.ndim + 1):
                    tmp_idx = self.voxel_idx[self.featureset_id == i]
                    img[tmp_idx] = subset[:, ii]
                    img = nib.Nifti1Image(img.reshape(
                        self.data_shape[pos_idx]), affine=self.affine[pos_idx])
                    to_return.append(img)
                    img = np.zeros(self.data_shape[pos_idx]).ravel()

            else:
                pos_idx = np.where(i == fids)[0][0]
                img[self.voxel_idx[self.featureset_id == i]] = subset
                img = nib.Nifti1Image(img.reshape(self.data_shape[pos_idx]),
                                      affine=self.affine[pos_idx])
                to_return.append(img)
                
        return to_return

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
        elif pipe.__class__.__name__ != 'Pipeline':
            # hack to allow non-pipelines
            pipe.idx_ = np.ones(pipe.coef_.size, dtype=bool)
            pipe_steps = {pipe.__class__.__name__: pipe}
        else:
            pipe_steps = copy(pipe.named_steps)

        for name, step in pipe_steps.items():

            if hasattr(step, 'best_estimator_'):
                pipe_steps[name] = step.best_estimator_
            else:
                pipe_steps[name] = step

        match = 'coef_' if self.fs in ['fwm', 'forward'] else 'scores_'
        val = [getattr(step, match) for step in pipe_steps.values()
               if hasattr(step, match)]

        ensemble = [step for step in pipe_steps.values()
                    if hasattr(step, 'estimators_')]

        if len(val) == 1:
            val = val[0]
        elif len(val) == 0 and len(ensemble) == 1:
            val = np.concatenate([ens.coef_ for ens in ensemble[0]]).mean(
                axis=0)
        elif len(val) == 0:
            raise ValueError('Found no %s attribute anywhere in the '
                             'pipeline!' % match)
        else:
            raise ValueError('Found more than one %s attribute in the '
                             'pipeline!' % match)

        idx = [step.get_support() for step in pipe_steps.values()
               if callable(getattr(step, "get_support", None))]

        if len(idx) == 0:
            idx = [getattr(step, 'idx_') for step in pipe_steps.values()
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

        # Cov[x(n), y(n)]
        A = np.cov(X.T).dot(W)

        return A


class MvpAverageResults(object):
    """
    Averages results from MVPA analyses on, for example, different subjects
    or different ROIs.

    Parameters
    ----------
    out_dir : str
        Absolute path to directory where the results will be saved.
    """

    def __init__(self, type_model='classification'):

        self.type_model = type_model

    def compute_statistics(self, mvpr_list, identifiers=None, metric='accuracy', h0=0.5):

        if identifiers is None:
            identifiers = np.arange(len(mvpr_list))
        
        scores = np.array([getattr(mvpr, metric) for mvpr in mvpr_list])
        n = scores.shape[1]
        df = {}
        df['mean'] = scores.mean(axis=1)
        df['std'] = scores.std(axis=1)
        df['t'] = (df['mean'] - h0) / (df['std'] / np.sqrt(n - 1))
        df['p'] = [stats.t.sf(abs(tt), n - 1) for tt in df['t']]
        df['n_vox'] = np.array([mvp.n_vox for mvp in mvpr_list]).mean(axis=1)
        df = pd.DataFrame(df, index=identifiers)
        df = df.sort_values(by='t', ascending=False)
        self.df = df
        return df

    def write(self, path, name='average_results'):

        fn = op.join(path, name + '.tsv')
        self.df.to_csv(fn, sep='\t')
