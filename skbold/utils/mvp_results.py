from __future__ import division, print_function

import os
import os.path as op
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, r2_score, mean_squared_error)
import nibabel as nib
from fnmatch import fnmatch
import pandas as pd
import joblib

class MvpResults(object):

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

    def _check_mvp_attributes(self):

        if not isinstance(self.affine, list):
            self.affine = [self.affine]

        if not isinstance(self.data_shape, list):
            self.data_shape = [self.data_shape]

    def write(self, to_tstat=True):

        self._check_mvp_attributes()
        values = self.voxel_values

        if to_tstat:
            n = values.shape[0]
            values = values.mean(axis=0) / ((values.std(axis=0)) / np.sqrt(n))
        else:
            values = values.mean(axis=0)

        for i in np.unique(self.featureset_id):
            img = np.zeros(self.data_shape[i]).ravel()
            subset = values[self.featureset_id == i]
            img[self.voxel_idx[self.featureset_id == i]] = subset
            img = nib.Nifti1Image(img.reshape(self.data_shape[i]),
                                  affine=self.affine[i])
            img.to_filename(op.join(self.out_path, self.data_name[i] + '.nii.gz'))

            n_nonzero = (subset > 0).sum()
            print('Number of non-zero voxels in %s: %i' % (self.data_name[i],
                                                             n_nonzero))

        self.df.to_csv(op.join(self.out_path, 'results.tsv'), sep='\t', index=False)

    def _update_voxel_values(self, values, idx):

        if values.__class__.__name__ == 'Pipeline':
            match = 'coef_' if self.fs in ['coef', 'forward'] else 'scores_'
            val = [getattr(step, match) for step in values.named_steps.values()
                   if hasattr(step, match)]

            if len(val) == 1:
                val = val[0]
            else:
                raise ValueError('Found more than one %s attribute in pipeline!'
                                 % match)

            if val.size != self.X.shape[1] and idx is None:

                idx = [step.get_support() for step in values.named_steps.values()
                       if callable(getattr(step, "get_support", None))]

                if len(idx) == 1:
                    idx = idx[0]
                else:
                    raise ValueError('Found more than one %s attribute in pipeline!'
                                     % match)

            values = val

        values = np.squeeze(values)
        self.n_vox[self.iter] = values.size

        if any(fnmatch(self.fs, typ) for typ in ['coef*', 'ufs']):
            self.voxel_values[self.iter, idx] = values

        elif self.fs == 'forward':

            # Haufe et al. (2014). On the interpretation of weight vectors of
            # linear models in multivariate neuroimaging. Neuroimage, 87, 96-110.

            W = values
            X = self.X[:, idx]
            s = W.T.dot(X.T)

            if self.n_class < 3:
                A = np.cov(X.T).dot(W)
            else:
                X_cov = np.cov(X.T)
                A = X_cov.dot(W).dot(np.linalg.pinv(np.cov(s)))

            self.voxel_values[self.iter, idx] = A

    def save_model(self, model):
        """ Method to serialize model(s) to disk."""

        # Can also be a pipeline!
        if model.__class__.__name__ == 'Pipeline':
            model = model.steps

        for step in model:
            fn = op.join(self.out_path, step[0] + '.jl')
            joblib.dump(step[1], fn, compress=3)

    def load_model(self, path, param=None):

        model = joblib.load(path)

        if param is None:
            return model
        else:
            if not isinstance(param, list):
                param = [param]
            return {p: getattr(model, p) for p in param}


class MvpResultsRegression(MvpResults):

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

    def update(self, test_idx, y_pred, values=None, idx=None):

        i = self.iter
        y_true = self.y[test_idx]

        self.R2[i] = r2_score(y_true, y_pred)
        self.mse = mean_squared_error(y_true, y_pred)

        if self.verbose:
            print('R2: %f' % self.R2[i])

        if values is not None:
            self._update_voxel_values(values, idx)

        self.iter += 1

    def compute_scores(self):

        df = pd.DataFrame({'R2': self.R2,
                           'MSE': self.mse,
                           'n_voxels': self.n_vox})
        self.df = df
        print(df.describe().loc[['mean', 'std']])


class MvpResultsClassification(MvpResults):

    def __init__(self, mvp, n_iter, feature_scoring='', verbose=False,
                 out_path=None):

        super(MvpResultsClassification, self).__init__(mvp=mvp, n_iter=n_iter,
                                                       feature_scoring=feature_scoring,
                                                       verbose=verbose,
                                                       out_path=out_path)

        self.accuracy = np.zeros(self.n_iter)
        self.recall = np.zeros(self.n_iter)
        self.precision = np.zeros(self.n_iter)
        self.n_class = np.unique(self.mvp.y).size
        self.confmat = np.zeros((self.n_iter, self.n_class, self.n_class))
        self.n_vox = np.zeros(self.n_iter)
        self.voxel_values = np.zeros((self.n_iter, mvp.X.shape[1]))

    def update(self, test_idx, y_pred, values=None, idx=None):

        i = self.iter
        y_true = self.y[test_idx]

        self.accuracy[i] = accuracy_score(y_true, y_pred)
        self.precision[i] = precision_score(y_true, y_pred, average='macro')
        self.recall[i] = recall_score(y_true, y_pred, average='macro')
        self.confmat[i, :, :] = confusion_matrix(y_true, y_pred)

        if self.verbose:
            print('Accuracy: %f' % self.accuracy[i])

        if values is not None:
            self._update_voxel_values(values, idx)

        self.iter += 1

    def compute_scores(self):

        df = pd.DataFrame({'Accuracy': self.accuracy,
                           'Precision': self.precision,
                           'Recall': self.recall,
                           'n_voxels': self.n_vox})

        self.df = df

        print('\n')
        print(df.describe().loc[['mean', 'std']])
        print('\nConfusion matrix:')
        print(self.confmat.sum(axis=0))
        print('\n')