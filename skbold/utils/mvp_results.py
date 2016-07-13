from __future__ import division, print_function

import os
import os.path as op
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, r2_score, mean_squared_error)
import nibabel as nib
from fnmatch import fnmatch


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

        for i in np.unique(mvp.featureset_id):
            img = np.zeros(mvp.data_shape[i]).ravel()
            subset = values[mvp.featureset_id == i]
            img[mvp.voxel_idx[mvp.featureset_id == i]] = subset
            img = nib.Nifti1Image(img.reshape(mvp.data_shape[i]),
                                  affine=mvp.affine[i])
            img.to_filename(op.join(self.out_path, mvp.data_name[i] + '.nii.gz'))

            n_nonzero = (subset > 0).sum()
            print('Number of non-zero voxels in %s: %i' % (mvp.data_name[i],
                                                             n_nonzero))

        self.df.to_csv(op.join(self.out_path, 'results.tsv'), sep='\t', index=False)

    def _update_voxel_values(self, values, idx):

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

if __name__ == '__main__':

    import joblib
    from sklearn.svm import SVC, SVR
    from sklearn.cross_validation import StratifiedKFold, KFold
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from skbold.data2mvp import MvpWithin, MvpBetween

    base_dir = '/media/lukas/piop/PIOP/'
    output_file = op.join(op.dirname(base_dir), 'behav', 'ALL_BEHAV.tsv')

    mvp = joblib.load('/home/lukas/between.jl')
    mvp.add_outcome_var(output_file, 'ZRaven_tot', normalize=True,
                        binarize={'type': 'constant',
                                  'cutoff': 0})

    scaler = StandardScaler()
    svm = SVC(kernel='linear', class_weight='balanced')
    anova = SelectKBest(k=10000, score_func=f_classif)

    folds = StratifiedKFold(mvp.y, n_folds=10)

    mvp_results = MvpResultsClassification(mvp, len(folds), verbose=True,
                                           feature_scoring='coef',
                                           out_path='/home/lukas/PIOPANALYSIS')

    pipe = Pipeline([('scaler', scaler),
                     ('anova', anova),
                     ('svm', svm)])

    for train_idx, test_idx in folds:
        train, test = mvp.X[train_idx, :], mvp.X[test_idx]
        y_train, y_test = mvp.y[train_idx], mvp.y[test_idx]

        pipe.fit(train, y_train)
        pred = pipe.predict(test)

        idx = np.argsort(pipe.named_steps['anova'].scores_)[::-1][:10000]
        mvp_results.update(test_idx, pred, pipe.named_steps['anova'].scores_[idx], idx)

    mvp_results.save_model(pipe)
    mvp_results.compute_scores()
    mvp_results.write()
