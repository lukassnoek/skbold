import os.path as op
from skbold.core import MvpBetween
from skbold import testdata_path, roidata_path
import os
import pytest
import numpy as np
from skbold.postproc import MvpResultsClassification
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC

dpath = op.join(testdata_path, 'mock_subjects', 'sub*', 'run1.feat', 'stats')
bmask = op.join(roidata_path, 'GrayMatter.nii.gz')
source = dict()
source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                       'sub*', 'run1.feat', 'stats',
                                       'cope1.nii.gz')}

mvp = MvpBetween(source=source, subject_idf='sub???', mask=bmask)
mvp.create()
fpath = op.join(testdata_path, 'sample_behav.tsv')
mvp.add_y(fpath, col_name='var_categorical', index_col=0, remove=999)


def test_mvp_results_init():

    clf = SVC(kernel='linear')
    folds = StratifiedKFold(mvp.y, n_folds=2)
    mvpr = MvpResultsClassification(mvp=mvp, n_iter=2, feature_scoring='fwm',
                                    out_path=op.join(testdata_path))

    for train_idx, test_idx in folds:
        train_X, test_X = mvp.X[train_idx, :], mvp.X[test_idx, :]
        train_y, test_y = mvp.y[train_idx], mvp.y[test_idx]
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)
        mvpr.update(test_idx, pred, pipeline=clf)

    mvpr.compute_scores()
    mvpr.write(feature_viz=True)

    for f in ['Contrast1.nii.gz', 'results.tsv', 'confmat.npy']:
        assert(op.isfile(op.join(testdata_path, f)))
        os.remove(op.join(testdata_path, f))
