from __future__ import absolute_import
import os.path as op
from ...core import MvpBetween
from ... import testdata_path, roidata_path
import os
from ...postproc import MvpResultsClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pytest

dpath = op.join(testdata_path, 'mock_subjects', 'sub*', 'run1.feat', 'stats')
bmask = op.join(roidata_path, 'other', 'GrayMatter_prob.nii.gz')
source = dict()
source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                       'sub*', 'run1.feat', 'stats',
                                       'cope1.nii.gz')}

mvp = MvpBetween(source=source, subject_idf='sub???', mask=bmask)
mvp.create()
fpath = op.join(testdata_path, 'sample_behav.tsv')
mvp.add_y(fpath, col_name='var_categorical', index_col=0, remove=999)


@pytest.mark.parametrize("method", ['fwm', 'forward', 'ufs'])
def test_mvp_results(method):

    clf = SVC(kernel='linear')
    ufs = SelectKBest(score_func=f_classif, k=100)
    pipe = Pipeline([('ufs', ufs), ('clf', clf)])

    folds = StratifiedKFold(n_splits=2)
    mvpr = MvpResultsClassification(mvp=mvp, n_iter=2,
                                    feature_scoring=method,
                                    out_path=testdata_path)

    for train_idx, test_idx in folds.split(mvp.X, mvp.y):
        train_X, test_X = mvp.X[train_idx, :], mvp.X[test_idx, :]
        train_y, test_y = mvp.y[train_idx], mvp.y[test_idx]
        pipe.fit(train_X, train_y)
        pred = pipe.predict(test_X)
        mvpr.update(test_idx, pred, pipeline=pipe)

    mvpr.compute_scores()
    mvpr.write(feature_viz=True)

    for f in ['Contrast1.nii.gz', 'results.tsv', 'confmat.npy']:
        assert(op.isfile(op.join(testdata_path, f)))
        os.remove(op.join(testdata_path, f))
