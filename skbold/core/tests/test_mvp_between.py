import os.path as op
from skbold.core import MvpBetween
from skbold import testdata_path, roidata_path
import os
import pytest
import numpy as np


cmd = 'cp -r %s/run1.feat %s/mock_subjects/sub00%i'
_ = [os.system(cmd % (testdata_path, testdata_path, i+1)) for i in range(9)
     if not op.isdir(op.join(testdata_path, 'mock_subjects',
                             'sub00%i' % (i + 1), 'run1.feat'))]

dpath = op.join(testdata_path, 'mock_subjects', 'sub*', 'run1.feat', 'stats')
bmask = op.join(roidata_path, 'other', 'GrayMatter_prob.nii.gz')
slist = ['sub001', 'sub002', 'sub003', 'sub004']


@pytest.mark.parametrize("source",
                         [{'Contrast1': {'path': dpath + '/cope1.nii.gz'}},
                          {'Contrast1': {'path': dpath + '/cope1.nii.gz'},
                           'Contrast2': {'path': dpath + '/cope2.nii.gz'}}])
@pytest.mark.parametrize("mask", [bmask, None])
@pytest.mark.parametrize("subject_list", [None, slist])
def test_mvp_between_create(source, mask, subject_list):
    source = dict()
    source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope1.nii.gz')}

    mvp = MvpBetween(source=source, subject_idf='sub???', mask=mask)
    mvp.create()


@pytest.fixture
def mvp1c():
    mask = op.join(roidata_path, 'other', 'GrayMatter_prob.nii.gz')

    source = dict()
    source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope1.nii.gz')}

    mvp = MvpBetween(source=source, subject_idf='sub???', mask=mask)
    mvp.create()
    return mvp


@pytest.fixture
def mvp2c():
    mask = op.join(roidata_path, 'other', 'GrayMatter_prob.nii.gz')

    source = dict()
    source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope1.nii.gz')}
    source['Contrast2'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope2.nii.gz')}

    mvp = MvpBetween(source=source, subject_idf='sub???', mask=mask)
    mvp.create()
    return mvp


def test_mvp_between_add_y(mvp1c):
    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp1c.add_y(fpath, col_name='var_categorical', index_col=0, remove=999)
    assert(len(mvp1c.y) == 7)
    assert(mvp1c.common_subjects == ['sub001', 'sub002', 'sub004',
                                     'sub005', 'sub006', 'sub007', 'sub009'])
    assert(len(mvp1c.common_subjects) == mvp1c.X.shape[0] == mvp1c.y.size)
    mvp1c.add_y(fpath, col_name='var_categorical', index_col=0, remove=999,
                ensure_balanced=True)
    assert(mvp1c.y.mean() == 0.5)


@pytest.mark.parametrize("mvp", [mvp1c(), mvp2c()])
def test_mvp_between_write_4D(mvp):

    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp.add_y(fpath, col_name='var_categorical', index_col=0, remove=999)
    mvp.write_4D(testdata_path)

    for data_name in mvp.data_name:
        assert(op.isfile(op.join(testdata_path, '%s.nii.gz' % data_name)))
        os.remove(op.join(testdata_path, '%s.nii.gz' % data_name))

    os.remove(op.join(testdata_path, 'y_4D_nifti.txt'))


@pytest.mark.parametrize("mvp", [mvp1c(), mvp2c()])
def test_mvp_between_split(mvp):

    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp.split(fpath, col_name='group', target='train')


@pytest.mark.parametrize("params", [{'type': 'percentile',
                                     'high': 60, 'low': 40},
                                    {'type': 'constant', 'cutoff': 100},
                                    {'type': 'median'},
                                    {'type': 'zscore', 'std': 0.25}])
def test_mvp_between_binarize_y(mvp1c, params):
    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp1c.add_y(fpath, col_name='var_continuous', index_col=0)
    mvp1c.binarize_y(params, ensure_balanced=True, save_path=testdata_path)
    assert((np.unique(mvp1c.y) == [0, 1]).all())
    assert(op.isfile(op.join(testdata_path, 'binarize_params.pkl')))


def test_mvp_between_apply_binarization_params(mvp1c):
    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp1c.add_y(fpath, col_name='var_continuous', index_col=0)
    mvp1c.apply_binarization_params(op.join(testdata_path,
                                            'binarize_params.pkl'))
    os.remove(op.join(testdata_path, 'binarize_params.pkl'))


def test_mvp_between_update_sample(mvp1c):

    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp1c.add_y(fpath, col_name='var_categorical', index_col=0,
                remove=999)
    idx = np.array([True, True, True, False, True, True, False])
    mvp1c.update_sample(idx)
    assert(len(mvp1c.y) == 5)
    assert(len(mvp1c.y) == mvp1c.X.shape[0] == len(mvp1c.common_subjects))
    assert(mvp1c.common_subjects == ['sub001', 'sub002', 'sub004',
                                     'sub006', 'sub007'])
