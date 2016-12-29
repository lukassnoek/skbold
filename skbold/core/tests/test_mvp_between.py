import os.path as op
from skbold.core import MvpBetween
from skbold import testdata_path, roidata_path
import os
from glob import glob
import shutil

mask = op.join(roidata_path, 'GrayMatter.nii.gz')
cmd = 'cp -rs %s/run1.feat %s/mock_subjects/sub00%i'
_ = [os.system(cmd % (testdata_path, testdata_path, i+1)) for i in range(9)
     if not op.isdir(op.join(testdata_path, 'mock_subjects',
                             'sub00%i' % (i+1), 'run1.feat'))]


def test_mvp_between_create():
    """ Tests create() method of MvpBetween. """

    source = dict()
    source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope1.nii.gz')}

    mvp = MvpBetween(source=source, subject_idf='sub???', mask=mask)
    mvp.create()


def test_mvp_between_add_y():
    source = dict()
    source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope1.nii.gz')}

    mvp = MvpBetween(source=source, subject_idf='sub???', mask=mask)
    mvp.create()
    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp.add_y(fpath, col_name='var_categorical', index_col=0,
              remove=999)
    assert(len(mvp.common_subjects) == mvp.X.shape[0] == mvp.y.size)
    mvp.add_y(fpath, col_name='var_categorical', index_col=0,
              remove=999, ensure_balanced=True)
    assert(mvp.y.mean() == 0.5)


def test_mvp_between_write_4D():

    source = dict()
    source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope1.nii.gz')}
    source['Contrast2'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope2.nii.gz')}

    mvp = MvpBetween(source=source, subject_idf='sub???', mask=mask)
    mvp.create()
    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp.add_y(fpath, col_name='var_categorical', index_col=0,
              remove=999)
    mvp.write_4D(testdata_path)
    assert(op.isfile(op.join(testdata_path, 'Contrast1.nii.gz')))
    assert(op.isfile(op.join(testdata_path, 'Contrast2.nii.gz')))

    # for local clean-up
    os.remove(op.join(testdata_path, 'Contrast1.nii.gz'))
    os.remove(op.join(testdata_path, 'Contrast2.nii.gz'))
    os.remove(op.join(testdata_path, 'y_4D_nifti.txt'))

def test_mvp_between_split():

    source = dict()
    source['Contrast1'] = {'path': op.join(testdata_path, 'mock_subjects',
                                           'sub*', 'run1.feat', 'stats',
                                           'cope1.nii.gz')}

    mvp = MvpBetween(source=source, subject_idf='sub???', mask=mask)
    mvp.create()
    fpath = op.join(testdata_path, 'sample_behav.tsv')
    mvp.split(fpath, col_name='group', target='train')
    spaths = glob(op.join(testdata_path, 'mock_subjects',
                          'sub*', 'run1.feat'))
    _ = [shutil.rmtree(s) for s in spaths]

