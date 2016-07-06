import os.path as op
from skbold.data2mvp import MvpWithin
from skbold import testdata_path
import shutil


def test_fsl2mvp_within():

    testfeats = [op.join(testdata_path, 'run1.feat'),
                 op.join(testdata_path, 'run2.feat')]

    true_labels = ['actie', 'actie', 'actie',
                   'interoception', 'interoception', 'interoception',
                   'situation', 'situation', 'situation']

    mvp_within = MvpWithin(source=testfeats, read_labels=True,
                           remove_contrast=[], invert_selection=None,
                           ref_space='epi', beta2tstat=True, remove_zeros=False,
                           mask=None)

    mvp_within.create()
    print(mvp_within.contrast_labels)

if __name__ == '__main__':
    test_fsl2mvp_within()