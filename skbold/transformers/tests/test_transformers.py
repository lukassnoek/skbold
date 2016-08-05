import inspect
import os.path as op
from skbold.data2mvp import MvpWithin
from skbold import transformers
from skbold import testdata_path
from skbold import harvardoxford_path

transf_objects = inspect.getmembers(transformers, inspect.isclass)

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

def test_transformers():


    to_skip = ['LabelFactorizer', 'IncrementalFeatureCombiner',
               'MultiRoiIndexer', 'RowIndexer', 'SelectFeatureset',
               'MultiPatternAverager']

    for name, cls in transf_objects:

        if name in to_skip:
            continue

        print('Testing %s' % name)
        kwargs = {}

        if name == 'RoiIndexer':
            kwargs['mask'] = op.join(harvardoxford_path, 'bilateral', 'Amygdala.nii.gz')

        try:
            kwargs['mvp'] = mvp_within
            transf = cls(**kwargs)
        except:
            kwargs = {}
            transf = cls(**kwargs)

        transf.fit(mvp_within.X, mvp_within.y)
        transf.transform(mvp_within.X)  