import inspect
import os.path as op
from skbold.data2mvp import MvpWithin
from skbold import transformers
from skbold import testdata_path
from skbold.transformers import *

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


def test_anova_cutoff():

    transf = AnovaCutoff(cutoff=2.3)
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


def test_array_permuter():

    transf = ArrayPermuter()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


def test_mean_euclidean():

    transf = MeanEuclidean()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


def test_features_to_contrast():

    transf = FeaturesToContrast()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


def test_mean_euclidean_balanced():

    transf = MeanEuclideanBalanced()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


def test_cluster_threshold():

    transf = ClusterThreshold(mvp=mvp_within)
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


def test_pattern_averager():

    transf = PatternAverager()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


def test_pca_filter():

    transf = PCAfilter()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)
