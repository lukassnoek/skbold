from __future__ import absolute_import, division, print_function

import os.path as op
from ...core import MvpWithin
from ... import testdata_path
from ...feature_extraction import *
from ...utils.roi_globals import available_atlases, other_rois
from ...utils.parse_roi_labels import parse_roi_labels
import pytest
import os
import random
from glob import glob

testfeats = [op.join(testdata_path, 'run1.feat'),
             op.join(testdata_path, 'run2.feat')]


mvp_within = MvpWithin(source=testfeats, read_labels=True,
                       remove_contrast=[], invert_selection=False,
                       ref_space='epi', beta2tstat=True, remove_zeros=False,
                       mask=None)

mvp_within.create()
reg_dir = mvp_within.source[0] + '/reg'

orig_reg_files = ['example_func.nii.gz', 'example_func2standard.mat',
                  'example_func2standard.nii.gz', 'standard.nii.gz',
                  'standard2example_func.mat']
orig_reg_files = [op.join(reg_dir, f) for f in orig_reg_files]


@pytest.mark.transformer
def test_array_permuter():

    transf = ArrayPermuter()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


@pytest.mark.transformer
def test_cluster_threshold():

    transf = ClusterThreshold(mvp=mvp_within, min_score=2)
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


@pytest.mark.transformer
def test_pattern_averager():

    transf = PatternAverager()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


@pytest.mark.transformer
def test_pca_filter():

    transf = PCAfilter()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


@pytest.mark.transformer
def test_roi_indexer():

    for roi in [r for r in other_rois if '1mm' not in r]:
        transf = RoiIndexer(mvp=mvp_within, mask=roi, mask_threshold=0,
                            reg_dir=reg_dir)
        transf.fit(mvp_within.X, mvp_within.y)
        transf.transform(mvp_within.X)

    for atlas in available_atlases:
        rois = parse_roi_labels(atlas).keys()
        roi = random.choice(list(rois))

        transf = RoiIndexer(mvp=mvp_within, mask=roi, mask_threshold=0,
                            reg_dir=reg_dir,
                            atlas_name=atlas)
        transf.fit(mvp_within.X, mvp_within.y)
        transf.transform(mvp_within.X)

    files_reg = glob(op.join(reg_dir, '*'))
    [os.remove(f) for f in files_reg if f not in orig_reg_files]


@pytest.mark.transformer
def test_average_region_transformer():

    for atlas in available_atlases:

        art = AverageRegionTransformer(mvp=mvp_within, atlas=atlas,
                                       reg_dir=reg_dir)
        art.fit(mvp_within.X, mvp_within.y)
        art.transform(mvp_within.X)

    files_reg = glob(op.join(reg_dir, '*'))
    [os.remove(f) for f in files_reg if f not in orig_reg_files]
