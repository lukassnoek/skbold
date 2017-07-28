from __future__ import absolute_import, division, print_function

import os.path as op
from ...core import MvpWithin
from ... import testdata_path
from ..misc_transformers import *
import pytest
import os
import random
from glob import glob

testfeats = [op.join(testdata_path, 'run1.feat'),
             op.join(testdata_path, 'run2.feat')]


mvp_within = MvpWithin(source=testfeats, read_labels=True,
                       remove_contrast=[], invert_selection=False,
                       ref_space='epi', statistic='cope', remove_zeros=False,
                       mask=None)

mvp_within.create()


@pytest.mark.transformer
def test_array_permuter():

    transf = ArrayPermuter()
    transf.fit(mvp_within.X, mvp_within.y)
    transf.transform(mvp_within.X)


@pytest.mark.transformer
def test_row_indexer():
    idx = np.arange(len(mvp_within.contrast_labels))[::2] 
    transf = RowIndexer(mvp_within, idx)
    mvp, sel, nsel = transf.transform()
