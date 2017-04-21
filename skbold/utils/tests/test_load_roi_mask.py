import pytest
import os
import os.path as op
from glob import glob
from ..roi_globals import available_atlases, other_rois
from ...utils import load_roi_mask, parse_roi_labels
from ... import testdata_path

reg_dir_test = op.join(testdata_path, 'run1.feat', 'reg')
reg_files = ['example_func.nii.gz', 'example_func2standard.mat',
             'example_func2standard.nii.gz', 'standard.nii.gz',
             'standard2example_func.mat']


@pytest.mark.parametrize("atlas_name", available_atlases)
@pytest.mark.parametrize("resolution", ['2mm'])  # Testing 1mm takes too long
@pytest.mark.parametrize("lateralized", [False, True])
@pytest.mark.parametrize("threshold", [0, 25])
@pytest.mark.parametrize("reg_dir", [None, reg_dir_test])
# @pytest.mark.parametrize("maxprob", [True, False])  # TAKES VERY LONG
def test_load_roi_mask_from_atlas(atlas_name, resolution, lateralized,
                                  threshold, reg_dir):
    maxprob = False  # Hardcoded to test
    info_dict = parse_roi_labels(atlas_type=atlas_name,
                                 lateralized=lateralized)
    rois = info_dict.keys()

    for roi in rois:
        mask, name = load_roi_mask(roi, atlas_name=atlas_name,
                                   resolution=resolution,
                                   lateralized=lateralized,
                                   which_hemifield='left',
                                   threshold=threshold, maxprob=maxprob,
                                   reg_dir=reg_dir)

        if mask is not None:
            assert(mask.ndim == 3)
            assert(mask.sum() > 0)

    masks = glob(op.join(reg_dir_test, '*nii.gz'))
    for mask in masks:

        if op.basename(mask) not in reg_files:
            os.remove(mask)


@pytest.mark.parametrize("roi_name", other_rois)
@pytest.mark.parametrize("threshold", [0, 25])
@pytest.mark.parametrize("reg_dir", [None, reg_dir_test])
def test_load_roi_mask_from_other_rois(roi_name, threshold, reg_dir):

    mask = load_roi_mask(roi_name, threshold=threshold, reg_dir=reg_dir)
    masks = glob(op.join(reg_dir_test, '*nii.gz'))

    for mask in masks:

        if op.basename(mask) not in reg_files:
            os.remove(mask)
