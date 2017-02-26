import pytest
from skbold.utils.roi_globals import available_atlases, other_rois
from skbold.utils import load_roi_mask, parse_roi_labels


@pytest.mark.parametrize("atlas_name", available_atlases)
@pytest.mark.parametrize("resolution", ['2mm'])#,'1mm'])
@pytest.mark.parametrize("lateralized", [False, True])
@pytest.mark.parametrize("threshold", [0, 25])
@pytest.mark.parametrize("maxprob", [True, False])
def test_load_roi_mask_from_atlas(atlas_name, resolution, lateralized,
                                  threshold, maxprob):

    info_dict = parse_roi_labels(atlas_type=atlas_name, lateralized=lateralized)
    rois = info_dict.keys()

    for roi in rois:
        mask = load_roi_mask(roi, atlas_name=atlas_name, resolution=resolution,
                             lateralized=lateralized, which_hemifield='left',
                             threshold=threshold, maxprob=maxprob)
