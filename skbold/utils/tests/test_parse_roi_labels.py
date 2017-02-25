import pytest
from skbold.utils import parse_roi_labels

available_atlases = ['HarvardOxford-Cortical',
                     'HarvardOxford-Subcortical',
                     # 'HarvardOxford-All',
                     'MNI', 'JHU-labels', 'JHU-tracts',
                     'Talairach', 'Yeo2011']


@pytest.mark.parametrize("atlas", available_atlases)
def test_parse_roi_labels(atlas):

    info_dict = parse_roi_labels(atlas, debug=False)
    assert isinstance(info_dict, dict)
    assert(len(values) == 2 for values in info_dict.values())
    assert(isinstance(values[0], int) and isinstance(values[1], tuple)
           for values in info_dict.values())
