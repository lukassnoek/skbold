import pytest
from ..roi_globals import available_atlases
from ...utils import parse_roi_labels


@pytest.mark.parametrize("atlas", available_atlases)
@pytest.mark.parametrize("lateralized", [True, False])
def test_parse_roi_labels(atlas, lateralized):

    info_dict = parse_roi_labels(atlas, lateralized=lateralized, debug=False)
    assert isinstance(info_dict, dict)
    assert(len(values) == 2 for values in info_dict.values())
    assert(isinstance(values[0], int) and isinstance(values[1], tuple)
           for values in info_dict.values())
