# Function to parse FSL-type xml's with info about atlas labels.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import os.path as op
from skbold import roidata_path
from glob import glob


def parse_roi_labels(atlas_type='Talairach'):
    """ Parses xml-files belonging to FSL atlases.

    Parameters
    ----------
    atlas_type : str
        String identifying which atlas needs to be parsed.

    Returns
    -------
    info_dict : dict
        Dictionary with indices and coordinates (values) per
        ROI (keys).
    """
    available_atlases = ['HarvardOxford-Cortical',
                         'HarvardOxford-Cortical-Lateralized',
                         'HardvardOxford-Subcortical',
                         'MNI', 'JHU-labels', 'JHU-tracts',
                         'Talairach']

    roidata_root = roidata_path
    # roidata_root = '../data/ROIs'

    xml = glob(op.join(roidata_root, '*', atlas_type + '.xml'))

    if xml:
        xml = xml[0]
    else:
        msg = "%s not found in atlases. Please pick from: %r" %\
              (atlas_type, available_atlases)
        raise ValueError(msg)

    with open(xml, 'rb') as fin:
        doc = fin.readlines()

    if type == 'Talairach':
        raw_labels = [label for label in doc if 'Brodmann' in label]
    else:
        raw_labels = [label for label in doc if 'label index' in label]

    indices = [int(s.split(' ')[1].split('=')[1].replace('"', '')) + 1
               for s in raw_labels]
    xs = [int(s.split(' ')[2].split('=')[1].replace('"', ''))
          for s in raw_labels]
    ys = [int(s.split(' ')[3].split('=')[1].replace('"', ''))
          for s in raw_labels]
    zs = [int(s.split(' ')[4].split('=')[1].split('>')[0].replace('"', ''))
          for s in raw_labels]
    coords = zip(xs, ys, zs)
    rois = [s.split('>')[1].split('<')[0] for s in raw_labels]

    info_dict = {roi: (idx, crds) for roi, idx, crds
                 in zip(rois, indices, coords)}

    if type == 'Talairach':
        info_dict = {(key.split(' ')[0] + ' ' + key.split('.')[-1]): values
                     for key, values in info_dict.items()}

    return info_dict


def merge_ho_atlases():
    # merges subcortical and cortical masks
    probs = ['0', '25', '50']

    pass


