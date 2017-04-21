# Function to parse FSL-type xml's with info about atlas labels.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function, absolute_import
from io import open
import skbold
import os.path as op
from skbold.utils.roi_globals import available_atlases

roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs')


def parse_roi_labels(atlas_type='Talairach', lateralized=False, debug=False):
    """ Parses xml-files belonging to FSL atlases.

    Parameters
    ----------
    atlas_type : str
        String identifying which atlas needs to be parsed.
    lateralized : bool
        Whether to use the lateralized version of the atlas (only applicable
        to HarvardOxford masks)

    Returns
    -------
    info_dict : dict
        Dictionary with indices and coordinates (values) per
        ROI (keys).
    """

    if atlas_type not in available_atlases:
        msg = "%s not found in atlases. Please pick from: %r" % \
              (atlas_type, available_atlases)
        raise ValueError(msg)

    if debug:
        roidata_root = '../data/ROIs'
    else:
        roidata_root = roi_dir

    if atlas_type == 'Yeo2011':
        info_dict = {'Network_%i' % i: (i, (0, 0, 0)) for i in range(1, 8)}
        return info_dict

    if lateralized and atlas_type == 'HarvardOxford-Cortical':
        xml = op.join(roidata_root, atlas_type,
                      atlas_type + '-Lateralized.xml')
    elif not lateralized and atlas_type == 'HarvardOxford-Subcortical':
        info_dict = {'Cerebral White Matter': (0, (0, 0, 0)),
                     'Cerebral Cortex': (1, (0, 0, 0)),
                     'Lateral Ventricle': (2, (0, 0, 0)),
                     'Thalamus': (3, (0, 0, 0)),
                     'Caudate': (4, (0, 0, 0)),
                     'Putamen': (5, (0, 0, 0)),
                     'Pallidum': (6, (0, 0, 0)),
                     'Brain-Stem': (7, (44, 49, 18)),
                     'Hippocampus': (8, (0, 0, 0)),
                     'Amygdala': (9, (0, 0, 0)),
                     'Accumbens': (10, (0, 0, 0))}
        return info_dict
    else:
        xml = op.join(roidata_root, atlas_type, atlas_type + '.xml')

    with open(xml, 'r') as fin:
        doc = fin.readlines()

    if atlas_type == 'Talairach':
        raw_labels = [label for label in doc if 'Brodmann' in label]
    else:
        raw_labels = [label for label in doc if 'label index' in label]

    rois = [s.split('>')[1].split('<')[0] for s in raw_labels]
    rois = [r.rstrip() for r in rois]
    raw_labels = [[slab for slab in label.split(' ') if slab]
                  for label in raw_labels]
    indices = [int(si[1].split('=')[1].replace('"', ''))
               for si in raw_labels]
    xs = [int(sx[2].split('=')[1].replace('"', ''))
          for sx in raw_labels]
    ys = [int(sy[3].split('=')[1].replace('"', ''))
          for sy in raw_labels]
    zs = [int(sz[4].split('=')[1].split('>')[0].replace('"', ''))
          for sz in raw_labels]
    coords = zip(xs, ys, zs)
    info_dict = {roi: (idx, crds) for roi, idx, crds
                 in zip(rois, indices, coords)}

    if atlas_type == 'Talairach':
        info_dict = {(key.split(' ')[0] + ' ' + key.split('.')[-1]): values
                     for key, values in info_dict.items()}

    return info_dict
