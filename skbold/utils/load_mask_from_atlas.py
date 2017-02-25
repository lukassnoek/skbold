# Function to create a mask from a roi from a given atlas.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function, absolute_import
import skbold
import os.path as op
import nibabel as nib
import numpy as np
from glob import glob
from skbold.utils.parse_roi_labels import parse_roi_labels


roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs')
available_atlases = ['HarvardOxford-Cortical',
                     'HarvardOxford-Subcortical',
                     # 'HarvardOxford-All',
                     'MNI', 'JHU-labels', 'JHU-tracts',
                     'Talairach', 'Yeo2011']

other_rois = {'GrayMatter_prob': op.join(roi_dir, 'other',
                                         'GrayMatter_prob.nii.gz'),
              'MNI152_1mm': op.join(roi_dir, 'MNI152_1mm.nii.gz'),
              'MNI152_2mm': op.join(roi_dir, 'MNI152_2mm.nii.gz'),
              'VentricleMask': op.join(roi_dir, 'other',
                                       'VentricleMask.nii.gz'),
              'GrayMatter_tp': op.join(roi_dir, 'tissuepriors',
                                       'avg152T1_gray.img'),
              'WhiteMatter_tp': op.join(roi_dir, 'tissuepriors',
                                        'avg152T1_white.img'),
              'CSF_tp': op.join(roi_dir, 'tissuepriors', 'avg152T1_csf.img')}


def load_mask_from_atlas(roi_name, atlas_name=None, resolution='2mm',
                         lateralized=False, which_hemifield=None,
                         threshold=0, maxprob=False):

    """ Loads a mask (from an atlas).

    Parameters
    ----------
    roi_name : str
        Name of the ROI (as specified in the FSL XML-files)
    atlas_name : str
        Name of the atlas. Choose from: 'HarvardOxford-Cortical',
        'HarvardOxford-Subcortical', 'HarvardOxford-All',
        'MNI', 'JHU-labels', 'JHU-tracts', 'Talairach', 'Yeo2011'.
    resolution : str
        Resolution of the mask/atlas ('1mm' or '2mm')
    lateralized : bool
        Whether to use lateralized masks (only available for Harvard-
        Oxford atlases). If this variable is specified, you have to specify
        which_hemifield too.
    threshold : int
        Threshold for probabilistic masks (everything below this threshold is
        set to zero before creating the mask).
    maxprob : bool
        Whether to select only the voxels that have the highest probability
        of that particular ROI for a given threshold. Setting this option to
        true ensures that each mask has unique voxels (substantially slows
        down the function, though).

    ToDo
    ----
    - Check maxprob only for probabilistic atlases!
    - Check lateralization for other atlases!
    """

    # Trick to parse all the ROIs in a mask!
    if roi_name is 'all':
        # ToDo: make a generator out of this to save memory?
        roi_name = sorted(parse_roi_labels(atlas_name, lateralized=lateralized,
                                          debug=True).keys())

    # First time of a (doubtful) use of a recursive function, yay!
    if isinstance(roi_name, list):
        return [load_mask_from_atlas(roi_n, atlas_name, resolution,
                                     lateralized, which_hemifield, threshold,
                                     maxprob) for roi_n in roi_name]
    else:
        print('Trying to load mask: %s' % roi_name)

    # Check compatibility settings
    _check_cfg(roi_name, atlas_name, lateralized, which_hemifield)

    # If roi is just a simple ROI-file, then load it and return it
    if roi_name in other_rois.keys():
        roi = nib.load(other_rois[roi_name], mmap=False).get_data()
        mask = roi > threshold
        return mask

    # Stupid hack to find correct atlas file when filenames vary too much
    lat_str = 'lateralized' if lateralized and 'HarvardOxford' in atlas_name else ''
    atlas = glob(op.join(roi_dir, atlas_name, '*%s*%s.nii.gz' % (lat_str, resolution)))

    # Another hack to get the lateralized atlas if queried
    if lateralized:
        if 'JHU' in atlas_name and roi_name.split(' ')[-1] not in ['L', 'R']:
            roi_name = roi_name + ' L' if which_hemifield == 'left' else ' R'

        elif 'JHU' not in atlas_name and roi_name.split(' ')[0] not in ['Left', 'Right']:
            roi_name = '%s %s' % (which_hemifield[0].upper() +
                                  which_hemifield[1:].lower(), roi_name)
    if len(atlas) > 1:
        msg = "Found more than one atlas, namely: %r" % atlas
        raise ValueError(msg)
    elif len(atlas) == 0:
        msg = "Didn't find any atlas at all ..."
        raise ValueError(msg)
    else:
        atlas = atlas[0]

    atlas_img = nib.load(atlas, mmap=False)
    info_dict = parse_roi_labels(atlas_name, lateralized=lateralized,
                                 debug=True)
    idx = info_dict[roi_name][0]

    # ToCheck: only maxprob with probabilistic atlases
    if maxprob:
        atlas_loaded = atlas_img.get_data()
        atlas_loaded[atlas_loaded < threshold] = 0
        atlas_maxprob = np.argmax(atlas_loaded, axis=3)
        mask = atlas_maxprob == idx
    else:
        roi = np.asarray(atlas_img.dataobj[..., idx])
        mask = roi > threshold

    return mask


def _check_cfg(roi_name, atlas_name, lateralized, which_hemifield):

    if roi_name not in other_rois.keys() and atlas_name is None:
        msg = ("Could not find your ROI and you also didn't specify an "
               "atlas in which it could be. Please specify the appropriate"
               " atlas or choose from: %r" % other_rois.keys())
        raise ValueError(msg)

    if atlas_name is not None and atlas_name not in available_atlases:
        msg = ("Could not find your specified atlas! "
               "Please choose from: %r" % available_atlases)
        raise ValueError(msg)

    if lateralized and which_hemifield is None:
        msg = ("You specified a lateralized mask, but you haven't indicated "
               "which hemifield. Please set which_hemifield={'left', 'right}.")
        raise ValueError(msg)

    if lateralized and atlas_name == 'Yeo2011':
        msg = ("The Yeo2011 is by definition bilateral! Cannot use "
               "lateralized masks. Set lateralization to False.")
        raise ValueError(msg)

if __name__ == '__main__':

    mmask = load_mask_from_atlas('all', 'JHU-labels',
                         resolution='2mm', lateralized=True, which_hemifield='right',
                         threshold=5, maxprob=False)
