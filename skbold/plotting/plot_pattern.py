# Class to select features based on their mean euclidean distance between
# average class values.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from nilearn.plotting import plot_stat_map
from skbold import harvardoxford_path
import os.path as op
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

def plot_region(mask_name, filepath=None, pattern_type='random', hemisphere='bilateral',
                mean=5, std=2, mask_threshold=20, orientation='x', cut=None, colorbar=False):
    """ Plots a brain region on top of an MNI152 brain for illustrative purposes.

    Plots a color-coded region, or set of regions, on top of an MNI152 (2mm) brain,
    which can be used for illustrative purposes. The 'fill' of a region can
    be random ('pattern-like'), uniform (one color), or smooth (fading
    towards edges).

    Parameters
    ----------
    mask_name : str (or list of str)
        Name of mask(s); drawn from Harvard-Oxford cortical atlas. 

    """

    if filepath is None:
        filepath = op.join(os.getcwd(), 'roiplot_%s_mean%i_std%i' % (pattern_type, mean, std))

    if pattern_type not in ['random', 'uniform', 'smooth']:
        opts = ['random', 'uniform', 'smooth']
        raise ValueError('For type, choose one of the following: %r' % opts)

    if hemisphere in ['left', 'right']:
        laterality = 'unilateral'
        prefix = 'L_' if hemisphere == 'left' else 'R_'
    elif hemisphere == 'bilateral':
        laterality = 'bilateral'
        prefix = ''
    else:
        opts = ['bilateral', 'left', 'right']
        raise ValueError('Please choose one of the following for hemisphere: %r' % opts)

    if type(mask_name) == str:
        mask_name = [mask_name]

    to_plot = np.zeros((91, 109, 91))
    maskdir_path = op.join(harvardoxford_path, laterality)

    for m in mask_name:
        paths = glob(op.join(maskdir_path, '*%s*%s*' % (prefix, m[1:])))

        if len(paths) == 0:
            raise ValueError("No mask '%s' found in %s" % (m, maskdir_path))
        elif len(paths) > 1:
            raise ValueError("Too many masks found in %s: %r" % (maskdir_path, paths))
        else:
            m = paths[0]

        mask = nib.load(m)
        affine = mask.affine
        data = mask.get_data()
        n_vox = np.sum(data > mask_threshold)

        if pattern_type == 'random':
            rand = np.random.normal(mean, std, n_vox)
            to_plot[data > mask_threshold] = rand + mean
        elif pattern_type == 'uniform':
            to_plot[data > mask_threshold] = mean
        elif pattern_type == 'smooth':
            to_plot[data > mask_threshold] = data[data > mask_threshold] * 1.1

    img = nib.Nifti1Image(np.ma.array(to_plot, mask=None), affine)
    plot_stat_map(img, annotate=False, draw_cross=False, display_mode=orientation,
                  cut_coords=cut, colorbar=colorbar, vmax=100)
    plt.savefig(filepath)


if __name__ == '__main__':
    masks = ['Angular_gyrus', 'Temporal_pole', 'Supplementary']
    plot_region(masks, filepath='/home/lukas/testtest.png', hemisphere='left', pattern_type='random', mean=20, std=20,
                orientation='ortho', mask_threshold=20, cut=None, colorbar=False)

