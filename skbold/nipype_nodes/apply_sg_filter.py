# Implementation of savitsky-golay filter in a format compatible
# with a Nipype node.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function


def apply_sg_filter(in_file, polyorder=5, deriv=0):
    """
    Applies a savitsky-golay filter to a 4D fMRI
    nifti file. The window is computed using the
    TR from the nifti header.

    Modified from H.S. Scholte's OP2_filtering().
    """

    import nibabel as nib
    from scipy.signal import savgol_filter
    import numpy as np
    import os

    data = nib.load(in_file)
    dims = data.shape
    affine = data.affine
    tr = data.header['pixdim'][4]

    if tr < 0.01:
        tr = np.round(tr * 1000, decimals=3)

    window = np.int(200 / tr)

    # Window must be odd
    if window % 2 == 0:
        window += 1

    data = data.get_data().reshape((np.prod(data.shape[:-1]), data.shape[-1]))
    data_filt = savgol_filter(data, window_length=window, polyorder=polyorder,
                              deriv=deriv, axis=1)

    data_filt = data-data_filt
    data_filt = data_filt.reshape(dims)
    img = nib.Nifti1Image(data_filt, affine)
    new_name = os.path.basename(in_file).split('.')[:-2][0] + '_sg.nii.gz'
    out_file = os.path.abspath(new_name)
    nib.save(img, out_file)

    return out_file
