# Function to convert a nifti in MNI space to Epi-space. Only works with
# reg_dir as created by FSL.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import os
import os.path as op
from nipype.interfaces import fsl


def convert2epi(file2transform, reg_dir, out_dir=None,
                interpolation='trilinear', suffix='epi'):
    """
    Transforms a nifti from mni152 (2mm) to EPI (native) format.
    Assuming that reg_dir is a directory with transformation-files (warps)
    including standard2example_func warps, this function uses nipype's
    fsl interface to flirt a nifti to EPI format.

    Parameters
    ----------
    file2transform : str or list
        Absolute path(s) to nifti file(s) that needs to be transformed
    reg_dir : str
        Absolute path to registration directory with warps
    out_dir : str
        Absolute path to desired out directory. Default is same directory as
        the to-be transformed file.
    interpolation : str
        Interpolation used by flirt. Default is 'trilinear'.

    Returns
    -------
    out_all : list
        Absolute path(s) to newly transformed file(s).
    """

    if type(file2transform) == str:
        file2transform = [file2transform]

    out_all = []

    for f in file2transform:

        if out_dir is None:
            out_dir = op.dirname(f)

        if suffix is not None:
            out_name = op.basename(f).split('.')[0] + '_%s.nii.gz' % suffix
        else:
            out_name = op.basename(f)

        out_file = op.join(out_dir, out_name)

        if op.exists(out_file):
            out_all.append(out_file)
            continue

        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        out_matrix_file = op.join(op.dirname(out_file), 'tmp_flirt')
        ref_file = op.join(reg_dir, 'example_func.nii.gz')
        matrix_file = op.join(reg_dir, 'standard2example_func.mat')
        apply_xfm = fsl.ApplyXfm()
        apply_xfm.inputs.in_file = f
        apply_xfm.inputs.reference = ref_file
        apply_xfm.inputs.in_matrix_file = matrix_file
        apply_xfm.inputs.out_matrix_file = out_matrix_file
        apply_xfm.interp = interpolation
        apply_xfm.inputs.out_file = out_file
        apply_xfm.inputs.apply_xfm = True
        apply_xfm.run()

        if op.exists(out_matrix_file):
            os.remove(out_matrix_file)

        out_all.append(out_file)

    if len(out_all) == 1:
        out_all = out_all[0]

    return out_all

