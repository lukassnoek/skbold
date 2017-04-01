# Function to convert a nifti in MNI space to Epi-space. Only works with
# reg_dir as created by FSL.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function, absolute_import
import os
import os.path as op
import subprocess


def convert2epi(file2transform, reg_dir, out_dir=None,
                interpolation='trilinear', suffix='epi',
                overwrite=False):
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
    suffix : str
        What to suffix the transformed file with (default : 'epi')
    overwrite : bool
        Whether to overwrite existing transformed files

    Returns
    -------
    out_all : list
        Absolute path(s) to newly transformed file(s).
    """

    if not isinstance(file2transform, list):
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

        if op.exists(out_file) and not overwrite:
            out_all.append(out_file)
            continue

        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        # out_matrix_file = op.join(op.dirname(out_file), 'tmp_flirt')
        ref_file = op.join(reg_dir, 'example_func.nii.gz')
        matrix_file = op.join(reg_dir, 'standard2example_func.mat')
        warp_file = op.join(reg_dir, 'standard2example_func_warp.nii.gz')
        if op.isfile(warp_file):
            cmd = 'applywarp -i %s -r %s -o %s -w %s --interp=%s' % \
                  (f, ref_file, out_file, warp_file, interpolation)
        else:
            cmd = ('flirt -in %s -ref %s -out %s -applyxfm -init %s '
                   '-interp %s' % (f, ref_file, out_file, matrix_file,
                                   interpolation))

        status = subprocess.call(cmd, shell=True)

        out_all.append(out_file)
        out_name = None

    if len(out_all) == 1:
        out_all = out_all[0]

    return out_all
