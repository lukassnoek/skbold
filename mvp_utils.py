import numpy as np
import fnmatch
import glob
import os
import pandas as pd
from os.path import join as opj
import nipype.interfaces.fsl as fsl


def sort_numbered_list(stat_list):
    """
    Sorts list with paths to statistic files (e.g. COPEs, VARCOPES),
    which are often sorted wrong (due to single and double digits).
    This function extracts the numbers from the stat files and sorts
    the original list accordingly.

    Args:
        stat_list: list with paths to files

    Returns:
        sorted_list: sorted stat_list
    """

    num_list = []
    for path in stat_list:
        num = [str(s) for s in str(os.path.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))

    sorted_list = [x for y, x in sorted(zip(num_list, stat_list))]
    return sorted_list


def convert_labels2numeric(class_labels, grouping):
    """
    Converts class labels (list of strings) to numeric numpy row vector.
    Groups string labels based on grouping, which is useful in factorial
    designs.

    Args:
        class_labels: list of strings (returned from extract_class_labels())
        grouping: list of strings that indicate grouping

    Returns:
        num_labels: numeric labels corresponding to class_labels
    """

    if len(grouping) == 0:
        grouping = np.unique(class_labels)

    num_labels = np.zeros(len(class_labels))
    for i, group in enumerate(grouping):

        if type(group) == list:
            matches = []
            for g in group:
                matches.append(fnmatch.filter(class_labels, '*%s*' % g))
            matches = [x for y in matches for x in y]
        else:
            matches = fnmatch.filter(class_labels, '*%s*' % group)
            matches = list(set(matches))

        for match in matches:
            for k, lab in enumerate(class_labels):
                if match == lab:
                    num_labels[k] = i + 1

    return np.array(num_labels)


def extract_class_vector(sub_path, ignore):
    """
    Extracts class-name of each trial and returns a vector of class labels.

    Args:
        sub_path: path to subject-specific first-level directory
        ignore: name of contrasts that do not need to be read in (e.g. cues,
        nuisance regressors)

    Returns:
        class_labels: list of class-labels (strings)

    Raises:
        IOError: if design.con file doesn't exist
    """

    sub_name = os.path.basename(os.path.normpath(sub_path))
    design_file = opj(sub_path, 'design.con')

    if not os.path.isfile(design_file):
        raise IOError('There is no design.con file for %s' % sub_name)

    # Find number of contrasts and read in accordingly
    contrasts = sum(1 if 'ContrastName' in line else 0 for line in open(design_file))
    n_lines = sum(1 for line in open(design_file))

    df = pd.read_csv(design_file, delimiter='\t', header=None,
                     skipfooter=n_lines-contrasts, engine='python')

    class_labels = list(df[1])

    # Remove to-be-ignored contrasts (e.g. cues)
    remove_idx = np.zeros((len(class_labels), len(ignore)))

    for i, name in enumerate(ignore):
        matches = [name in label for label in class_labels]
        remove_idx[:, i] = np.array(matches)

    remove_idx = np.where(remove_idx.sum(axis=1).astype(int))[0]
    _ = [class_labels.pop(idx) for idx in np.sort(remove_idx)[::-1]]
    class_labels = [s.split('_')[0] for s in class_labels]

    return class_labels, remove_idx


def transform2mni(stat_paths, varcopes, sub_path):
    """
    Transforms (VAR)COPEs to MNI space.

    Args:
        stat_paths: list with paths to COPEs
        varcopes: list with paths to VARCOPEs
        sub_path: path to first-level directory

    Returns:
        stat_paths: transformed COPEs
        varcopes: transformed VARCOPEs
    """

    os.chdir(sub_path)
    print "Transforming COPES to MNI for %s." % sub_path
    ref_file = opj(sub_path, 'reg', 'standard.nii.gz')
    field_file = opj(sub_path, 'reg', 'example_func2standard_warp.nii.gz')
    out_dir = opj(sub_path, 'reg_standard')

    for stat, varc in zip(stat_paths, varcopes):

        out_file = opj(out_dir, os.path.basename(stat))
        apply_warp = fsl.ApplyWarp()
        apply_warp.inputs.in_file = stat
        apply_warp.inputs.ref_file = ref_file
        apply_warp.inputs.field_file = field_file
        apply_warp.interp = 'trilinear'
        apply_warp.inputs.out_file = out_file
        apply_warp.run()

        out_file = opj(out_dir, os.path.basename(varc))
        apply_warp = fsl.ApplyWarp()
        apply_warp.inputs.in_file = varc
        apply_warp.inputs.ref_file = ref_file
        apply_warp.inputs.field_file = field_file
        apply_warp.interp = 'trilinear'
        apply_warp.inputs.out_file = out_file
        apply_warp.run()

    stat_dir = opj(sub_path, 'reg_standard')
    stat_paths = glob.glob(opj(stat_dir, 'cope*'))
    stat_paths = sort_numbered_list(stat_paths) # see function below

    varcopes = glob.glob(opj(stat_dir, 'varcope*'))
    varcopes = sort_numbered_list(varcopes) # see function below

    os.chdir('..')

    return stat_paths, varcopes
