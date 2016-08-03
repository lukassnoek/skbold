from __future__ import division
import numpy as np
import nibabel as nib
import pandas as pd
import os.path as op
import glob


def check_mc_output(directory, sub_id='sub', cutoff_spike=1,
                    output_dir=None, split=None):
    """
    Check motion-correction output and generate summary statistics.

    Parameters
    ----------
    directory : str
        Project-root containing subject-directories
    sub_id : str
        Subject-identifier (prefix)
    cutoff_spike : int
        Minimal movement from one TR to another to count as a 'spike'.
    output_dir : str
        Directory to save output to.

    Returns
    -------
    df : Dataframe
        Dataframe with mc summary statistics
    """

    sub_dirs = glob.glob(op.join(directory, '%s*' % sub_id))
    df_list = []

    for sub_dir in sub_dirs:

        sub_name = op.basename(sub_dir)
        task_dirs = glob.glob(op.join(sub_dir, '*', 'mc'))

        for task_dir in task_dirs:

            disp_file = glob.glob(op.join(task_dir, '*mcf_abs.rms'))[0]
            disp = np.loadtxt(disp_file)

            disp_std = disp.std()
            disp_diff = disp.max() - disp.min()
            disp_spikes = pd.Series(disp).diff()
            max_spike = disp_spikes.max()
            nr_spikes = np.sum(disp_spikes > cutoff_spike)

            task_name = op.basename(op.dirname(task_dir))
            if split:
                task_name = task_name.split(split)[-1]

            X = np.c_[np.ones(disp.size), np.arange(disp.size)]
            drift = np.linalg.lstsq(X, disp)[0] # only slope

            df_list.append({'Subject': sub_name, 'Task': task_name,
                            'Max-min': disp_diff, 'Stdev': disp_std,
                            'Max. spike': max_spike, 'Nr. spikes': nr_spikes,
                            'Drift': drift[1]})

    df = pd.DataFrame(df_list).sort_values(by='Subject')
    df.set_index(['Subject', 'Task'], inplace=True)
    df['Rec. conservative'] = np.all(np.c_[df['Max-min'] < 1.5, df['Nr. spikes'] == 0], axis=1).astype(int)
    df['Rec. liberal'] = (df['Max-min'] < 1.5).astype(int)

    print(df)
    output_dir = output_dir if output_dir is not None else directory
    out_name = op.join(output_dir, 'check_mc_stats.tsv')
    df.to_csv(out_name, sep='\t', index=True)

    return df


def check_nifti_header(directory, sub_id='sub', task_id='func',
                       func_id='mcst_sg', calc_zeros=False,
                       output_dir=None):
    """
    Creates dataframe with scan params to spot abnormalities.

    Parameters
    ----------
    directory : str
        Directory with subject-directories to check
    sub_id : str
        Subject-identifier
    task_id : str or [str]
        Task-identifier (can be list of strings)
    func_id : str
        Identifier for functional file name
    calc_zeros : bool
        Whether to calculate the number of voxels which are zero (e.g. outside mask);
        Takes significantly longer if true, because has to read in 4D func file.
    output_dir : str
        Directory to save .tsv with scan-parameters (default: same as
        directory arg).
    """

    if isinstance(task_id, str):
        task_id = [task_id]

    sub_dirs = glob.glob(op.join(directory, '%s*' % sub_id))

    df_list = []
    for sub_dir in sub_dirs:

        task_dirs = []
        _ = [task_dirs.extend(glob.glob(op.join(sub_dir, '*%s*' % tid)))
             for tid in task_id]

        for task_dir in task_dirs:

            task_name = op.basename(task_dir)

            print('Task: %s' % task_name)

            func = glob.glob(op.join(task_dir, '*%s*' % func_id))

            if len(func) == 1:
                img = nib.load(func[0])
            elif len(func) > 1:
                msg = 'More than one func-file found in %s' % task_dir
                raise ValueError(msg)
            else:
                msg = 'No func-file found in %s!' % task_dir

            tr = img.header.get_zooms()[-1]
            (dim1, dim2, n_slices, n_dynamics) = img.header['dim'][1:5]

            if calc_zeros:
                zero_voxels = np.sum(img.get_data()==0) / n_dynamics
            else:
                zero_voxels = 0

            df_tmp = {'Subject': op.basename(sub_dir), 'Task': task_name,
                      'TR': tr, 'zero_voxels': zero_voxels, 'n_slices': n_slices,
                      'dim1': dim1, 'dim2': dim2, 'n_dynamics': n_dynamics}

            df_list.append(pd.DataFrame(df_tmp, index=[0]))

    df = pd.concat(df_list)

    if not calc_zeros:
        df.drop('zero_voxels', inplace=True, axis=1)

    print(df)
    output_dir = output_dir if output_dir is not None else directory
    out_name = op.join(output_dir, 'check_MR_params.tsv')
    df.to_csv(out_name, sep='\t', index=False)