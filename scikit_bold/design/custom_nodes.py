"""
Some functions that can be used as 'Nodes' within a
Nipype (fMRI preprocessing) workflow.

Lukas Snoek, University of Amsterdam
"""

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


def mcflirt_across_runs(in_file, cost='mutualinfo', stages=3):
    """
    Performs motion correction on the 'middle run' and subsequently
    does motion correction to the left over runs relative to the
    middle run.

    Modified from H.S. Scholte's OP2 processing pipeline
    """

    import numpy as np
    from nipype.interfaces.fsl import MCFLIRT
    import os
    import nibabel as nib

    mc_files = []
    plot_files = []
    run_nrs = []

    for f in in_file:
        parts = np.array(f.split('_'))
        idx = parts[np.where(parts == 'SENSE')[0] + 1]
        run_nrs.append(int(idx[0]))

    middle_idx = np.floor(np.median(run_nrs)) == run_nrs
    middle_run = [f for idx, f in zip(middle_idx, in_file) if idx]
    middle_run = middle_run[0]
    other_runs = [f for f in in_file if f != middle_run]

    middle_data = nib.load(middle_run)
    middle_vol_idx = np.round(middle_data.shape[3] / 2).astype(int)

    new_name = os.path.basename(middle_run).split('.')[:-2][0] + '_mc.nii.gz'
    out_name = os.path.abspath(new_name)

    mcflt = MCFLIRT(in_file=middle_run, cost=cost, ref_vol=middle_vol_idx,
                    interpolation='sinc', out_file=out_name, stages=stages,
                    save_plots=True)

    results = mcflt.run(in_file=middle_run)

    mc_files.append(out_name)
    plot_files.append(results.outputs.par_file)

    mean_bold_path = os.path.abspath('mean_bold.nii.gz')
    _ = os.system('fslmaths %s -Tmean %s' % (out_name, mean_bold_path))

    for other_run in other_runs:

        new_name = os.path.basename(other_run).split('.')[:-2][0] + '_mc.nii.gz'
        out_name = os.path.abspath(new_name)

        mcflt = MCFLIRT(in_file=other_run, cost=cost, ref_file=mean_bold_path,
                        interpolation='sinc', out_file=out_name, stages=stages,
                        save_plots=True)

        results = mcflt.run()
        mc_files.append(out_name)
        plot_files.append(results.outputs.par_file)

    out_files = mc_files
    mean_bold = mean_bold_path

    return out_files, mean_bold, plot_files


def parse_presentation_logfile(in_file, con_names, con_codes, con_design=None,
                               pulsecode=30, write_bfsl=False, verbose=False):
    """
    Parses a Presentation-logfile and extracts stimulus/event times and
    durations given their corresponding codes in the logfile.

    To do: build feature to input list of strings for codes
    (e.g. ['anger', 'sadness', 'disgust'] --> name: 'negative';
    see custom piopfaces logfile crawler for example
    """

    from nipype.interfaces.base import Bunch
    import pandas as pd
    import numpy as np
    import os
    import glob

    if verbose:
        print('Processing %s' % in_file)

    base_dir = os.path.dirname(in_file)
    _ = [os.remove(x) for x in glob.glob(os.path.join(base_dir, '*.bfsl'))]

    if not con_design:
        con_design = ['univar'] * len(con_names)

    df = pd.read_table(in_file, sep='\t', skiprows=3, header=0,
                       skip_blank_lines=True)

    # Convert to numeric and drop all rows until first pulse
    df['Code'] = df['Code'].astype(str)
    df['Code'] = [np.float(x) if x.isdigit() else x for x in df['Code']]
    pulse_idx = np.where(df['Code'] == pulsecode)[0]

    if len(pulse_idx) > 1:
        pulse_idx = int(pulse_idx[0])

    df = df.drop(range(pulse_idx))

    # Clean up unnecessary columns
    df.drop(['Uncertainty', 'Subject', 'Trial', 'Uncertainty.1', 'ReqTime',
             'ReqDur', 'Stim Type', 'Pair Index'], axis=1, inplace=True)

    # pulse_t = absolute time of first pulse
    pulse_t = df['Time'][df['Code'] == pulsecode].iloc[0]
    df['Time'] = (df['Time']-float(pulse_t)) / 10000.0
    df['Duration'] = df['Duration'] / 10000.0

    trial_names = []
    trial_onsets = []
    trial_durations = []

    for i, code in enumerate(con_codes):
        to_write = pd.DataFrame()
        if len(code) > 1:
            if type(code[0]) == int:
                idx = df['Code'].isin(range(code[0], code[1]+1))
        elif len(code) == 1 and type(code[0]) == str:
            idx = [code[0] in x if type(x) == str else False for x in df['Code']]
            idx = np.array(idx)
        else:
            idx = df['Code'] == code

        # Generate dataframe with time, duration, and weight given idx
        to_write['Time'] = df['Time'][idx]
        to_write['Duration'] = df['Duration'][idx]
        to_write['Duration'] = [np.round(x, decimals=2) for x in to_write['Duration']]
        to_write['Weight'] = np.ones((np.sum(idx), 1))
        to_write['Name'] = [con_names[i] + '_%i' % (j+1) for j in range(idx.sum())]

        if con_design[i] == 'univar':
            trial_names.append(to_write['Name'].tolist())
            trial_onsets.append(to_write['Time'].tolist())
            trial_durations.append(to_write['Duration'].tolist())
        elif con_design[i] == 'multivar':
            _ = [trial_names.append([x]) for x in to_write['Name'].tolist()]
            _ = [trial_onsets.append([x]) for x in to_write['Time'].tolist()]
            _ = [trial_durations.append([x]) for x in to_write['Duration'].tolist()]

        if write_bfsl:

            if con_design[i] == 'univar':
                to_write.drop('Name', axis=1, inplace=True)
                name = os.path.join(base_dir, con_names[i] + '.bfsl')
                to_write.to_csv(name, sep='\t', index=False, header=False)

            elif con_design[i] == 'multivar':

                for row in to_write.iterrows():
                    ev_name = row[1]['Name'] + '.bfsl'
                    name = os.path.join(base_dir, ev_name)
                    df_tmp = pd.DataFrame({'Time': row[1]['Time'],
                                           'Duration': row[1]['Duration'],
                                           'Weight': row[1]['Weight']}, index=[0])
                    df_tmp.to_csv(name, index=False, sep='\t', header=False)

    subject_info = Bunch(conditions=con_names,
                         onsets=trial_onsets,
                         durations=trial_durations,
                         amplitudes=None,
                         regressor_names=con_names,
                         regressors=None)

    return subject_info


# Only for testing
if __name__ == '__main__':

    testfile = '/home/lukas/Nipype_testset/working_directory/sub002/func_hww/sub002_hww.log'
    con_names = ['Action', 'Interoception', 'Situation', 'Cue']
    con_codes = [[100, 199], [200, 299], [300, 399], ['Cue']]
    con_design = ['univar', 'univar', 'univar', 'univar']

    parse_presentation_logfile(testfile, con_names, con_codes, con_design,
                               pulsecode=30, write_bfsl=True)
