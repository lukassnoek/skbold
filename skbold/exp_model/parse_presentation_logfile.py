# Parses a Presentation (neurobs.com) logfile.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function

import os
import os.path as op
import pandas as pd
import numpy as np
from glob import glob
from nipype.interfaces.base import Bunch


class PresentationLogfileCrawler(object):
    """ Parses a Presentation logfile.

    Logfile crawler for Presentation (Neurobs) files.
    """

    def __init__(self, in_file, con_names, con_codes, con_design=None,
                 con_duration=None, pulsecode=30, write_bfsl=False, verbose=True):

        if isinstance(in_file, str):
            in_file = [in_file]

        self.in_file = in_file
        self.con_names = con_names
        self.con_codes = con_codes

        if con_duration is not None:

            if isinstance(con_duration, (int, float)):
                con_duration = [con_duration]

            if len(con_duration) < len(con_names):
                con_duration = con_duration * len(con_names)

        self.con_duration = con_duration

        if con_design is None:
            con_design = ['univar'] * len(con_names)

        self.con_design = con_design
        self.pulsecode = pulsecode
        self.write_bfsl = write_bfsl
        self.verbose = verbose
        self.df = None
        self.to_write = None
        self.base_dir = None

    def clean(self):
        print('This should be implemented for a specific, subclassed crawler!')
        # set self.df to cleaned dataframe
        pass

    def _parse(self, f):

        if self.verbose:
            print('Processing %s' % f)

        # Remove existing .bfsl files
        self.base_dir = op.dirname(f)
        _ = [os.remove(x) for x in glob(op.join(self.base_dir, '*.bfsl'))]

        # If .clean() has not been called (and thus logfile hasn't been loaded,
        # load in the logfile now.
        if self.df is not None:
            df = self.df
        else:
            df = pd.read_table(f, sep='\t', skiprows=3,
                               skip_blank_lines=True)

        # Clean up unnecessary columns
        to_drop = ['Uncertainty', 'Subject', 'Trial', 'Uncertainty.1', 'ReqTime',
                   'ReqDur', 'Stim Type', 'Pair Index']
        _ = [df.drop(col, axis=1, inplace=True) for col in to_drop if col in df.columns]

        # Ugly hack to find pulsecode, because some numeric codes are written as str
        df['Code'] = df['Code'].astype(str)
        df['Code'] = [np.float(x) if x.isdigit() else x for x in df['Code']]
        pulse_idx = np.where(df['Code'] == self.pulsecode)[0]

        if len(pulse_idx) > 1: # take first pulse if multiple pulses are logged
            pulse_idx = int(pulse_idx[0])

        # pulse_t = absolute time of first pulse
        pulse_t = df['Time'][df['Code'] == self.pulsecode].iloc[0]
        df['Time'] = (df['Time'] - float(pulse_t)) / 10000.0
        df['Duration'] = df['Duration'] / 10000.0

        trial_names = []
        trial_onsets = []
        trial_durations = []

        # Loop over condition-codes to find indices/times/durations
        for i, code in enumerate(self.con_codes):

            to_write = pd.DataFrame()

            if type(code) == str:
                code = [code]

#            if len(code) > 1:
#               code = [[c] for c in code if type(c) != list]

            if len(code) > 1:
                # Code is list of possibilities
                if all(isinstance(c, int) for c in code):
                    idx = df['Code'].isin(code)

                elif all(isinstance(c, str) for c in code):
                    idx = [x in code for x in df['Code'] if type(x) == str]
                    idx = np.array(idx)

            elif len(code) == 1 and type(code[0]) == str:
                # Code is single string
                idx = [code[0] in x if type(x) == str else False for x in df['Code']]
                idx = np.array(idx)
            else:
                idx = df['Code'] == code

            if idx.sum() == 0:
                raise ValueError('No entries found for code: %r' % code)

            # Generate dataframe with time, duration, and weight given idx
            to_write['Time'] = df['Time'][idx]

            if self.con_duration is None:
                to_write['Duration'] = df['Duration'][idx]
                n_nan = np.sum(np.isnan(to_write['Duration']))
                if n_nan > 1:
                    msg = 'In total, %i NaNs found for Duration. Specify duration manually.' % n_nan
                    raise ValueError(msg)
                to_write['Duration'] = [np.round(x, decimals=2) for x in to_write['Duration']]
            else:
                to_write['Duration'] = [self.con_duration[i]] * idx.sum()

            to_write['Weight'] = np.ones((np.sum(idx), 1))
            to_write['Name'] = [con_names[i] + '_%i' % (j + 1) for j in range(idx.sum())]

            if self.con_design[i] == 'univar':
                trial_names.append(to_write['Name'].tolist())
                trial_onsets.append(to_write['Time'].tolist())
                trial_durations.append(to_write['Duration'].tolist())
            elif self.con_design[i] == 'multivar':
                _ = [trial_names.append([x]) for x in to_write['Name'].tolist()]
                _ = [trial_onsets.append([x]) for x in to_write['Time'].tolist()]
                _ = [trial_durations.append([x]) for x in to_write['Duration'].tolist()]

            self.to_write = to_write

            if self.write_bfsl:
                self._write_bfsl(i)

        subject_info = Bunch(conditions=con_names,
                             onsets=trial_onsets,
                             durations=trial_durations,
                             amplitudes=None,
                             regressor_names=con_names,
                             regressors=None)

        return subject_info

    def _write_bfsl(self, i):

        to_write = self.to_write

        if self.con_design[i] == 'univar':
            to_write.drop('Name', axis=1, inplace=True)
            name = op.join(self.base_dir, '%s.bfsl' % self.con_names[i])
            to_write = to_write[['Time', 'Duration', 'Weight']]
            to_write.to_csv(name, sep='\t', index=False, header=False)

        elif self.con_design[i] == 'multivar':

            for ii, (index, row) in enumerate(to_write.iterrows()):
                ev_name = '%s.bfsl' % row['Name']
                name = os.path.join(self.base_dir, ev_name)
                df_tmp = pd.DataFrame({'Time': row['Time'],
                                       'Duration': row['Duration'],
                                       'Weight': row['Weight']}, index=[0])
                df_tmp = df_tmp[['Time', 'Duration', 'Weight']]
                df_tmp.to_csv(name, index=False, sep='\t', header=False)

    def parse(self):

        subject_info_list = [self._parse(f) for f in self.in_file]

        if len(subject_info_list) == 1:
            return subject_info_list[0]
        else:
            return subject_info_list


def parse_presentation_logfile(in_file, con_names, con_codes, con_design=None,
                               pulsecode=30, write_bfsl=False, verbose=True):
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

    if type(in_file) == str:
        in_file = [in_file]

    subject_info_files = []

    for f in in_file:

        if verbose:
            print('Processing %s' % f)

        base_dir = os.path.dirname(f)
        _ = [os.remove(x) for x in glob.glob(os.path.join(base_dir, '*.bfsl'))]

        if not con_design:
            con_design = ['univar'] * len(con_names)

        df = pd.read_table(f, sep='\t', skiprows=3, header=0,
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
                    idx = df['Code'].isin(code)
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
                    name = os.path.join(base_dir, '%s_00%i.bfsl' % (con_names[i], (i+1)))
                    to_write = to_write[['Time', 'Duration', 'Weight']]
                    to_write.to_csv(name, sep='\t', index=False, header=False)

                elif con_design[i] == 'multivar':

                    for ii, (index, row) in enumerate(to_write.iterrows()):
                        ev_name = '%s.bfsl' % row['Name']
                        name = os.path.join(base_dir, ev_name)
                        df_tmp = pd.DataFrame({'Time': row['Time'],
                                               'Duration': row['Duration'],
                                               'Weight': row['Weight']}, index=[0])
                        df_tmp = df_tmp[['Time', 'Duration', 'Weight']]
                        df_tmp.to_csv(name, index=False, sep='\t', header=False)

        subject_info = Bunch(conditions=con_names,
                             onsets=trial_onsets,
                             durations=trial_durations,
                             amplitudes=None,
                             regressor_names=con_names,
                             regressors=None)
        subject_info_files.append(subject_info)

    return subject_info_files


if __name__ == '__main__':

    test_file = '/Users/steven/Documents/Syncthing/MscProjects/Decoding/dat/piop_logfile_samples/piopharriri/piopharriri.log'
    con_names = ['control', 'emotion']
    control = [x for x in range(50, 78)]
    emotion = [x for x in range(10, 48)]
#    control = [50, 77]
#    emotion = [10, 47]
    con_codes = [control, emotion]
    print(con_codes)
    con_design = ['univar'] * len(con_codes)
    con_duration = None
    plc = PresentationLogfileCrawler(in_file=test_file, con_names=con_names, con_codes=con_codes,
                                     con_design=con_design, con_duration=con_duration, pulsecode=255, write_bfsl=True)
    plc.parse()

    test_file = '/Users/steven/Documents/Syncthing/MscProjects/Decoding/dat/piop_logfile_samples/piopgstroop/piopgstroop.log'
    ff = np.concatenate([np.arange(101, 113), np.arange(201, 213), np.arange(301, 313), np.arange(401, 413)], axis=0)
    mm = np.concatenate([np.arange(513, 525), np.arange(613, 625), np.arange(713, 725), np.arange(813, 825)], axis=0)
    fm = np.concatenate([np.arange(113, 125), np.arange(213, 225), np.arange(313, 325), np.arange(413, 424)], axis=0)
    mf = np.concatenate([np.arange(501, 513), np.arange(601, 613), np.arange(701, 713), np.arange(801, 813)], axis=0)

    congruent = np.concatenate([ff, mm])
    incongruent = np.concatenate([fm, mf])
    print(congruent)
    con_names = ['congruent', 'incongruent']
    con_codes = [congruent, incongruent]
    con_design = ['univar'] * len(con_codes)
    con_duration = None
    plc = PresentationLogfileCrawler(in_file=test_file, con_names=con_names, con_codes=con_codes,
                                     con_design=con_design, con_duration=con_duration, pulsecode=255, write_bfsl=True)
    plc.parse()