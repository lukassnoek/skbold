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
    """
    Logfile crawler for Presentation (Neurobs) files; cleans logfile,
    calculates event onsets and durations, and (optionally) writes out
    .bfsl files per condition.

    Parameters
    ----------
    in_file : str or list
        Absolute path to logfile (can be a list of paths).
    con_names : list
        List with names for each condition
    con_codes : list
        List with codes for conditions. Can be a single integer or string (in
        the latter case, it may be a substring) or a list with possible values.
    con_design : list or str
        Which 'design' to assume for events (if 'multivar', all events -
        regardless of condition - are treated as a separate
        condition/regressor; if 'univar', all events from a single condition
        are treated as a single condition). Default: 'univar' for all
        conditions.
    con_duration : list
        If the duration cannot be parsed from the logfile, you can specify them
        here manually (per condition).
    pulsecode : int
        Code with which the first (or any) pulse is logged.
    write_bfsl : bool
        Whether to write out a .bfsl file per condition.
    verbose : bool
        Print out intermediary output.

    Attributes
    ----------
    df : Dataframe
        Dataframe with cleaned and parsed logfile.
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

        design_params = ['univar', 'multivar', None]
        msg = 'Unknown design-parameter; please choose from: %r' % design_params
        if isinstance(con_design, str):
            if con_design not in design_params:
                raise ValueError(msg)

        elif isinstance(con_design, list):

            if not all(d in design_params for d in con_design):
                raise ValueError(msg)

        else:
            msg = 'Unknown type for con_design; please specify list or str.'
            raise ValueError(msg)

        if con_design is None:
            con_design = ['univar'] * len(con_names)

        self.con_design = con_design
        self.pulsecode = pulsecode
        self.write_bfsl = write_bfsl
        self.verbose = verbose
        self.df = None
        self.to_write = None
        self.base_dir = None

    def _parse(self, f):

        if self.verbose:
            print('Processing %s' % f)

        # Remove existing .bfsl files
        self.base_dir = op.dirname(f)
        _ = [os.remove(x) for x in glob(op.join(self.base_dir, '*.bfsl'))]

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

            if len(code) > 1:
                # Code is list of possibilities
                if all(isinstance(c, int) for c in code):
                    idx = df['Code'].isin(code)

                elif all(isinstance(c, str) for c in code):
                    idx = [any(c in x for c in code) if isinstance(x, str) else False for x in df['Code']]
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
            to_write['Name'] = [self.con_names[i] + '_%i' % (j + 1) for j in range(idx.sum())]

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

        subject_info = Bunch(conditions=self.con_names,
                             onsets=trial_onsets,
                             durations=trial_durations,
                             amplitudes=None,
                             regressor_names=self.con_names,
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
        """
        Parses logfile, writes bfsl (optional), and return subject-info.

        Returns
        -------
        subject_info_list : Nilearn bunch object
            Bunch object to be used in Nipype pipelines.
        """
        subject_info_list = [self._parse(f) for f in self.in_file]

        if len(subject_info_list) == 1:
            return subject_info_list[0]
        else:
            return subject_info_list


def parse_presentation_logfile(in_file, con_names, con_codes, con_design=None,
                               con_duration=None, pulsecode=30,
                               write_bfsl=False, verbose=True):
    """
    Function-interface for PresentationLogfileCrawler. Can be used to create
    a Nipype node.

    Parameters
    ----------
    in_file : str or list
        Absolute path to logfile (can be a list of paths).
    con_names : list
        List with names for each condition
    con_codes : list
        List with codes for conditions. Can be a single integer or string (in
        the latter case, it may be a substring) or a list with possible values.
    con_design : list or str
        Which 'design' to assume for events (if 'multivar', all events -
        regardless of condition - are treated as a separate
        condition/regressor; if 'univar', all events from a single condition
        are treated as a single condition). Default: 'univar' for all
        conditions.
    con_duration : list
        If the duration cannot be parsed from the logfile, you can specify them
        here manually (per condition).
    pulsecode : int
        Code with which the first (or any) pulse is logged.
    write_bfsl : bool
        Whether to write out a .bfsl file per condition.
    verbose : bool
        Print out intermediary output.
    """

    from skbold.exp_model import PresentationLogfileCrawler

    plc = PresentationLogfileCrawler(in_file=in_file, con_names=con_names,
                                     con_codes=con_codes, con_design=con_design,
                                     con_duration=con_duration,
                                     pulsecode=pulsecode, write_bfsl=True,
                                     verbose=False)

    subject_info_files = plc.parse()

    return subject_info_files
