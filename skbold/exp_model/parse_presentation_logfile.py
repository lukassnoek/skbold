# Parses a Presentation (neurobs.com) logfile.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function, absolute_import
from builtins import range
import os.path as op
import pandas as pd
import numpy as np


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

    def __init__(self, in_file, con_names, con_codes,
                 con_duration=None, pulsecode=30, write_tsv=True,
                 verbose=True, write_code=False):

        if isinstance(in_file, str):
            in_file = [in_file]

        self.in_file = in_file
        self.con_names = con_names
        self.con_codes = con_codes
        self.write_tsv = write_tsv
        self.write_code = write_code

        if con_duration is not None:

            if isinstance(con_duration, (int, float)):
                con_duration = [con_duration]

            if len(con_duration) < len(con_names):
                con_duration *= len(con_names)

        self.con_duration = con_duration
        self.pulsecode = pulsecode
        self.verbose = verbose
        self.df = None
        self.to_write = None
        self.base_dir = None

    def _parse(self, f):

        if self.verbose:
            print('Processing %s' % f)

        self.base_dir = op.dirname(f)

        if self.df is not None:
            df = self.df
        else:
            df = pd.read_table(f, sep='\t', skiprows=3,
                               skip_blank_lines=True)

        # Clean up unnecessary columns
        to_drop = ['Uncertainty', 'Subject', 'Trial', 'Uncertainty.1',
                   'ReqTime', 'ReqDur', 'Stim Type', 'Pair Index']
        _ = [df.drop(col, axis=1, inplace=True)
             for col in to_drop if col in df.columns]

        # Ugly hack to find pulsecode, because some numeric codes are
        # written as str
        df['Code'] = df['Code'].astype(str)
        df['Code'] = [np.float(x) if x.isdigit() else x for x in df['Code']]
        pulse_idx = np.where(df['Code'] == self.pulsecode)[0]

        if len(pulse_idx) > 1:  # take first pulse if mult pulses are logged
            pulse_idx = int(pulse_idx[0])

        # pulse_t = absolute time of first pulse
        pulse_t = df['Time'][df['Code'] == self.pulsecode].iloc[0]
        df['Time'] = (df['Time'] - float(pulse_t)) / 10000.0
        df['Duration'] /= 10000.0

        to_write_list = []
        # Loop over condition-codes to find indices/times/durations
        for i, code in enumerate(self.con_codes):

            to_write = pd.DataFrame()

            if type(code) == str:
                code = [code]

            if len(code) > 1:
                # Code is list of possibilities
                if all(isinstance(c, (int, np.int64)) for c in code):
                    idx = df['Code'].isin(code)

                elif all(isinstance(c, str) for c in code):
                    idx = [any(c in x for c in code)
                           if isinstance(x, str) else False
                           for x in df['Code']]
                    idx = np.array(idx)

            elif len(code) == 1 and type(code[0]) == str:
                # Code is single string
                idx = [code[0] in x if type(x) == str
                       else False for x in df['Code']]
                idx = np.array(idx)
            else:
                idx = df['Code'] == code

            if idx.sum() == 0:
                raise ValueError('No entries found for code: %r' % code)

            # Generate dataframe with time, duration, and weight given idx
            to_write['onset'] = df['Time'][idx]

            if self.con_duration is None:
                to_write['duration'] = df['Duration'][idx]
                n_nan = np.sum(np.isnan(to_write['duration']))
                if n_nan > 1:
                    msg = ('In total, %i NaNs found for Duration. '
                           'Specify duration manually.' % n_nan)
                    raise ValueError(msg)
                to_write['duration'] = [np.round(x, decimals=2)
                                        for x in to_write['duration']]
            else:
                to_write['duration'] = [self.con_duration[i]] * idx.sum()

            to_write['trial_type'] = [self.con_names[i] for j in range(idx.sum())]

            if self.write_code:
                to_write['code'] = df['Code'][idx]

            to_write_list.append(to_write)

        events_df = pd.concat(to_write_list).sort_values(by='onset')

        if self.write_tsv:
            outname = op.join(self.base_dir, op.basename(f).split('.')[0] + '.tsv')
            events_df.to_csv(outname, sep='\t', index=False)

        return events_df

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


def parse_presentation_logfile(in_file, con_names, con_codes, con_duration=None,
                               write_tsv=True, write_code=False, pulsecode=30):
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
    """

    from skbold.exp_model import PresentationLogfileCrawler

    plc = PresentationLogfileCrawler(in_file=in_file, con_names=con_names,
                                     con_codes=con_codes,
                                     con_duration=con_duration,
                                     pulsecode=pulsecode, write_tsv=write_tsv,
                                     write_code=write_code,
                                     verbose=False)

    subject_info_files = plc.parse()

    return subject_info_files
