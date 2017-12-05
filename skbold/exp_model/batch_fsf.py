from __future__ import division, print_function, absolute_import
from builtins import range
from io import open
import os
import numpy as np
import os.path as op
from glob import glob
import nibabel as nib
import multiprocessing
import pandas as pd


class FsfCrawler(object):
    """
    Given an fsf-template, this crawler creates subject-specific fsf-FEAT files
    assuming that appropriate .bfsl files exist.

    Parameters
    ----------
    data_dir : str
        Absolute path to directory with BIDS-formatted data.
    run_idf : str
        Identifier for run to apply template fsf to.
    template : str
        Absolute path to template fsf-file. Default is 'mvpa', which models
        each event as a separate regressor (and contrast against baseline).
    preprocess : bool
        Whether to apply preprocessing (as specified in template) or whether to
        only run statistics/GLM.
    register : bool
        Whether to calculate registration (func -> highres, highres -> standard)
    mvpa_type : str
        Whether to estimate patterns per trial (mvpa_type='trial_wise') or
        to estimate patterns per condition (or per run, mvpa_type='run_wise')
    output_dir : str
        Path to desired output dir of first-levels.
    subject_idf : str
        Identifier for subject-directories.
    event_file_ext : str
        Extension for event-file; if 'bfsl/txt' (default, for legacy reasons),
        then assumes single event-file per predictor. If 'tsv' (cf. BIDS),
        then assumes a single tsv-file with all predictors.
    sort_by_onset : bool
        Whether to sort predictors by onset (first trial = first predictor),
        or, when False, sort by condition (all trials condition A, all trials
        condition B, etc.).
    n_cores : int
        How many CPU cores should be used for the batch-analysis.
    feat_options : key-word arguments
        Which preprocessing options to set (only relevant if template='mvpa' or
        if you want to deviate from template). Examples:
            mc='1' (apply motion correction),
            st='1' (apply regular-up slice-time correction),
            bet_yn='1' (do brain extraction of func-file),
            smooth='5.0' (smooth with 5 mm FWHM),
            temphp_yn='1' (do HP-filtering),
            paradigm_hp='100' (set HP-filter to 100 seconds),
            prewhiten_yn='1' (do prewhitening),
            motionevs='1' (add motion-params as nuisance regressors)
    """

    def __init__(self, data_dir, run_idf=None, template='mvpa', preprocess=True,
                 register=True, mvpa_type='trial_wise', output_dir=None,
                 subject_idf='sub', event_file_ext='txt', sort_by_onset=False,
                 prewhiten=True, n_cores=1, **feat_options):

        self.template = template
        self.data_dir = data_dir
        self.mvpa_type = mvpa_type
        self.preprocess = preprocess
        self.register = register
        if output_dir is None:
            output_dir = op.join(op.dirname(data_dir), 'firstlevel')

        if not op.isdir(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.run_idf = '' if run_idf is None else run_idf
        self.subject_idf = subject_idf
        self.event_file_ext = event_file_ext
        self.feat_options = feat_options

        if n_cores < 0:
            n_cores = multiprocessing.cpu_count() - n_cores

        self.n_cores = n_cores
        self.clean_fsf = None
        self.out_fsf = []
        self.sub_dirs = None

        if mvpa_type != 'trial_wise':
            sort_by_onset = False

        self.sort_by_onset = sort_by_onset

    def crawl(self):
        """ Crawls subject-directories and spits out subject-specific fsf. """

        self._read_fsf()
        search_cmd = op.join(self.data_dir, '*%s*' % self.subject_idf)
        self.sub_dirs = sorted(glob(search_cmd))

        if not self.sub_dirs:
            msg = "Could not find any subdirs with command: %s" % search_cmd
            raise ValueError(msg)

        fsf_paths = [self._write_fsf(sub) for sub in self.sub_dirs]

        shell_script = op.join(op.dirname(self.output_dir), 'batch_fsf.sh')
        with open(shell_script, 'w') as fout:

            for i, fsf in enumerate(fsf_paths):

                fout.write(str('feat %s &\n' % fsf))
                if (i+1) % self.n_cores == 0:
                    fout.write('wait\n')

    def _read_fsf(self):
        """ Reads in template-fsf and does some cleaning. """

        if self.template == 'mvpa':
            template = op.join(op.dirname(op.dirname(
                               op.abspath(__file__))), 'data',
                               'Feat_single_trial_template.fsf')

        else:
            template = self.template

        with open(template, 'r') as f:
            template = f.readlines()

        template = [txt.replace('\n', '') for txt in template if txt != '\n']
        template = [txt for txt in template if txt[0] != '#']  # remove commnts

        self.clean_fsf = template

    def _write_fsf(self, sub_dir):
        """ Creates and writes out subject-specific fsf. """
        sub_name = op.basename(sub_dir)
        func_file = glob(op.join(sub_dir, 'func', '*%s*.nii.gz' % self.run_idf))

        if len(func_file) == 0:
            msg = "Found no func-file for sub %s" % sub_dir
            raise IOError(msg)
        elif len(func_file) > 1:
            msg = "Found more than one func-file for sub %s: %r" % (sub_dir,
                                                                    func_file)
            raise IOError(msg)
        else:
            func_file = func_file[0]

        highres_file = glob(op.join(sub_dir, 'anat', '*_brain.nii.gz'))
        if len(highres_file) == 0:
            msg = "Found no highres-file for sub %s" % sub_dir
            raise IOError(msg)
        elif len(highres_file) > 1:
            msg = ("Found more than one highres-file for sub "
                   "%s: %r" % (sub_dir, highres_file))
            raise IOError(msg)
        else:
            highres_file = highres_file[0]

        feat_dir = sub_name + '_%s' % self.run_idf if self.run_idf else sub_dir
        out_dir = op.join(self.output_dir, feat_dir)
        hdr = nib.load(func_file).header

        if self.event_file_ext == 'tsv':
            probable_df = glob(op.join(sub_dir, 'func', '*%s*.tsv' % self.run_idf))
            if len(probable_df) != 1:
                raise ValueError("Found %i event-files for subject %s" %
                                 (len(probable_df), sub_dir))
            else:
                event_file = probable_df[0]
            events = self._tsv2event_files(event_file)
        else:
            search_str = '*%s*.%s' % (self.run_idf, self.event_file_ext)
            events = sorted(glob(op.join(sub_dir, search_str)))

        arg_dict = {'analysis': '7' if self.preprocess else '2',
                    'filtering_yn': '1' if self.preprocess else '0',
                    'reghighres_yn': '1' if self.register else '0',
                    'regstandard_yn': '1' if self.register else '0',
                    'tr': hdr['pixdim'][4],
                    'npts': hdr['dim'][4],
                    'custom': events,
                    'feat_files': "\"%s\"" % func_file,
                    'outputdir': "\"%s\"" % out_dir,
                    'highres_files': "\"%s\"" % highres_file,
                    'totalVoxels': np.prod(hdr['dim'][1:5])}

        if self.feat_options:
            arg_dict.update(self.feat_options)

        if self.template == 'mvpa':
            arg_dict['ncon_orig'] = str(len(arg_dict['custom']))
            arg_dict['ncon_real'] = str(len(arg_dict['custom']))
            arg_dict['nftests_orig'] = '1'
            arg_dict['nftests_real'] = '1'
            arg_dict['evs_orig'] = str(len(arg_dict['custom']))
            arg_dict['evs_real'] = str(len(arg_dict['custom']))
            arg_dict.pop('custom')

        fsf_out = []
        # Loop over lines in cleaned template-fsf
        for line in self.clean_fsf:

            if any(key in line for key in arg_dict.keys()):
                parts = [txt for txt in line.split(' ') if txt]
                this_key = [key for key in arg_dict.keys() if key in line][0]
                values = arg_dict[this_key]

                if this_key != 'custom':
                    parts[-1] = values
                elif this_key == 'custom':
                    ev = line.split(os.sep)[-1].replace("\"", '')
                    search_str = op.join(sub_dir, 'func',
                                         '*%s*.%s' % (self.run_idf,
                                                      self.event_file_ext))
                    event_files = sorted(glob(search_str))
                    to_set = [e for e in event_files if ev in e]

                    if len(to_set) == 1:
                        parts[-1] = "\"%s\"" % to_set[0]
                    else:
                        raise ValueError("Ambiguous ev (%s) for event-files "
                                         "(%r)" % (ev, event_files))

                parts = [str(p) for p in parts]
                fsf_out.append(" ".join(parts))
            else:
                fsf_out.append(line)

        if self.template == 'mvpa':
            fsf_out = self._append_single_trial_info(events, fsf_out)

        to_write = op.join(sub_dir, 'design.fsf')
        with open(to_write, 'w') as fsfout:
            print("Writing fsf to %s" % sub_dir)
            fsfout.write(str("\n".join(fsf_out)))

        return to_write

    def _tsv2event_files(self, tsv_file):

        ev_files_dir = op.join(op.dirname(tsv_file), 'ev_files')
        if not op.isdir(ev_files_dir):
            os.makedirs(ev_files_dir)

        df = pd.read_csv(tsv_file, sep='\t')
        ev_files = []
        if self.mvpa_type == 'trial_wise':

            iters = {con: 0 for con in np.unique(df.trial_type)}
            for i in range(len(df)):
                info_event = df.iloc[i][['onset', 'duration', 'weight']]
                name_event = df.iloc[i].trial_type
                iters[name_event] += 1
                fn = '%s_%i.txt' % (name_event, iters[name_event])
                fn = op.join(ev_files_dir, fn)
                np.savetxt(fn, np.array(info_event), delimiter=' ', fmt='%.3f')
                ev_files.append(fn)

        else:  # assume run-wise estimation
            evs = np.unique(df.trial_type)
            for ev in evs:
                info_event = df[df.trial_type == ev]
                info_event = info_event[['onset', 'duration', 'weight']]
                fn = op.join(ev_files_dir, '%s.txt' % ev)
                np.savetxt(fn, np.array(info_event))
                ev_files.append(fn)
        return ev_files

    def _append_single_trial_info(self, events, fsf_out):
        """ Does some specific 'single-trial' (mvpa) stuff. """

        if self.sort_by_onset and self.mvpa_type == 'trial_wise':
            events = sorted(events, key=lambda x: np.loadtxt(x)[0])

        for i, ev in enumerate(events):

            ev_name = op.basename(ev).split('.')[0]

            fsf_out.append('set fmri(evtitle%i) \"%s\"' % ((i + 1), ev_name))
            fsf_out.append('set fmri(shape%i) 3' % (i + 1))
            fsf_out.append('set fmri(convolve%i) 3' % (i + 1))
            fsf_out.append('set fmri(convolve_phase%i) 0' % (i + 1))
            fsf_out.append('set fmri(tempfilt_yn%i) 0' % (i + 1))
            fsf_out.append('set fmri(deriv_yn%i) 0' % ((i + 1)))
            fsf_out.append('set fmri(custom%i) %s' % ((i + 1), ev))

            for x in range(len(events) + 1):
                fsf_out.append('set fmri(ortho%i.%i) 0' % ((i + 1), x))

            fsf_out.append('set fmri(conpic_real.%i) 0' % (i + 1))
            fsf_out.append('set fmri(conname_real.%i) \"%s\"' % ((i + 1),
                                                                 ev_name))
            fsf_out.append('set fmri(conpic_orig.%i) 0' % (i + 1))
            fsf_out.append('set fmri(conname_orig.%i) \"%s\"' % ((i + 1),
                                                                 ev_name))

            for x in range(len(events)):
                to_set = "1" if (x + 1) == (i + 1) else "0"

                fsf_out.append('set fmri(con_real%i.%i) %s' % (
                    (i + 1), (x + 1), to_set))

                fsf_out.append('set fmri(con_orig%i.%i) %s' % (
                    (i + 1), (x + 1), to_set))

            fsf_out.append('set fmri(ftest_real1.%i) 1' % (i + 1))
            fsf_out.append('set fmri(ftest_orig1.%i) 1' % (i + 1))

        for x in range(len(events)):

            for y in range(len(events)):

                if (x + 1) == (y + 1):
                    continue

                fsf_out.append('set fmri(conmask%i_%i) 0' % ((x + 1),
                                                             (y + 1)))

        return fsf_out
