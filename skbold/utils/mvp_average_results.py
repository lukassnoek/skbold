# Class to average results across subjects from a classification analysis.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import print_function, division, absolute_import
import cPickle
import json
import pandas as pd
import numpy as np
import os.path as op
import os
import glob
from skbold.transformers import *
import nibabel as nib


class MvpAverageResults(object):
    """ Class that averages individual subject classification performance.

    Is able to load individual result-files, process/analyze them and write
    their average/processed results as a pandas dataframe.

    """

    def __init__(self, directory, resultsdir='analysis_results', params=None,
                 threshold=None, cleanup=True):
        """ Initializes MvpAverageResults object.

        Parameters
        ----------
        directory : str
            Absolute path to where individual classification files are located.
        threshold : int
            Threshold for summation-accuracy metric of voxel scores
            (i.e. score = sum(voxel_scores > threshold)).
        cleanup = bool
            Whether to clean up subject-specific nifti-files
        params : dict
            Dictionary with analysis parameters
        """

        self.directory = directory
        self.threshold = threshold
        self.params = params
        self.resultsdir = resultsdir
        self.cleanup = cleanup
        self.threshold = threshold
        self.cleanup = cleanup
        self.df_ = pd.DataFrame()
        self.feature_scoring = None

    def average(self):
        """ Loads and computes average performance metrics. """

        results_dir = op.join(self.directory, self.resultsdir)
        files = glob.glob(op.join(results_dir, '*.pickle'))

        if not files:
            raise ValueError('Couldnt find files to be averaged!')

        for i, f in enumerate(files):
            results = cPickle.load(open(f, 'rb'))
            tmp = {'Sub_name': results.sub_name,
                   'Accuracy': results.accuracy,
                   'Precision': results.precision,
                   'Recall': results.recall,
                   'n_feat': results.n_features}
            self.df_ = self.df_.append(pd.DataFrame(tmp, index=[i]))

        if not self.threshold:
            n_class = results.n_class
            self.threshold = (1/n_class) + .2 * (1/n_class)

        tmp = {'Sub_name': 'Average',
               'Accuracy': self.df_['Accuracy'].mean(),
               'Precision': self.df_['Precision'].mean(),
               'Recall': self.df_['Recall'].mean(),
               'n_feat': self.df_['n_feat'].mean()}

        self.df_ = self.df_.append(pd.DataFrame(tmp, index=[i+1]))
        cols = list(self.df_)
        cols.insert(0, cols.pop(cols.index('Sub_name')))
        self.df_ = self.df_.ix[:, cols]

        if self.cleanup:
            _ = [os.remove(f) for f in files]

        print(self.df_)

        filename = op.join(results_dir, 'average_results.csv')
        self.df_.to_csv(filename, sep='\t', header=True, index=False)

        if self.params:
            file2open = op.join(self.directory, 'analysis_parameters.json')
            with open(file2open, 'w') as f:
                json.dump(self.params, f)

        feature_files = glob.glob(op.join(results_dir, 'vox_results_mni',
                                          '*.nii*'))
        # Some really ugly code ...
        if len(feature_files) > 0:

            for i, feature_file in enumerate(feature_files):
                data = nib.load(feature_file).get_data()
                if i == 0:
                    if data.ndim > 3:
                        a, b, c, d = data.shape
                        s = np.zeros((len(feature_files), a, b, c, d))
                        s[i, :, :, :, :] = data
                    else:
                        a, b, c = data.shape
                        s = np.zeros((len(feature_files), a, b, c))
                        s[i, :, :, :] = data
                else:
                    try:
                        s[i, :, :, :] = data
                    except:
                        s[i, :, :, :, :] = data

            metric = feature_file.split('_')[-1].split('.')[0]

            if metric == 'accuracy':
                s = (s > self.threshold).astype(int).sum(axis=0)
            if 'coef' in metric:
                nsub = s.shape[0]
                s = s.mean(axis=0) / (s.std(axis=0) / np.sqrt(nsub))
            else:
                s = s.mean(axis=0)

            if self.cleanup:
                cmd = 'rm %s/*%s*.nii' % results_dir
                _ = os.system(cmd)

            fn = op.join(results_dir, 'AverageScores')
            img = nib.Nifti1Image(s, np.eye(4))
            nib.save(img, fn)

        # Remove annoying .mat files (from epi-mni conversion)
        to_remove = glob.glob(op.join(os.getcwd(), '*.mat'))
        _ = [os.remove(f) for f in to_remove]
