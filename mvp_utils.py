import numpy as np
import fnmatch
import glob
import os
import pickle
import pandas as pd
from os.path import join as opj
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
     confusion_matrix


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


class MvpResults(object):
    """ Contains info about model performance across iterations
    """
    def __init__(self, mvp, iterations, n_test, method='iteration-based',
                 verbose=0):

        self.method = method
        self.n_iter = iterations
        self.verbose = verbose
        self.sub_name = mvp.subject_name
        self.run_name = mvp.run_name
        self.n_features = np.zeros((self.n_iter, 1))
        self.y_true = mvp.y
        self.n_tested = n_test * mvp.n_class

        self.precision = None
        self.accuracy = None
        self.recall = None
        self.conf_mat = np.zeros((mvp.n_class, mvp.n_class))

        if method == 'iteration-based':
            self.precision = np.zeros((self.n_iter, 1))
            self.accuracy = np.zeros((self.n_iter, 1))
            self.recall = np.zeros((self.n_iter, 1))
        elif method == 'trial-based':
            self.trials_mat = np.zeros((len(mvp.y), mvp.n_class))

    def update_results(self, iter, n_feat, test_idx, y_pred):
        """ Updates results after a cross-validation iteration """
        self.n_features[iter, 0] = n_feat

        if self.method == 'iteration-based':
            y_true = self.y_true[test_idx]
            self.conf_mat += confusion_matrix(y_true, y_pred)
            self.precision[iter] = precision_score(y_true, y_pred, average='macro')
            self.recall[iter] = recall_score(y_true, y_pred, average='macro')
            self.accuracy[iter] = accuracy_score(y_true, y_pred)

            if self.verbose:
                print('Accuracy: %f' % self.accuracy[iter])

        elif self.method == 'trial-based':
            self.trials_mat[test_idx, y_pred.astype(int)] += 1
        return self

    def compute_score(self):

        self.n_features = self.n_features.mean()

        if self.method == 'trial-based':
            filter_trials = np.sum(self.trials_mat, 1) == 0
            trials_max = np.argmax(self.trials_mat, 1) + 1
            trials_max[filter_trials] = 0
            y_pred = trials_max[trials_max > 0]-1
            y_true = self.y_true[trials_max > 0]

            self.conf_mat = confusion_matrix(y_true, y_pred)
            self.precision = precision_score(y_true, y_pred, average='macro')
            self.recall = recall_score(y_true, y_pred, average='macro')
            self.accuracy = accuracy_score(y_true, y_pred)

        elif self.method == 'iteration-based':

            self.accuracy = self.accuracy.mean()
            self.precision = self.precision.mean()
            self.recall = self.recall.mean()

        print('Accuracy over iterations: %f' % self.accuracy)
        return self

    def write_results(self, directory):
        filename = os.path.join(directory, '%s_%s_results.pickle' % \
                                (self.sub_name, self.run_name))
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

        # write meta-data (analysis parameters and stuff)

class MvpAverageResults(object):
    """
    Object that is able to load individual result-files, process/analyze
    them and write their average/processed results as a pandas dataframe
    """

    def __init__(self, directory, identifier, write_df=True, cleanup=True):
        self.directory = directory
        self.identifier = identifier
        self.write_df = write_df
        self.cleanup = cleanup
        self.df_ = pd.DataFrame()

    def load(self):

        files = glob.glob(os.path.join(self.directory, '*%s*.pickle' % \
                                       self.identifier))

        for i, f in enumerate(files):
            results = pickle.load(open(f, 'rb'))
            tmp = {'Sub_name': results.sub_name,
                   'Accuracy': results.accuracy,
                   'Precision': results.precision,
                   'Recall': results.recall,
                   'n_feat': results.n_features}
            self.df_ = self.df_.append(pd.DataFrame(tmp, index=[i]))

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
        return self

    def write(self):
        filename = os.path.join(self.directory, 'average_results_%s.csv' % \
                                self.identifier)
        self.df_.to_csv(filename, sep='\t', header=True, index=False)



