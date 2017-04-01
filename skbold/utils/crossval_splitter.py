from __future__ import division, print_function, absolute_import
from builtins import range
import pandas as pd
import numpy as np
import scipy.stats as stat
import os.path as op

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Matplotlib not installed; cannot plot!')


class CrossvalSplitter(object):

    def __init__(self, data, train_size, vars, cb_between_splits=False,
                 binarize=None, include=None, exclude=None,
                 interactions=True, sep='\t', index_col=0, ignore=None,
                 iterations=1000):

        if isinstance(data, (str, unicode)):
            data = pd.read_csv(data, sep=sep, index_col=index_col)

        data['cv_group'] = np.nan
        if include is not None:
            data = data.loc[include]

        for var in vars.keys():
            # ignore values, such as 9999
            data.loc[data[var] == ignore, var] = np.nan

        if exclude is not None:
            for key, value in exclude.items():
                data = data[data[key] != value]

        self.data = data

        if 0 < train_size < 1:  # percentage
            train_size = np.round(data.shape[0] * train_size)

        test_size = data.shape[0] - train_size
        self.train_size = train_size
        self.test_size = test_size
        self.cb_between_splits = cb_between_splits
        self.vars = vars
        self.interactions = interactions
        self.exclude = exclude
        self.ignore = ignore
        self.iterations = iterations
        self.best_all_samples = None
        self.best_train_set = None
        self.best_test_set = None
        self.best_min_p_val = 0

    def split(self, verbose=False):

        full_size = self.train_size + self.test_size

        for i in range(self.iterations):

            p_this_iter = []

            data = self.data
            all_idx = data.index
            # take two random samples:
            full_sample = np.random.choice(all_idx, size=full_size,
                                           replace=False)

            train_idx = full_sample[:self.train_size]
            end = (self.train_size + self.test_size)
            test_idx = full_sample[self.train_size:end]
            data.loc[train_idx, 'cv_group'] = 'train'
            data.loc[test_idx, 'cv_group'] = 'test'
            data = data.loc[full_sample]  # only take the sampled data

            # make sure everything is goin' all right
            assert(len(train_idx) == self.train_size)
            assert(len(test_idx) == self.test_size)
            assert(sum(np.in1d(train_idx, test_idx) == 0))

            ps = self._counterbalance(data.loc[train_idx])
            p_this_iter.extend(ps)
            if self.cb_between_splits:
                ps = self._counterbalance(data.loc[test_idx])
            p_this_iter.extend(ps)

            if min(p_this_iter) > self.best_min_p_val:
                self.best_min_p_val = min(p_this_iter)
                self.best_all_samples = full_sample
                self.best_train_set = train_idx
                self.best_test_set = test_idx

            if verbose:
                print('Iteration %d, best min p-value found: %.3f...' %
                      (i, self.best_min_p_val))

        self.data = self.data.loc[self.best_all_samples]
        self.data.loc[self.best_train_set, 'cv_group'] = 'train'
        self.data.loc[self.best_test_set, 'cv_group'] = 'test'

        return self.best_train_set, self.best_test_set

    def _counterbalance(self, data):

        p_this_set = []

        for var, values in self.vars.items():

            categorical = False is isinstance(values, (str, unicode))
            if categorical:
                chisq, p = self._test_categorical(data, var, values)
            p_this_set.append(p)

            if self.interactions:
                ps = self._test_categorical_interaction(data)
                p_this_set.extend(ps)

        return p_this_set

    def _test_continuous(self, s1, s2):
        t, p = stat.ttest_ind(s1, s2, nan_policy='omit')
        return t, p

    def _test_categorical_interaction(self, data):

        p_ints = []
        for i, (var, values) in enumerate(self.vars.items()):

            if i == 0:
                cvar, cvalues = var, values
            else:
                s1 = data[data[var].isin(values)][var]
                s2 = data[data[cvar].isin(cvalues)][cvar]
                sint = s1 * s2
                count = sint.value_counts()
                chisq, p = stat.chisquare(count.tolist())
                p_ints.append(p)
        return p_ints

    def _test_categorical(self, data, var, values):
        count = data[data[var].isin(values)][var].value_counts()
        chisq, p = stat.chisquare(count.tolist())
        return chisq, p

    def save(self, out_dir, save_plots=True):
        if self.best_min_p_val == 0:
            IOError('split not yet run, nothing to save!')

        self.data = self.data.sort_index()
        self.data.to_csv(op.join(out_dir, 'split.tsv'), sep='\t')

        if save_plots:

            self.plot_results(out_dir)

    def plot_results(self, out_dir):

        train_idx = self.best_train_set
        test_idx = self.best_test_set
        data = self.data

        for ii, (var, values) in enumerate(self.vars.items()):

            plt.figure(ii)

            if isinstance(values, list):
                fig, (ax1, ax2) = plt.subplots(1, 2)
                labels = values
                count = data.groupby([var, 'cv_group']).size()
                count = count.unstack(level=0)[labels]
                train_vals = count.loc['train'].values
                test_vals = count.loc['test'].values

                ax1.pie(train_vals, labels=labels, autopct='%1.1f%%',
                        shadow=True,
                        startangle=90)
                ax1.set_title('%s train group' % var)
                ax2.pie(test_vals, labels=labels, autopct='%1.1f%%',
                        shadow=True,
                        startangle=90)
                ax2.set_title('%s test group' % var)
                fn = op.join(out_dir, var + '.png')
                fig.savefig(fn)
            else:
                train_vals = data.loc[train_idx, var].values
                test_vals = data.loc[test_idx, var].values
                train_vals = train_vals[~np.isnan(train_vals)]
                test_vals = test_vals[~np.isnan(test_vals)]
                plt.hist(train_vals, alpha=0.5, label='Train')
                plt.hist(test_vals, alpha=0.5, label='Test')
                plt.legend(loc='upper right')
                plt.title(var)
                fn = op.join(out_dir, var + '.png')
                plt.savefig(fn)
