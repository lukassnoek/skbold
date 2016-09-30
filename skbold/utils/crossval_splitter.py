import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import os.path as op

class CrossvalSplitter(object):

    def __init__(self, file_path, train_size, test_size=0, categorical={},
                 continuous=[], binarize=None, include=[], exclude=None, interactions=True,
                 sep='\t', index_col=0, ignore=None, iterations=1000):

        data = pd.read_csv(file_path, sep=sep, index_col=index_col)
        data["cv_group"] = np.nan

        if len(include) > 0:
            data = data.loc[include]

        if len(categorical) > 0:
            for cat in categorical.keys():
                # ignore values, such as 9999
                data.loc[data[cat] == ignore, cat] = np.nan

        if len(continuous) > 0:
            for cont in continuous:
                # ignore values, such as 9999
                data.loc[data[cont] == ignore, cont] = np.nan

        if exclude is not None:
            for key, value in exclude.items():
                data = data[data[key] != value]

        if binarize is not None:

            for var in continuous:
                data, idx_ = binarize_continuous_variable(data, var, binarize,
                                                          save=None)

            self.binar_idx = idx_
        else:
            self.binar_idx = None

        self.data = data

        if train_size < 1 and train_size > 0: #percentage
            train_size = np.round(data.shape[0] * train_size)
            test_size = data.shape[0] - train_size

        self.train_size = train_size
        self.test_size = test_size
        self.categorical = categorical
        self.continuous = continuous
        self.interactions = interactions
        self.exclude = exclude
        self.ignore = ignore
        self.iterations = iterations

        self.best_train_set = None
        self.best_test_set = None
        self.best_min_p_val = 0
        self.fig = None

    def split(self, verbose=False):

        for i in range(self.iterations):
            data = self.data

            all_idx = data.index

            # take two random samples:
            full_sample = np.random.choice(all_idx,
                                           size=self.train_size+self.test_size,
                                           replace=False)
            train_idx = full_sample[:self.train_size]
            end = (self.train_size+self.test_size)
            test_idx = full_sample[self.train_size:end]
            data.loc[train_idx, 'cv_group'] = 'train'
            data.loc[test_idx, 'cv_group'] = 'test'

            data = data.loc[full_sample]  # only take the sampled data

            # make sure everything is goin' all right
            assert(len(train_idx) == self.train_size)
            assert(len(test_idx) == self.test_size)
            assert(sum(np.in1d(train_idx, test_idx) == 0))

            p_this_iter = []

            # first, check if train and test group do not differ
            # on continuous variables:
            for cont in self.continuous:
                (t, p) = stat.ttest_ind(data.loc[train_idx, cont],
                                         data.loc[test_idx, cont],
                                         nan_policy='omit')
                p_this_iter.append(p)

                if verbose:
                    print('T-testing %s:' % cont)
                    print('t: %.4f, p: %.4f' % (t, p))

                if p < 0.05 or np.isnan(p):
                    print('test significant, trying new split...')
                    continue

            for cat, vals in self.categorical.items():
                count = data.groupby([cat, 'cv_group']).size()
                count = count.unstack(level=0)[vals]
                (chisq, p, dof, expected) = stat.chi2_contingency(count)
                p_this_iter.append(p)

                if verbose:
                    print('Chi square test on continquency table of %s:' % cat)
                    print('chi_sq(%.4d): %.4f, p: %.4f' % (dof, chisq, p))

                if p < 0.05 or np.isnan(p):
                    print('test significant, trying new split...')
                    continue

            if verbose:
                print('Iteration %d, best min p-value found: %.3f...' %
                      (i, self.best_min_p_val))

            if min(p_this_iter) > self.best_min_p_val:
                self.best_min_p_val = min(p_this_iter)
                self.best_all_samples = full_sample
                self.best_train_set = train_idx
                self.best_test_set = test_idx

        self.data = self.data.loc[self.best_all_samples]
        self.data.loc[self.best_train_set, 'cv_group'] = 'train'
        self.data.loc[self.best_test_set, 'cv_group'] = 'test'

        return self.best_train_set, self.best_test_set

    def save(self, fid, save_plots=True):
        if self.best_min_p_val == 0:
            IOError('split not yet run, nothing to save!')

        self.data = self.data.sort_index()
        self.data.to_csv(fid, sep='\t')

        if save_plots:

            self.plot_results(show=False)
            fn = op.splitext(fid)[0]
            self.fig.savefig(fn + '.png')

    def plot_results(self, show=True):

        train_idx = self.best_train_set
        test_idx = self.best_test_set
        data = self.data

        for i, cont in enumerate(self.continuous):
            train_vals = data.loc[train_idx, cont].values
            test_vals = data.loc[test_idx, cont].values
            train_vals = train_vals[~np.isnan(train_vals)]
            test_vals = test_vals[~np.isnan(test_vals)]

            plt.figure(i)
            plt.hist(train_vals, alpha=0.5, label='Train')
            plt.hist(test_vals, alpha=0.5, label='Test')
            plt.legend(loc='upper right')
            plt.title(cont)

        for cat in self.categorical:
            i += 1
            count = data.groupby([cat, 'cv_group']).size()
            count = count.unstack(level=0)[self.categorical[cat]]
            print(count)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            labels = self.categorical[cat]

            train_vals = count.loc['train'].values
            test_vals = count.loc['test'].values

            ax1.pie(train_vals, labels=labels, autopct='%1.1f%%', shadow=True,
                    startangle=90)
            ax1.set_title('%s train group' % cat)
            ax2.pie(test_vals, labels=labels, autopct='%1.1f%%', shadow=True,
                    startangle=90)
            ax2.set_title('%s test group' % cat)

        self.fig = fig

        if show:
            plt.show()

def binarize_continuous_variable(data, column_name, binarize, save=None):
    y = data[column_name]

    if binarize['type'] == 'percentile':
        y_rank = [stat.percentileofscore(y, a, 'rank') for a in y]
        y_rank = np.array(y_rank)
        idx = (y_rank < binarize['low']) | (y_rank > binarize['high'])
        low = stat.scoreatpercentile(y, binarize['low'])
        high = stat.scoreatpercentile(y, binarize['high'])
        y = (y_rank[idx] > 50).astype(int)
        data = data[idx]

    elif binarize['type'] == 'zscore':
        y_norm = (y - y.mean()) / y.std()  # just to be sure
        idx = np.abs(y_norm) > binarize['std']
        # self.binarize_params = {'type': binarize['type'],
        #                         'mean': y.mean(),
        #                         'std': y.std(),
        #                         'n_std': binarize['std']}
        y = (y_norm[idx] > 0).astype(int)
        data = data[idx] # only select part of the data!

    elif binarize['type'] == 'constant':
        idx = y > binarize['cutoff']
        y = idx.astype(int)
        # self.binarize_params = {'type': binarize['type'],
        #                         'cutoff': binarize['cutoff']}

    elif binarize['type'] == 'median':  # median-split
        median = np.median(y)
        y = (y > median).astype(int)
        idx = None
        # self.binarize_params = {'type': binarize['type'],
        #                         'median': median}
    data[column_name] = y

    if not save==None:
        data.to_csv(save, sep='\t')

    return data, idx