import pandas as pd
import numpy as np
import scipy.stats as stats
import os.path as op

class CrossvalSplitter(object):

    def __init__(self, file_path, train_size, test_size=0, categorical={}, continuous=[], include=[], exclude=None, interactions=True, sep='\t', index_col=0, ignore=None, iterations=1000):

        data = pd.read_csv(file_path, sep=sep, index_col=index_col)
        data["cv_group"] = np.nan

        if len(include)>0:
            data = data.loc[include]

        if len(categorical)>0:
            for cat in categorical.keys():
                data.loc[data[cat] == ignore] = np.nan  # ignore values, such as 9999

        if len(continuous)>0:
            for cont in continuous:
                data.loc[data[cont] == ignore] = np.nan  # ignore values, such as 9999

        self.data = data
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

    def split(self, verbose=False):

        for i in range(self.iterations):
            data = self.data
            if self.exclude != None:
                for key, value in self.exclude.items():
                    data = data[data[key] != value]

            all_idx = data.index

            # take two random samples:
            full_sample = np.random.choice(all_idx, size=self.train_size+self.test_size, replace=False)
            train_idx = full_sample[:self.train_size]
            test_idx = full_sample[self.train_size:(self.train_size+self.test_size)]
            data.loc[train_idx, 'cv_group'] = 'train'
            data.loc[test_idx, 'cv_group'] = 'test'

            data = data.loc[full_sample] # only take the sampled data

            # make sure everything is goin' all right
            assert(len(train_idx) == self.train_size)
            assert(len(test_idx) == self.test_size)
            assert(sum(np.in1d(train_idx, test_idx)==0))

            p_this_iter = []

            # first, check if train and test group do not differ on continuous variables:
            for cont in self.continuous:
                (t, p) = stats.ttest_ind(data.loc[train_idx, cont], data.loc[test_idx, cont], nan_policy='omit')
                p_this_iter.append(p)

                if verbose:
                    print('T-testing %s:' %cont)
                    print('t: %.4f, p: %.4f' %(t, p))

                if p < 0.05 or np.isnan(p):
                    print('test significant, trying new split...')
                    continue

            for cat, vals in self.categorical.items():
                count = data.groupby([cat, 'cv_group']).size()
                count = count.unstack(level=0)[vals]
                (chisq, p, dof, expected) = stats.chi2_contingency(count)
                p_this_iter.append(p)

                if verbose:
                    print('Chi square test on continquency table of %s:' %cat)
                    print('chi_sq(%.4d): %.4f, p: %.4f' %(dof, chisq, p))

                if p < 0.05 or np.isnan(p):
                    print('test significant, trying new split...')
                    continue
            print('Iteration %d, best min p-value found: %.3f...' %(i, self.best_min_p_val))

            if min(p_this_iter) > self.best_min_p_val:
                self.best_min_p_val = min(p_this_iter)
                self.best_all_samples = full_sample
                self.best_train_set = train_idx
                self.best_test_set = test_idx
            #break #if it's made this far, all tests are non-significant: yey!

        self.data = self.data.loc[self.best_all_samples]
        self.data.loc[self.best_train_set, 'cv_group'] = 'train'
        self.data.loc[self.best_test_set, 'cv_group'] = 'test'

        return (self.best_train_set, self.best_test_set)

    def save(self, fid):
        if self.best_min_p_val == 0:
            IOError('split not yet run, nothing to save!')

        self.data = self.data.sort_index()
        self.data.to_csv(fid, sep='\t')

    def plot_results(self):
        import matplotlib.pyplot as plt

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
            i =+ 1
            count = data.groupby([cat, 'cv_group']).size()
            count = count.unstack(level=0)[self.categorical[cat]]
            print(count)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            labels = self.categorical[cat]

            train_vals = count.loc['train'].values
            test_vals = count.loc['test'].values

            ax1.pie(train_vals, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.set_title('%s train group' %cat)
            ax2.pie(test_vals, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.set_title('%s test group' %cat)

        plt.show()


if __name__ == '__main__':
    tsv_path = '/users/steven/Documents/Syncthing/MscProjects/Decoding/code/multimodal/MultimodalDecoding/behavioral_data/ALL_BEHAV_2.tsv'
    subs_path = '/users/steven/Documents/Syncthing/MscProjects/Decoding/code/multimodal/MultimodalDecoding/behavioral_data/subjects_for_groups.txt'
    output_path = '/users/steven/Documents/Syncthing/MscProjects/Decoding/code/multimodal/MultimodalDecoding/behavioral_data/ALL_BEHAV_splitted.tsv'

    with open(subs_path, 'r') as fid:
        subs = fid.read().splitlines()

    train_size =  len(subs)/2
    test_size = len(subs)-train_size

    crosval = CrossvalSplitter(file_path = tsv_path, iterations=1000, train_size=train_size, test_size=test_size, include=subs, categorical={'Sekse': [1, 2]}, continuous=['Lftd', 'pashlerH'], ignore=9999)
    (train_idx, test_idx) = crosval.split(verbose=False)

    crosval.save(output_path)
    crosval.plot_results()