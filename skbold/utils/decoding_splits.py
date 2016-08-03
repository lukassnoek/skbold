import pandas as pd
import os.path as op
import numpy as np
import random
from scipy.stats import ks_2samp, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import os

def make_counterbalanced_split(file_name, variable_name=None, split=0.5,
                               method='Kolmogorov-Smirnov', iterations=10000,
                               plot=True):
    """ Splits a dataset into counterbalanced train/test sets w.r.t. a
    variable.

    Parameters
    ----------
    file_name : str
        Absolute path to file (csv/tsv) with outcome variable that needs to be
        counterbalanced.
    variable_name : str
        Variable (column) name of outcome variable.
    split : float
        Train proportion to select.
    method : str
        Test according to which the dataset wil be split. Either
        'Kolmogorov-Smirnov' or 'Mann-Whitney-U'.
    iterations : int
        Number of permutations to run.
    plot : bool
        Whether to plot the histograms of the split.
    """

    if isinstance(file_name, pd.DataFrame):
        df = file_name
        base_dir = os.getcwd()
    elif op.isfile(file_name):
        df = pd.read_csv(file_name, sep='\t')
        base_dir = op.dirname(file_name)
    else:
        raise ValueError('File_name should be either a filename of a pandas dataframe' \
                         'or a dataframe itself.')

    if isinstance(variable_name, str):
        variable_name = [variable_name]

    if variable_name is not None:
        df = df[variable_name]

    if method == 'Kolmogorov-Smirnov':
        opt_func = ks_2samp
    elif method == 'Mann-Whitney-U':
        opt_func = mannwhitneyu
    else:
        raise ValueError("Use either 'Kolmorov-Smirnov' or 'Mann-Whitney-U test!")

    sub_col = [col for col in df.columns if col in ['Subject', 'Sub', 'sub_id', 'Subject_id']]
    if sub_col:
        df = df.set_index(sub_col)

    n_inst = len(df)
    n_col = len(df.columns)
    print('Optimizing split for %i variable(s), being:\n%r' % (n_col, df.columns.tolist()))

    results = []
    for i in range(iterations):
        idx = random.sample(range(n_inst), int(split * n_inst))
        train = df.iloc[idx]
        test_idx = [item for item in range(n_inst) if item not in set(idx)]
        test = df.iloc[test_idx]

        p_val = np.array([opt_func(train[col], test[col]) for col in df.columns]).mean()
        results.append((idx, p_val))

    p_vals = np.array([item[1] for item in results])
    p_max = p_vals.argmax()

    train_df = df.iloc[results[p_max][0]]
    train_df.to_csv(op.join(base_dir, 'train.csv'), sep='\t', header=True)
    test_df = df.iloc[[item for item in range(n_inst) if item not in set(results[p_max][0])]]
    test_df.to_csv(op.join(base_dir, 'test.csv'), sep='\t', header=True)

    if plot:

        for col in train_df.columns:
            sns_plot = sns.distplot(train_df[col], label='Train').get_figure()
            sns_plot = sns.distplot(test_df[col], label='Test').get_figure()
            plt.legend()
            plt.title('Train-test split, variable=%s, split=%f, iter=%i, method=%s' \
                    % (col, split, iterations, method))
            sns_plot.savefig(op.join(base_dir, 'distributions_%s' % col))
            plt.clf()
