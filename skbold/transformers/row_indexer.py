import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RowIndexer(object):
    """
    Selects a subset of rows from an Mvp object.

    Notes
    -----
    NOT a scikit-learn style transformer.

    Parameters
    ----------
    idx : ndarray
        Array with indices.
    mvp : mvp-object
        Mvp-object to drawn metadata from.
    """

    def __init__(self, mvp, train_idx):
        self.idx = train_idx
        self.mvp = mvp

    def transform(self):
        """

        Returns
        -------
        mvp : mvp-object
            Indexed mvp-object.
        X_not_selected : ndarray
            Data which has not been selected.
        y_not_selected : ndarray
            Labels which have not been selected.
        """
        mvp = self.mvp
        selection = np.zeros(mvp.X.shape[0], dtype=bool)
        selection[self.idx] = True
        X_not_selected = mvp.X[~selection,:]
        y_not_selected = mvp.y[~selection]
        mvp.X = mvp.X[selection,:]
        mvp.y = mvp.y[selection]

        return mvp, X_not_selected, y_not_selected