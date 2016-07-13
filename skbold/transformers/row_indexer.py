import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RowIndexer(object):
    ### NOT a SK-LEARN TRANSFORMER CLASS ###

    def __init__(self, mvp, train_idx):
        self.idx = train_idx
        self.mvp = mvp

    def fit(self):
        return self

    def transform(self):
        mvp = self.mvp
        selection = np.ones(mvp.X.shape[0], dtype=bool)
        selection[self.idx] = False
        X_not_selected = mvp.X[~selection,:]
        y_not_selected = mvp.y[~selection]
        mvp.X = mvp.X[selection,:]
        mvp.y = mvp.y[selection]

        return mvp, X_not_selected, y_not_selected


if __name__ == '__main__':
    import joblib
    from skbold.data2mvp import MvpBetween

    mvp = joblib.load(filename='/users/steven/Documents/Syncthing/MscProjects/Decoding/code/multimodal/MultimodalDecoding/data/between.jl')
    print(mvp.X.shape)

    r_indexer = RowIndexer(mvp=mvp, train_idx=[0,2, 104])
    (mvp2, X_test, y_test) = r_indexer.transform()

    print(mvp2.X.shape)
    print(mvp2.y.shape)
    print(X_test.shape)
    print(y_test.shape)