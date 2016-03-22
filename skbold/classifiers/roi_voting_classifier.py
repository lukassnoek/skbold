# Class to implement a voting classifier from the output of
# classifiers trained on different feature sets from different ROIs.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import glob
import os
import os.path as op
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import skbold.data.rois.harvard_oxford as roi


class RoiVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Voting classifier for an ensemble patterns from different ROIs.
    """

    default_clf = SVC(C=1.0, kernel='linear', probability=True,
                      decision_function_shape='ovo')

    def __init__(self, clf=default_clf, mask_type='unilateral', voting='soft',
                 n_cores=1):

        self.clf = clf
        self.voting = voting
        self.n_cores = n_cores
        self.mask_type = mask_type

        mask_dir = op.join(op.dirname(roi.__file__), 'harvard_oxford')
        self.masks = glob.glob(op.join(mask_dir, mask_type, '*nii.gz'))
