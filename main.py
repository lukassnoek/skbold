"""
Main (testing) script for scikit_BOLD project.

Lukas Snoek
"""

__author__ = "Lukas Snoek"

import numpy as np
from glm2mvpa import create_subject_mats
import os
from sklearn.utils.estimator_checks import check_estimator

test_dir = '/home/lukas/DecodingEmotions/HWW_002/HWW_002-20140701-0004-WIPPM_Zinnen1.feat'
mask = '/home/lukas/Dropbox/PhD_projects/DynamicAffect_Multiscale/ROIs/GrayMatter.nii.gz'

#create_subject_mats(test_dir, mask=mask,
#                    mask_threshold=30, remove_class=[], grouping=[],
#                        normalize_to_mni=False, beta2tstat=True)

np.load('/home/lukas/DecodingEmotions/HWW_002/mvp_data/HWW_002_data.npy')



