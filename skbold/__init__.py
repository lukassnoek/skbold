# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

__version__ = '0.3.0'

import classifiers
import core
import data2mvp
import exp_model
import postproc
import transformers
import utils

from os.path import dirname, join

data_path = join(dirname(dirname(utils.__file__)), 'data')
testdata_path = join(data_path, 'test_data')
roidata_path = join(data_path, 'ROIs')
harvardoxford_path = join(roidata_path, 'harvard_oxford')

__all__ = ['classifiers', 'core', 'data', 'data2mvp', 'exp_model',
           'postproc', 'transformers', 'utils', 'harvardoxford_path']
