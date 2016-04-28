# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

__version__ = '0.2.4'

import classifiers
import core
import data2mvp
import exp_model
import nipype_nodes
import postproc
import transformers
import utils

from os.path import dirname, join
from utils import DataHandler

data_path = join(dirname(dirname(utils.__file__)), 'data')
testdata_path = join(data_path, 'test_data')
roidata_path = join(data_path, 'ROIs')
harvardoxford_path = join(roidata_path, 'harvard_oxford')

loader = DataHandler(identifier='merged')
sample_data = loader.load_separate_sub(testdata_path)

__all__ = ['classifiers', 'core', 'data', 'data2mvp', 'exp_model',
           'nipype_nodes', 'postproc', 'transformers', 'utils', 'sample_data',
           'harvardoxford_path']
