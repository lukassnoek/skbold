# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD


from . import estimators
import core
import exp_model
import postproc
import feature_extraction
import feature_selection
import utils
from os.path import dirname, join

__version__ = '0.3.0'

data_path = join(dirname(dirname(utils.__file__)), 'data')
testdata_path = join(data_path, 'test_data')
roidata_path = join(data_path, 'ROIs')
harvardoxford_path = join(roidata_path, 'harvard_oxford')

__all__ = ['estimators', 'core', 'data', 'exp_model',
           'postproc', 'feature_extraction', 'utils',
           'harvardoxford_path', 'feature_selection']
