# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The exp_model subpackage contains some (pre)processing functions and classes
that help in preparing to fit a first-level GLM on fMRI data across multiple
subjects.

.. _Presentation: https://www.neurobs.com/menu_presentation/menu_features/features_overview
.. _Eprime: https://www.pstnet.com/eprime.cfm

The PresentationLogfileCrawler (and its function-equivalent
'parse_presentation_logfile') can be used to parse Presentation_-logfile,
which are often used at the University of Amsterdam.

Also, there is an experimental Eprime-logfile converter, which converts the
Eprime_ .txt-file to a tsv-file format.

"""

from .parse_presentation_logfile import parse_presentation_logfile
from .parse_presentation_logfile import PresentationLogfileCrawler
from .convert_eprime import Eprime2tsv
from batch_fsf import FsfCrawler

__all__ = ['parse_presentation_logfile', 'PresentationLogfileCrawler',
           'Eprime2tsv', 'FsfCrawler']