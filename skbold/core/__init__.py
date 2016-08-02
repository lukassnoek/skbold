# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The ``core`` subpackage contains skbold's most important data-structure: the
``Mvp``. This class forms the basis of the 'multivoxel-patterns' (i.e. mvp)
that are used throughout the package. Subclasses of Mvp (most prominently
``MvpWithin`` and ``MvpBetween``) are defined in the ``data2mvp`` subpackage.
Also, functional-to-standard (i.e. ``convert2mni``) and standard-to-functional
(i.e. ``convert2epi``) warp-functions for niftis are defined here, because
they have caused circular import errors in the past.
"""

from .mvp import Mvp
from .convert_to_epi import convert2epi
from .convert_to_mni import convert2mni

__all__ = ['Mvp', 'convert2epi', 'convert2mni']