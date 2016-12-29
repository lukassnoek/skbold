# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
.. _ReadTheDocs: http://skbold.readthedocs.io

The ``core`` subpackage contains skbold's most important data-structure: the
``Mvp``. This class forms the basis of the 'multivoxel-patterns' (i.e. mvp)
that are used throughout the package. Subclasses of Mvp (``MvpWithin`` and
``MvpBetween``) are also defined in this core module.

The ``MvpWithin`` object is meant as a data-structure that contains a set of
multivoxel fMRI patterns of *single trials, for a single subject*, hence
the 'within' part (i.e. within-subjects). Currently, it has a single
public method, ``create()``, loading a set of contrasts from a FSL-firstlevel
directory (i.e. a .feat-directory). Thus, importantly, it assumes that the
single-trial patterns are already modelled, on a single-trial basis, using
some kind of GLM. These trialwise patterns are then horizontally stacked
to create a 2D samples by features matrix, which is set to the ``X`` attribute
of MvpWithin.

The ``MvpBetween`` object is meant as a data-structure that contains a set of
multivoxel fMRI patterns of *single conditions, for a set of subjects*. It is,
so to say, a 'between-subjects' multivoxel pattern, in which subjects are
'samples'. In contrast to MvpWithin, contrasts that will be loaded are less
restricted in terms of their format; the only requisite is that they are
nifti files. Notably, the MvpBetween format allows to vertically stack
different kind of 'feature-sets' in a single MvpBetween object. For example,
it is possible to, for a given set of subjects, stack a functional contrast
(e.g. a high-load minus low-load functional contrast) with another functional
contrast (e.g. a conflict minus no-conflict functional contrast) in order to
use features from both sets to predict a certain psychometric or behavioral
variable of the corresponding subjects (such as, e.g., intelligence).
Also, the MvpBetween format allows to load (and stack!) VBM, TBSS,
resting-state (to extract connectivity measures), and dual-regression data.
More information can be found below in the API. A use case can be found on
the main page of ReadTheDocs_.

Also, functional-to-standard (i.e. ``convert2mni``) and standard-to-functional
(i.e. ``convert2epi``) warp-functions for niftis are defined here, because
they have caused circular import errors in the past.

"""

from .mvp import Mvp
from .convert_to_epi import convert2epi
from .convert_to_mni import convert2mni
from .mvp_between import MvpBetween
from .mvp_within import MvpWithin

__all__ = ['Mvp', 'convert2epi', 'convert2mni', 'MvpBetween', 'MvpWithin']
