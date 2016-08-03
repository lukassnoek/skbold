# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
.. _ReadTheDocs: http://skbold.readthedocs.io

The data2mvp subpackage contains two major classes of relevance to the
rest of the skbold-package: ``MvpWithin`` and ``MvpBetween``, which are
subclasses of the ``Mvp`` class (defined in the ``core`` subpackage).

The ``MvpWithin`` object is meant as a data-structure that contains a set of
multivoxel fMRI patterns of *single trials, for a single subject*, hence
the 'within' part (i.e. within-subjects). Currently, it has a single
public method, ``create()``, that loads a set of contrasts from a FSL-firstlevel
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

"""

from mvp_within import MvpWithin
from mvp_between import MvpBetween

__all__ = ['MvpWithin', 'MvpBetween']