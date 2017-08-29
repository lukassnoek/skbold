skbold - utilities and tools for machine learning on BOLD-fMRI data
===================================================================

.. image:: https://travis-ci.org/lukassnoek/skbold.svg?branch=master
    :target: https://travis-ci.org/lukassnoek/skbold

.. image:: https://readthedocs.org/projects/skbold/badge/?version=latest
    :target: http://skbold.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/lukassnoek/skbold/badge.svg?branch=develop
    :target: https://coveralls.io/github/lukassnoek/skbold?branch=develop

.. image:: https://img.shields.io/badge/python-2.7-blue.svg
    :target: https://www.python.org/download/releases/2.7

.. image:: https://img.shields.io/badge/python-3.5-blue.svg
    :target: https://www.python.org/downloads/release/python-350

The Python package ``skbold`` offers a set of tools and utilities for
machine learning analyses of functional MRI (BOLD-fMRI) data.
Instead of (largely) reinventing the wheel, this package builds upon an
existing machine learning framework in Python: `scikit-learn <http://scikit-learn.org/>`_.
The modules of skbold are applicable in several 'stages' of
typical pattern analyses (see image below), including pattern estimation,
data representation, pattern preprocessing, feature selection/extraction,
and model evaluation/feature visualization.

.. image:: img/scope.png
    :align: center

Documentation
-------------
Please see skbold's `ReadTheDocs <skbold.readthedocs.io>`_ page for more
info on how to use skbold!

Installation & dependencies
---------------------------

Although the package is very much in development, it can be installed using *pip*::

	$ pip install skbold

However, the pip-version is likely behind compared to the code on Github, so to get the
most up to date version, use git::

	$ pip install git+https://github.com/lukassnoek/skbold.git@master

Skbold is largely Python-only (both Python2.7 and Python3) and is built
around the "PyData" stack, including:

* Numpy
* Scipy
* Pandas
* Scikit-learn

And it uses the awesome `nibabel <http://nipy.org/nibabel/>`_ package
for reading/writing nifti-files. Also, skbold uses `FSL <https://fsl.fmrib.ox.ac.uk>`_
(primarily the ``FLIRT`` and ``applywarp`` functions) to transform files from functional
(native) to standard (here: MNI152 2mm) space. These FSL-calls are embedded in the
``convert2epi`` and ``convert2mni`` functions, so avoid this functionality if
you don't have a working FSL installation.

Authors & credits
-----------------
This package is being develop by `Lukas Snoek <lukas-snoek.com>`_
from the University of Amsterdam with contributions from
`Steven Miletic <https://github.com/StevenM1>`_ and help from
`Joost van Amersfoort <https://github.com/y0ast>`_.

License and contact
-------------------
The code is BSD (3-clause) licensed. You can find my contact details on my
`Github <https://github.com/lukassnoek>`_ profile page.
