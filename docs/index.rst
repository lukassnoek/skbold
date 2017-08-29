.. skbold documentation master file, created by
   sphinx-quickstart on Mon Aug  1 15:31:13 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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

The documentation of skbold is split up into three sections:

* :ref:`getting-started`
* :ref:`examples`
* :ref:`API`

.. _getting-started:

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting-started/installation
   getting-started/data_organization

.. _examples:

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/mvp
   examples/pipelines

.. _API:

.. toctree::
   :maxdepth: 2
   :caption: API

   source/skbold.core
   source/skbold.exp_model
   source/skbold.feature_extraction
   source/skbold.feature_selection
   source/skbold.pipelines
   source/skbold.postproc
   source/skbold.preproc
   source/skbold.utils
