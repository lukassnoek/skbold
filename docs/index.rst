.. skbold documentation master file, created by
   sphinx-quickstart on Mon Aug  1 15:31:13 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Python package ``skbold`` offers a set of tools and utilities for
machine learning and RSA-type analyses of functional MRI (BOLD-fMRI) data.
Instead of (largely) reinventing the wheel, this package builds upon an existing
machine learning framework in Python: `scikit-learn <www.scikit-learn.org>`_.
Specifically, it offers a module with scikit-learn-style 'transformers' (with
the corresponding scikit-learn API) and some (experimental) scikit-learn
type estimators.

Next to these transformer- and estimator-functionalities, ``skbold`` offers
a new data-structure, the ``Mvp`` (Multivoxel pattern), that allows for an
efficient way to store and access data and metadata necessary for multivoxel
analyses of fMRI data. A novel feature of this data-structure is that it is
able to easily load data from `FSL <www.fmrib.ox.ac.uk/fsl>`_-FEAT output
directories. As the ``Mvp`` object is available in two 'options', they are
explained in more detail below.

MvpWithin vs. MvpBetween
------------------------
At the core, an ``Mvp``-object is simply a collection of data - a 2D array
of samples by features - and fMRI-specific metadata necessary to perform
customized preprocessing and feature engineering. However, machine learning
analyses, or more generally any type of multivoxel-type analysis (i.e. MVPA),
can be done in two basic ways.

One way is to perform analyses *within subjects*. This means that a model is
fit on each subjects' data separately. Data, in this context, often refers to
single-trial data, in which each trial comprises a sample in our data-matrix and
the values per voxel constitute our features. This type of analysis is
alternatively called *single-trial decoding*, and is often performed as an
alternative to massively (whole-brain) univariate analysis. Ultimately, this
type of analysis aims to predict some kind of attribute of the trials (for
example condition/class membership in classification analyses or some
continuous feature in regression analyses). Ultimately, group-analyses may
be done on subject-specific analysis metrics (such as classification accuracy
or R2-score) and group-level feature-importance maps may be calculated to
draw conclusions about the model's predictive power and the spatial
distribution of informative features, respectively.

With the apparent increase in large-sample neuroimaging datasets, another
type of analysis starts to become feasible, which we'll call *between subject*
analyses.

Below, a typical analysis workflow using ``skbold`` is described
to get a better idea of the package's functionality.

An example workflow
-------------------
Blabla

Contents:
=========
.. toctree::
   :maxdepth: 2

   source/skbold.classifiers
   source/skbold.transformers
   source/skbold.core
   source/skbold.data2mvp
   source/skbold.exp_model
   source/skbold.postproc
   source/skbold.quality_control
   source/skbold.utils

