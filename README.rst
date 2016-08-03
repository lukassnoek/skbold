skbold - utilities and tools for machine learning on BOLD-fMRI data
===================================================================

.. image:: https://travis-ci.org/lukassnoek/skbold.svg?branch=develop
    :target: https://travis-ci.org/lukassnoek/skbold

.. image:: https://readthedocs.org/projects/skbold/badge/?version=develop
    :target: http://skbold.readthedocs.io/en/develop/?badge=develop
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/lukassnoek/skbold/badge.svg
    :target: https://coveralls.io/github/lukassnoek/skbold

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

.. image:: img/MvpWithin.png

With the apparent increase in large-sample neuroimaging datasets, another
type of analysis starts to become feasible, which we'll call *between subject*
analyses. In this type of analyses, single subjects constitute the data's
samples and a corresponding single multivoxel pattern constitutes the data's
features.

Below, a typical analysis workflow using ``skbold`` is described
to get a better idea of the package's functionality.

An example workflow
-------------------
Blabla

Installing skbold
-----------------

Although the package is very much in development, it can be installed using *pip*::

	$ pip install skbold

However, the pip-version is likely behind compared to the code on Github, so to get the
most up to date version, use git::

	$ pip install git+https://github.com/lukassnoek/skbold.git@master

Or, alternatively, download the package as a zip-file from Github, unzip, and run::

	$ python setup.py install


.. code:: python

    from skbold.data2mvp import Fsl2mvp

    # Specify arguments
    firstlevel_dir = '/home/user/data/subject_001/firstlevel_dir.feat'
    mask_path = '/path/to/mask' # to index the patterns by, e.g., grey-matter mask
    mask_threshold = 0 # lower bound for probabilistic masks
    beta2tstat = True # convert beta estimates to t-values
    ref_space = 'epi' # load patterns in native functional-space (alternative: mni-space)
    remove_class = ['condition2', 'condition3'] # conditions to ignore

    # Initialize converter
    fsl2mvp = Fsl2mvp(directory=firstlevel_dir, mask_path=mask_path, mask_threshold=mask_threshold,
                      beta2tstat=beta2tstat, ref_space=ref_space, remove_class=remove_class)

    # Transform directory
    fsl2mvp.glm2mvp()

Credits
~~~~~~~
At the advent of this package, I knew next to nothing about Python programming
in general and packaging in specific. The `mlxtend
<https://github.com/rasbt/mlxtend>`_ package has been a great 'template' and
helped a great deal in structuring the current package. Also, `Steven
<https://github.com/StevenM1>`_ has contributed some very nice features as
part of his internship. Lastly, `Joost <https://github.com/y0ast`_ has been
a major help in virtually every single phase of this package!

License and contact
~~~~~~~~~~~~~~~~~~~
The code is BSD (3-clause) licensed. You can find my contact details at my
`Github profile page <https://github.com/lukassnoek>`_.