skbold - utilities for machine learning analyses on BOLD-fMRI data
------------------------------------------------------------------

Functional MRI (fMRI) data has traditionally been analyzed by calculating average
signal differences between conditions. In the past decade, however,
pattern-based type of analyses have become increasingly popular. Especially
machine-learning based analyses experience a surge in popularity among
(cognitive) neuroscientists.

While many great resources for domain-general machine learning exists
(e.g. `scikit-learn <www.scikit-learn.org>`_,
`caret <http://topepo.github.io/caret/index.html>`_, and
`libsvm <https://www.csie.ntu.edu.tw/~cjlin/libsvm>`_), few resources are
available specifically for machine learning analyses of neuroimaging data
(but see `nilearn <https://nilearn.github.io/>`_).

As my PhD involved mainly machine learning analyses of fMRI data, I decided
to bundle my (relevant) code into this package, which provides a nice
opportunity for me to develop my programming skills by forcing me to write
concise, readable, and efficient code.

The skbold-package contains mostly extensions and utilities for machine learning
analyses of fMRI data. Its structure/setup draws heavily upon the *scikit-learn*
(sklearn, hence the name) machine learning library in Python. Also, credit should
be given to `this <http://rasbt.github.io/mlxtend/>`_ repository, as it has
a similar purpose and served as an example for much of my code.

Functionality
-------------

Currently, the package contains two main features:
1) Transforming first-level FSL directories to scikit-learn compatible data
structures (data2mvp module);
2) Classes that provide scikit-learn style *transformers* that transform
patterns of fMRI data in various ways as preprocessing or feature-selection steps
in ML analyses.

More extensive documentation will be developed soon.

Installing skbold
-----------------

Although the package is very much in development, it can be installed using *pip*::

	pip install skbold

However, the pip-version is likely behind compared to the code on Github, so to get the
most up to date version, use git::

	pip install git+https://github.com/lukassnoek/skbold.git@master

Or, alternatively, download the package as a zip-file from Github, unzip, and run::
	
	python setup.py install


