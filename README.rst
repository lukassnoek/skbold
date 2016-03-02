skbold
--------
This package provides several tools to (pre)process and analyse BOLD-fMRI
data. Most of the package's functionality is centered around machine learning
analyses, which are structured similarly to the *scikit-learn* machine learning
library in Python, using the same .fit() and .transform() methods.

Next to these machine learning tools, this package contains some 
miscellaneous tools to, for example, parse Presentation (www.neurobs.com)
logfiles, extract region-of-interest information from activation-based
statistics images (niftis), and various other tools that I use in my research.
So basically *skbold* contains all the stuff I program during my PhD.

More specifically, at this stage, the package contains a module to load in first-level FSL data
(i.e. estimates of patterns of activation for single trials; *data2mvp*), which
transforms these patterns into an object (Mvp, which stands for multivoxel pattern)
containing various attributes with meta-data and, importantly, the voxel-patterns
(trials x features) and class labels. Moreover, it contains a module to transform
these voxel-pattern matrices in various ways (*transformers* module), adhering
to the scikit-learn API for transformer-objects.

In the future, this package might be extended with representational similarity analysis
(RSA) transformers (i.e. transform voxels patterns to representational dissimilarity matrices),
and various other multivoxel pattern analyses (e.g. cross-validated MANOVA).

Installing skbold
########

Although the package is very much in development, it can be installed using *pip*::

	pip install skbold

However, the pip-version is likely behind compared to the code on Github, so to get the
most up to date version, use git::

	pip install git+https://github.com/lukassnoek/skbold.git@master

Or, alternatively, download the package as a zip-file from Github, unzip, and run::
	
	python setup.py install


