scikit-bold
--------
This package provides several modules to preprocess and transform BOLD-fMRI data, in a way
that is compatible with the *scikit-learn* machine learning library in Python. 
At this stage, the package contains a module to load in first-level FSL data
(i.e. estimates of patterns of activation for single trials; *data2mvp*), which
transforms these patterns into an object (Mvp, which stands for multivoxel pattern)
containing various attributes with meta-data and, importantly, the voxel-patterns
(trials x features) and class labels. Moreover, it contains a module to transform
these voxel-pattern matrices in various ways (*transformers* module), adhering
to the scikit-learn API for transformer-objects.

In the future, this package will be extended with representational similarity analysis
(RSA) transformers (i.e. transform voxels patterns to representational dissimilarity matrices),
and various other multivoxel pattern analyses (e.g. cross-validated MANOVA).

Installing scikit-bold
########

Althoug the package is very much in development, it can be installed using *pip*::
pip install scikit-bold


