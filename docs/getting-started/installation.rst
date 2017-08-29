Installation & dependencies
---------------------------

Although the package is very much in development, it can be installed using
*pip*::

	$ pip install skbold

However, the pip-version is likely behind compared to the code on Github, so to
get the most up to date version, use git::

	$ pip install git+https://github.com/lukassnoek/skbold.git@master

Skbold is largely Python-only (both Python2.7 and Python3) and is built
around the "PyData" stack, including:

* Numpy
* Scipy
* Pandas
* Scikit-learn

And it uses the awesome `nibabel <http://nipy.org/nibabel/>`_ package
for reading/writing nifti-files. Also, skbold uses
`FSL <https://fsl.fmrib.ox.ac.uk>`_ (primarily the ``FLIRT`` and ``applywarp``
functions) to transform files from functional (native) to standard
(here: MNI152 2mm) space. These FSL-calls are embedded in the ``convert2epi``
and ``convert2mni`` functions, so avoid this functionality if you don't have
a working FSL installation.
