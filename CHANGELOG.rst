CHANGELOG
=========

Version 0.3.3
-------------
Refactoring of `postproc` and `feature_selection`/`feature_extraction` modules.

- ENH: refactoring `MvpResults`, no subclasses anymore, one consistent class with variable metrics
- ENH: proper distinction between feature-selection and feature-extraction functionality

Version 0.3.2
-------------
Minor bugfixes and extend MvpWithin functionality.

- ENH: MvpWithin can now load in any statistic-file (tstat, zstat)
- FIX: convert img/hdr tissuepriors to nifti.gz

Version 0.3.1
-------------
- ENH: FsfCrawler now also works with (BIDS-style) tsv event-files
- FIX: remove confound-regression methods in MvpBetween (this should be done fold-wise)
- ENH: add ConfoundRegressor transformer in new confounds-module

Versions < 0.3.0
----------------
The changelog for versions < 0.3.0 have not been documented.