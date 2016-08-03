# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The quality_control subpackage contains several functions that can be run
during or after preprocessing to check whether it has executed correctly and if
all files show the desired characteristics. For example, the check_nifti_header
function summarizes the nifti-header from all functional files such that, e.g.,
incorrect TR-values can be easily spotted. The check_mc_output parses FSL
motion-correction parameters and summarizes it in a csv to look for subjects
that clearly have moved too much.
"""

from .check_preprocessing_output import check_nifti_header, check_mc_output

__all__ = ['check_nifti_header', 'check_mc_output']
