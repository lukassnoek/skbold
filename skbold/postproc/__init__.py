# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

"""
The postproc subpackage thus far only consists of the function
'extract_roi_info', which allows to calculate the amount of voxels (and
other statistics) per ROI in a single statistical brain map and output a
csv-file.
"""

from .extract_roi_info import extract_roi_info

__all__ = ['extract_roi_info']