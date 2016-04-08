# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function


def filter_list(in_files, to_match):

    if len(to_match) < 2:
        msg = 'Filtering list of < 2, so probably not necessary.'
        raise ValueError(msg)

