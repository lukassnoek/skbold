# Function to sort an unsorted list (due to globbing) using a number
# occuring in the path.

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import os.path as op


def sort_numbered_list(stat_list):
    """ Sorts a list containing numbers.

    Sorts list with paths to statistic files (e.g. COPEs, VARCOPES),
    which are often sorted wrong (due to single and double digits).
    This function extracts the numbers from the stat files and sorts
    the original list accordingly.

    Parameters
    ----------
    stat_list : list or str
        list with absolute paths to files

    Returns
    -------
    sorted_list : list of str
        sorted stat_list
    """

    num_list = []
    for path in stat_list:
        num = [str(s) for s in str(op.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))

    sorted_list = [x for y, x in sorted(zip(num_list, stat_list))]
    return sorted_list
