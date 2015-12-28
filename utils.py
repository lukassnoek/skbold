import numpy as np
import fnmatch


def sort_numbered_list(stat_list):
    """
    Sorts list with paths to statistic files (e.g. COPEs, VARCOPES),
    which are often sorted wrong (due to single and double digits).
    This function extracts the numbers from the stat files and sorts
    the original list accordingly.

    Args:
        stat_list: list with paths to files

    Returns:
        sorted_list: sorted stat_list
    """

    num_list = []
    for path in stat_list:
        num = [str(s) for s in str(os.path.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))

    sorted_list = [x for y, x in sorted(zip(num_list, stat_list))]
    return sorted_list


def convert_labels2numeric(class_labels, grouping):
    """
    Converts class labels (list of strings) to numeric numpy row vector.
    Groups string labels based on grouping, which is useful in factorial
    designs.

    Args:
        class_labels: list of strings (returned from extract_class_labels())
        grouping: list of strings that indicate grouping

    Returns:
        num_labels: numeric labels corresponding to class_labels
    """

    if len(grouping) == 0:
        grouping = np.unique(class_labels)

    num_labels = np.zeros(len(class_labels))
    for i, group in enumerate(grouping):

        if type(group) == list:
            matches = []
            for g in group:
                matches.append(fnmatch.filter(class_labels, '*%s*' % g))
            matches = [x for y in matches for x in y]
        else:
            matches = fnmatch.filter(class_labels, '*%s*' % group)
            matches = list(set(matches))

        for match in matches:
            for k, lab in enumerate(class_labels):
                if match == lab:
                    num_labels[k] = i + 1

    return np.array(num_labels)
