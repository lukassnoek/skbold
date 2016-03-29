# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

from __future__ import division, print_function

def load_scaninfo(in_file):

    import cPickle
    scaninfo = cPickle.load(open(in_file))
    TR = scaninfo['repetition_time']
    x, y = scaninfo['scan_resolution']
    z = scaninfo['max_slices']
    dims = (x, y, z)
    return(TR, dims, scaninfo)