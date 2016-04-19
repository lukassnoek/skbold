# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import os.path as op
import h5py
import shutil
from skbold.data2mvp import Fsl2mvp
from skbold import testdata_path

def test_fsl2mvp():

    run1 = op.join(testdata_path, 'run1.feat')
    run2 = op.join(testdata_path, 'run2.feat')

    true_labels = ['actie', 'actie', 'actie',
                   'interoception', 'interoception', 'interoception',
                   'situation', 'situation', 'situation']

    mvp_dir = op.join(testdata_path, 'mvp_data')

    if op.isdir(mvp_dir):
        shutil.rmtree(mvp_dir)

    for r in [run1, run2]:
        fsl2mvp = Fsl2mvp(r, mask_threshold=0, beta2tstat=True,
                          ref_space='mni', mask_path=None, remove_class=[])
        fsl2mvp.glm2mvp()
        data_file = op.join(mvp_dir, 'test_data_data_%s.hdf5' %
                            op.basename(r).split('.')[0])
        hdr_file = op.join(mvp_dir, 'test_data_header_%s.pickle' %
                           op.basename(r).split('.')[0])
        assert (op.exists(data_file))
        assert (op.exists(hdr_file))
        shutil.rmtree(op.join(r, 'reg_standard'))

    fsl2mvp.merge_runs(iD='merged')
    merged_data = op.join(mvp_dir, 'test_data_data_merged.hdf5')
    merged_hdr = op.join(mvp_dir, 'test_data_header_merged.pickle')
    assert (op.exists(merged_data))
    assert (op.exists(merged_hdr))

    h5f = h5py.File(merged_data, 'r')
    data = h5f['data'][:]
    h5f.close()
    assert (data.shape[1] == 91 * 109 * 91)

    shutil.rmtree(mvp_dir)

    for r in [run1, run2]:
        fsl2mvp = Fsl2mvp(r, mask_threshold=0, beta2tstat=True,
                          ref_space='epi', mask_path=None, remove_class=[])
        fsl2mvp.glm2mvp()
        assert(fsl2mvp.class_labels == true_labels)
        assert(op.isdir(mvp_dir))

        data_file = op.join(mvp_dir, 'test_data_data_%s.hdf5' % op.basename(r).split('.')[0])
        hdr_file = op.join(mvp_dir, 'test_data_header_%s.pickle' % op.basename(r).split('.')[0])
        assert(op.exists(data_file))
        assert(op.exists(hdr_file))

    fsl2mvp.merge_runs(iD='merged')
    merged_data = op.join(mvp_dir, 'test_data_data_merged.hdf5')
    merged_hdr = op.join(mvp_dir, 'test_data_header_merged.pickle')
    assert(op.exists(merged_data))
    assert(op.exists(merged_hdr))

    h5f = h5py.File(merged_data, 'r')
    data = h5f['data'][:]
    h5f.close()

    assert(data.shape[0] == len(true_labels) * 2)


if __name__ == '__main__':

    test_fsl2mvp()