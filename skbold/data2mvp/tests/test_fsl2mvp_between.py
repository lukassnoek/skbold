import os.path as op
from skbold.data2mvp.fsl2mvp import Fsl2mvpBetween
from skbold import testdata_path
import shutil
import h5py

def test_fsl2mvp_between():

    run1 = op.join(testdata_path, 'run1.feat')
    run2 = op.join(testdata_path, 'run2.feat')

    mvp_dir = op.join(testdata_path, 'mvp_data')

    if op.isdir(mvp_dir):
        shutil.rmtree(mvp_dir)

    for r in [run1, run2]:
        fsl2mvp = Fsl2mvpBetween(r, mask_threshold=0, beta2tstat=True,
                          ref_space='mni', mask_path=None, remove_contrast=[])
        fsl2mvp.glm2mvp()

        data_file = op.join(mvp_dir, 'test_data_data_%s.hdf5' %
                            op.basename(r).split('.')[0])
        hdr_file = op.join(mvp_dir, 'test_data_header_%s.pickle' %
                           op.basename(r).split('.')[0])
        assert (op.exists(data_file))
        assert (op.exists(hdr_file))
        shutil.rmtree(op.join(r, 'reg_standard'))

    fsl2mvp.merge_runs(idf='merged')
    merged_data = op.join(mvp_dir, 'test_data_data_merged.hdf5')
    merged_hdr = op.join(mvp_dir, 'test_data_header_merged.pickle')
    assert (op.exists(merged_data))
    assert (op.exists(merged_hdr))

    h5f = h5py.File(merged_data, 'r')
    data = h5f['data'][:]
    h5f.close()
    assert(data.shape[0] == 1)
    assert(data.shape[1] == 91 * 109 * 91 * 9 * 2)

    shutil.rmtree(mvp_dir)

    for r in [run1, run2]:
        fsl2mvp = Fsl2mvpBetween(r, mask_threshold=0, beta2tstat=True,
                          ref_space='epi', mask_path=None, remove_contrast=[])
        fsl2mvp.glm2mvp()
        assert(op.isdir(mvp_dir))

        data_file = op.join(mvp_dir, 'test_data_data_%s.hdf5' % op.basename(r).split('.')[0])
        hdr_file = op.join(mvp_dir, 'test_data_header_%s.pickle' % op.basename(r).split('.')[0])
        assert(op.exists(data_file))
        assert(op.exists(hdr_file))

    fsl2mvp.merge_runs(idf='merged')
    merged_data = op.join(mvp_dir, 'test_data_data_merged.hdf5')
    merged_hdr = op.join(mvp_dir, 'test_data_header_merged.pickle')
    assert(op.exists(merged_data))
    assert(op.exists(merged_hdr))

    h5f = h5py.File(merged_data, 'r')
    data = h5f['data'][:]
    h5f.close()

if __name__ == '__main__':
    test_fsl2mvp_between()