import glob
import os
import os.path as op
import skbold
from skbold.data2mvp.fsl2mvp import Fsl2mvpBetween

def test_fsl2mvp_between():
    feat_dir = '/Users/steven/Desktop/pioptest'

    subdir = glob.glob(os.path.join(feat_dir, 'pi*'))

    gm_mask = op.join(op.dirname(op.dirname(skbold.utils.__file__)), 'data', 'ROIs', 'GrayMatter.nii.gz')

    contr = {'faces': ['emo-neu', 'pos-neu'],
             'wm': ['act-pas'],
             'harriri': ['emo-control'],
             'anticipatie': ['mismatch-match'],
             'gstroop': ['con-incon']}

    true_labels = ['pos-neu', 'emo-neu', 'con-incon', 'emo-control', 'act-pas', 'mismatch-match']

    for sub in subdir:
        taskdirs = glob.glob(os.path.join(sub, '*.feat'))

        for task in taskdirs:
            taskname = os.path.basename(os.path.normpath(task)).split('piop')[1][:-5]
            fsl2mvpb = Fsl2mvpBetween(directory=task, mask_threshold=0, beta2tstat=True,
                                ref_space='mni', mask_path=gm_mask, remove_cope=contr[taskname],
                                invert_selection=True, output_var_file='raven.txt')
            fsl2mvpb.glm2mvp()
        fsl2mvpb.merge_runs()

    from skbold.utils import DataHandler

    alldat = DataHandler()
    alldat = alldat.load_concatenated_subs(directory=feat_dir)

    assert(alldat.cope_labels == true_labels)
    assert(alldat.X.shape == (3, 269412 * 6))
    assert(alldat.y == [14, 25, 19])

if __name__=='__main__':
    test_fsl2mvp_between()