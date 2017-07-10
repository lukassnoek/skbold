import skbold
import os.path as op
roi_dir = op.join(op.dirname(skbold.__file__), 'data', 'ROIs')


available_atlases = ['HarvardOxford-Cortical',
                     'HarvardOxford-Subcortical',
                     'MNI', 'JHU-labels', 'JHU-tracts',
                     # 'Talairach',
                     'Yeo2011']

other_rois = {'GrayMatter_prob': op.join(roi_dir, 'other',
                                         'GrayMatter_prob.nii.gz'),
              'MNI152_2mm': op.join(roi_dir, 'MNI152_2mm.nii.gz'),
              'VentricleMask': op.join(roi_dir, 'other',
                                       'VentricleMask.nii.gz'),
              'GrayMatter_tp': op.join(roi_dir, 'tissuepriors',
                                       'avg152T1_gray.nii.gz'),
              'WhiteMatter_tp': op.join(roi_dir, 'tissuepriors',
                                        'avg152T1_white.nii.gz'),
              'CSF_tp': op.join(roi_dir, 'tissuepriors',
                                'avg152T1_csf.nii.gz')}
