import os
import os.path as op
from fnmatch import fnmatch
import pandas as pd
import numpy as np
import nibabel as nib
from skbold.core import Mvp, convert2epi, convert2mni
from skbold.utils import sort_numbered_list
from glob import glob

# TO DO:
# - field 'name' in args (especially for contrasts)

class MvpBetween(Mvp):

    def __init__(self, source, subject_idf='sub0???', output_var_file=None,
                 remove_zeros=True, X=None, y=None, mask=None, mask_threshold=0,
                 subject_list=None):

        super(MvpBetween, self).__init__(X=X, y=y, mask=mask,
                                         mask_threshold=mask_threshold)

        self.source = source
        self.remove_zeros = remove_zeros
        self.subject_idf = subject_idf
        self.subject_list = subject_list
        self.common_subjects = None
        self.y = None
        self.X = []
        self.featureset_id = []
        self.affine = [] # This could be an array
        self.nifti_header = []
        self.voxel_idx = []
        self.data_shape = [] # This could be an array
        self.mask_shape = None

        if not isinstance(source, dict):
            msg = "Source must be a dictionary with type (e.g. 'VBM') " \
                  "path ('path/to/VBM_file.nii.gz) mappings!"
            raise TypeError(msg)

    def create(self):

        self._load_mask()
        self._check_complete_data()

        for data_type, args in self.source.iteritems():

            if data_type in ['VBM', 'TBSS', 'Contrast']:
                self._load_3D(args)
            elif data_type == 'dual_reg':
                self._load_dual_reg(args)
            elif data_type == '4D_func':
                msg = "Data_type '4D_func' not yet implemented!"
                print(msg)
            else:
                allowed = ['VBM', 'dual_reg', 'TBSS', 'Contrast', '4D_func']
                msg = "Data-type '%s' not recognized; please use one of the " \
                      "following: %r" % (data_type, allowed)
                raise KeyError(msg)

        # Here, everything is concatenated (but by axis=1, instead of axis=0 in
        # MvpWithin)
        self.X = np.concatenate(self.X, axis=1)
        self.featureset_id = np.concatenate(self.featureset_id, axis=0)

        if self.remove_zeros:
            idx = np.invert((self.X == 0)).all(axis=0)
            self.X = self.X[:, idx]
            self.voxel_idx = self.voxel_idx[idx]

        #if self.output_var_file is not None:
        #    sub_paths = 
        #    _ = [self._add_outcome_var(op.join(sub, self.output_var_file)) for sub in sub_paths)]

        print("Final size of array: %r" % list(self.X.shape))

    def _add_outcome_var(self, file_path):
        with open(file_path, 'rb') as f:
            subject_y = float(f.readline())
        return subject_y

    def _load_mask(self):

        self.mask_index = nib.load(self.mask).get_data() > self.mask_threshold
        self.mask_shape = self.mask_index.shape
        self.mask_index = self.mask_index.ravel()

    def _load_dual_reg(self, args):

        paths = args['paths']
        subjects = args['subjects']

        data = []

        # Kinda ugly, but switching loop over subjects and loop over comps
        # is not possible.
        for path in paths:
            tmp = nib.load(path)
            tmp_shape = tmp.shape[:3]
            n_comps = tmp.shape[-1]

            if args['components'] is not None:

                final_comps = set.intersection(set(args['components']),
                                               set(range(1, n_comps + 1)))
                final_comps = [x - 1 for x in list(final_comps)]
            else:
                final_comps = range(n_comps)

            vols = []
            for n_comp in final_comps:
                vol = tmp.dataobj[..., n_comp].ravel()

                if self.mask_shape == tmp_shape:
                    vol = vol[self.mask_index]

                vol = vol[np.newaxis, :]
                vols.append(vol)
            data.append(vols)

        # Hack to infer attributes by looking at last volume
        voxel_idx = np.arange(np.prod(tmp_shape))
        if self.mask_shape == tmp_shape:
            voxel_idx = voxel_idx[self.mask_index]

        _ = [self.voxel_idx.append(voxel_idx) for i in final_comps]
        _ = [self.data_shape.append(tmp_shape) for i in final_comps]
        _ = [self.affine.append(tmp.affine) for i in final_comps]
        _ = [self.nifti_header.append(tmp.header) for i in final_comps]

        self.featureset_id.append([np.ones(vol.shape[1]) * (i+1)
                                   for i in range(n_comps)])

        # Concatenate list (subjects) of lists (components)
        data_concat = np.concatenate([np.concatenate(sub, axis=1)
                                      for sub in data], axis=0)

        self.X.append(data_concat)

    def _load_3D(self, args):

        paths = args['paths']
        subjects = args['subjects']

        data = []
        for path in paths:
            tmp = nib.load(path)
            self.affine.append(tmp.affine)
            self.data_shape.append(tmp.shape)
            tmp_data = tmp.get_data().ravel()

            voxel_idx = np.arange(self.mask_index.size)

            if self.mask_shape == tmp.shape:
                tmp_data = tmp_data[self.mask_index]
                voxel_idx = voxel_idx[self.mask_index]

            self.voxel_idx.append(voxel_idx)
            data.append(tmp_data[np.newaxis, :])

        data = np.concatenate(data, axis=0)
        self.X.append(data)
        self.featureset_id.append(np.ones(data.shape[1],
                                          dtype=np.uint32) * len(self.X))

    def _check_complete_data(self):

        for data_type, args in self.source.iteritems():
            args['paths'] = sorted(glob(args['path']))

            ex_path = args['paths'][0].split(os.sep)
            idx = [True if fnmatch(p, self.subject_idf) else False for p in ex_path]
            position = list(np.arange(len(ex_path))[np.array(idx)])

            if len(position) > 1:
                msg = "Couldn't resolve to which subject the file '%s' belongs" \
                      "because subject-identifier (%s) is ambiguous!" % (ex_path,
                                                                         self.subject_idf)
                raise ValueError(msg)
            elif len(position) == 1:
                position = position[0]
            else:
                msg = "Couldn't determine which subject belongs to which path" \
                      "for data-type = '%s'" % data_type
                raise ValueError(msg)

            args['position'] = position
            args['subjects'] = [p.split(os.sep)[position] for p in args['paths']]

        all_subjects = [set(args['subjects']) for args in self.source.itervalues()]

        if self.subject_list:
            all_subjects.append(set(self.subject_list))

        self.common_subjects = sorted(set.intersection(*all_subjects))

        print("Found a set of %i complete subjects for data-types: %r" % \
              (len(self.common_subjects), [key for key in self.source]))

        for data_type, args in self.source.iteritems():
            args['paths'] = [p for p in args['paths']
                             if p.split(os.sep)[args['position']] in self.common_subjects]

if __name__ == '__main__':

    base_dir = '/media/lukas/piop/PIOP/FirstLevel_piop'
    source = {}
    source['dual_reg'] = {'path': op.join(base_dir, 'pi*', '*_dualreg.nii.gz'),
                          'components': [1, 5]}
    #source['VBM'] = {'path': op.join(base_dir, 'pi*', '*_vbm.nii.gz')}
    #source['TBSS'] = {'path': op.join(base_dir, 'pi*', '*_tbss.nii.gz')}
    #source['Contrast'] = {'path': op.join(base_dir, 'pi*', '*piopwm*', 'reg_standard',
    #                                      'tstat3.nii.gz')}

    mvp_between = MvpBetween(source=source, remove_zeros=True, mask='/home/lukas/GrayMatter.nii.gz',
                             mask_threshold=0, subject_idf='pi0???',
                             subject_list=['pi0041', 'pi0042', 'pi0010', 'pi0230'])
    mvp_between.create()
    mvp_between.write(path='/home/lukas', backend='joblib')
