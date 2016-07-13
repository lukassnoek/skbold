import os
import os.path as op
import pandas as pd
from fnmatch import fnmatch
import numpy as np
import nibabel as nib
from skbold.core import Mvp
from glob import glob
import scipy.stats as stat

class MvpBetween(Mvp):

    def __init__(self, source, subject_idf='sub0???', remove_zeros=True,
                 X=None, y=None, mask=None, mask_threshold=0, subject_list=None):

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
        self.data_name = []

        if not isinstance(source, dict):
            msg = "Source must be a dictionary with type (e.g. 'VBM') " \
                  "path ('path/to/VBM_file.nii.gz) mappings!"
            raise TypeError(msg)

    def create(self):

        self._load_mask()
        self._check_complete_data()

        # Loop over data-types as defined in source
        for data_type, args in self.source.iteritems():

            print('Processing: %s ...' % data_type)

            self.data_name.append(data_type)

            if any(fnmatch(data_type, typ) for typ in ['VBM', 'TBSS', 'Contrast*']):
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

        # If remove_zeros, all columns with all zeros are remove to reduce space
        if self.remove_zeros:
            indices = [np.invert(x == 0).all(axis=0) for x in self.X]

            for i, index in enumerate(indices):
                # Also other attributes are adapted to new shape
                self.X[i] = self.X[i][:, index]
                self.voxel_idx[i] = self.voxel_idx[i][index]
                self.featureset_id[i] = self.featureset_id[i][index]

        self.X = np.concatenate(self.X, axis=1)
        self.featureset_id = np.concatenate(self.featureset_id, axis=0)
        self.voxel_idx = np.concatenate(self.voxel_idx, axis=0)

        # 'Safety-check' to see whether everything corresponds properly
        assert(self.X.shape[1] == self.featureset_id.shape[0])
        all_vox = np.sum([self.voxel_idx[i].size for i in range(len(self.voxel_idx))])
        assert(self.X.shape[1] == all_vox)

        print("Final size of array: %r" % list(self.X.shape))

    def add_outcome_var(self, file_path, col_name, normalize=True,
                        binarize=None):
        """ Adds target-variabel from csv """

        # Assumes index corresponds to self.common_subjects
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        behav = df.loc[df.index.isin(self.common_subjects), col_name]
        y = np.array(behav.sort_index())

        if normalize:
            y = (y - y.mean()) / y.std()

        if binarize['type'] == 'percentile':
            y = np.array([stat.percentileofscore(y, a, 'rank') for a in y])
            idx = (y < binarize['low']) | (y > binarize['high'])
            y = (y[idx] > 50).astype(int)
            self.X = self.X[idx, :]
        elif binarize['type'] == 'zscore':
            y = (y - y.mean()) / y.std() # just to be sure
            idx = np.abs(y) > binarize['std']
            y = (y[idx] > 0).astype(int)
            self.X = self.X[idx, :]
        elif binarize['type'] == 'constant':
            y = (y > binarize['cutoff']).astype(int)
        elif binarize['type'] == 'median': # median-split
            median = np.median(y)
            y = (y > median).astype(int)

        self.y = y

    def _load_mask(self):

        self.mask_index = nib.load(self.mask).get_data() > self.mask_threshold
        self.mask_shape = self.mask_index.shape
        self.mask_index = self.mask_index.ravel()

    def _load_dual_reg(self, args):

        data = []

        # Kinda ugly, but switching loop over subjects and loop over comps
        # is not possible.
        for path in args['paths']:
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

        for i in range(len(final_comps)):
            feature_ids = np.ones(vol.shape[1], dtype=np.uint32) * len(self.X)
            self.featureset_id.append(feature_ids)
            data_concat = np.concatenate([sub[i] for sub in data], axis=0)
            self.X.append(data_concat)

    def _load_3D(self, args):

        data = []
        for path in args['paths']:
            tmp = nib.load(path)
            tmp_data = tmp.get_data().ravel()

            if self.mask_shape == tmp.shape:
                tmp_data = tmp_data[self.mask_index]

            data.append(tmp_data[np.newaxis, :])

        voxel_idx = np.arange(np.prod(tmp.shape))

        if self.mask_shape == tmp.shape:
            voxel_idx = voxel_idx[self.mask_index]

        self.voxel_idx.append(voxel_idx)
        self.affine.append(tmp.affine)
        self.data_shape.append(tmp.shape)

        data = np.concatenate(data, axis=0)
        feature_ids = np.ones(data.shape[1], dtype=np.uint32) * len(self.X)
        self.featureset_id.append(feature_ids)
        self.X.append(data)

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
    import skbold
    import os.path as op

    base_dir = '/media/lukas/piop/PIOP/FirstLevel_piop'
    gm = op.join(op.dirname(skbold.__file__), 'data', 'ROIs', 'GrayMatter.nii.gz')
    source = {}
    #source['VBM'] = {'path': op.join(base_dir, 'pi*', '*_vbm.nii.gz')}
    #source['TBSS'] = {'path': op.join(base_dir, 'pi*', '*_tbss.nii.gz')}
    source['ContrastWM'] = {'path': op.join(base_dir, 'pi*', '*piopwm.feat', 'reg_standard', '*tstat3.nii.gz')}
    source['ContrastGstroop'] = {'path': op.join(base_dir, 'pi*', '*piopgstroop.feat', 'reg_standard', '*tstat3.nii.gz')}
    source['ContrastHarriri'] = {'path': op.join(base_dir, 'pi*', '*piopharriri.feat', 'reg_standard', '*tstat3.nii.gz')}

    mvp_between = MvpBetween(source=source, remove_zeros=True, mask=gm,
                             mask_threshold=0, subject_idf='pi0???')
                             #subject_list=['pi0041', 'pi0042', 'pi0010', 'pi0230'])
    mvp_between.create()
    mvp_between.write(path='/home/lukas', name='between', backend='joblib')