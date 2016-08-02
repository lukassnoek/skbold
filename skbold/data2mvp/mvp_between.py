import os
import os.path as op
import pandas as pd
from fnmatch import fnmatch
import numpy as np
import nibabel as nib
from skbold.core import Mvp
from glob import glob
import scipy.stats as stat
import re
from sklearn.linear_model import LinearRegression
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_harvard_oxford
from sklearn.covariance import GraphLassoCV
from joblib import Parallel, delayed


class MvpBetween(Mvp):
    """ MvpBetween class for subject x feature arrays of (f)MRI data. """

    def __init__(self, source, subject_idf='sub0???', remove_zeros=True,
                 X=None, y=None, mask=None, mask_threshold=0, subject_list=None):
        """ Initializes an MvpBetween object. """

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

        # Globs all paths and checks for which subs there is complete data
        self._check_complete_data()

        # If a mask is specified, load it
        if self.mask is not None:
            self._load_mask(self.mask, self.mask_threshold)

        # Loop over data-types as defined in source
        for data_type, args in self.source.iteritems():

            print('Processing: %s ...' % data_type)
            self.data_name.append(data_type)

            if 'mask' in args.keys():
                th = args['mask_threshold'] if 'mask_threshold' in args.keys() else 0
                self._load_mask(args['mask'], th)
            elif self.mask is not None:
                self._load_mask(self.mask, self.mask_threshold)

            if any(fnmatch(data_type, typ) for typ in ['VBM', 'TBSS', 'Contrast*']):
                self._load_3D(args)
            elif data_type == 'dual_reg':
                self._load_dual_reg(args)
            elif '4D_func' in data_type:
                self._load_4D_func(args)
            elif '4D_anat' in data_type:
                self._load_4D_anat(args)
            else:
                allowed = ['VBM', 'dual_reg', 'TBSS', 'Contrast', '4D_func',
                           '4D_anat']
                msg = "Data-type '%s' not recognized; please use one of the " \
                      "following: %r" % (data_type, allowed)
                raise KeyError(msg)

        if self.remove_zeros:
            self._remove_zeros()

        self.X = np.concatenate(self.X, axis=1)
        self.featureset_id = np.concatenate(self.featureset_id, axis=0)
        self.voxel_idx = np.concatenate(self.voxel_idx, axis=0)

        # 'Safety-check' to see whether everything corresponds properly
        assert(self.X.shape[1] == self.featureset_id.shape[0])
        all_vox = np.sum([self.voxel_idx[i].size for i in range(len(self.voxel_idx))])
        assert(self.X.shape[1] == all_vox)

        print("Final size of array: %r" % list(self.X.shape))

    def _remove_zeros(self):

        # If remove_zeros, all columns with all zeros are remove to reduce space
        if self.remove_zeros:
            indices = [np.invert(x == 0).all(axis=0) for x in self.X]

            for i, index in enumerate(indices):
                # Also other attributes are adapted to new shape
                self.X[i] = self.X[i][:, index]
                self.voxel_idx[i] = self.voxel_idx[i][index]
                self.featureset_id[i] = self.featureset_id[i][index]

    def _read_behav_file(self, file_path, sep, index_col):

        df = pd.read_csv(file_path, sep=sep, index_col=index_col)
        df.index = [str(i) for i in df.index.tolist()]
        return df

    def regress_out_confounds(self, file_path, col_name, backend='numpy',
                              sep='\t', index_col=0):

        df = self._read_behav_file(file_path=file_path, sep=sep, index_col=index_col)
        confound = df.loc[df.index.isin(self.common_subjects), col_name]
        confound = np.array(confound)

        if confound.ndim == 1:
            confound = confound[:, np.newaxis]

        if backend == 'sklearn':

            lr = LinearRegression(normalize=True, fit_intercept=True)

        elif backend == 'numpy':
            confound = np.hstack((np.ones((confound.shape[0], 1)), confound))

        for i in range(self.X.shape[1]):

            if i % 10000 == 0:
                print('Processed %i / %i voxels' % (i, self.X.shape[1]))

            if backend == 'sklearn':

                lr.fit(confound, self.X[:, i])
                pred = lr.predict(confound)
                self.X[:, i] = self.X[:, i] - pred

            elif backend == 'numpy':

                b, _, _, _ = np.linalg.lstsq(confound, self.X[:, i])
                b = b[:, np.newaxis]
                self.X[:, i] -= np.squeeze(confound.dot(b))

    def add_outcome_var(self, file_path, col_name, sep='\t', index_col=0,
                        normalize=True, binarize=None, remove=None):
        """ Adds target-variabel from csv """

        # Assumes index corresponds to self.common_subjects
        df = self._read_behav_file(file_path=file_path, sep=sep,
                                   index_col=index_col)
        behav = df.loc[df.index.isin(self.common_subjects), col_name]

        if behav.empty:
            print('Couldnt find any data common to .common_subjects in ' \
                  ' the MvpBetween object!')
            return 0

        behav.index = check_zeropadding_and_sort(behav.index.tolist())
        self.y = np.array(behav)

        if remove is not None:
            self.y = self.y[self.y != remove]
            self.X[self.y != remove, :]

        if normalize:
            self.y = (self.y - self.y.mean()) / self.y.std()

        if binarize is None:
            return 0
        else:
            y = self.y

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

    def split(self, file_path, col_name, target, sep='\t', index_col=0):

        # Assumes index corresponds to self.common_subjects
        df = self._read_behav_file(file_path=file_path, sep=sep,
                                   index_col=index_col)
        behav = df.loc[df.index.isin(self.common_subjects), col_name]

        if behav.empty:
            print('Couldnt find any data common to .common_subjects in ' \
                  ' the MvpBetween object!')
            return 0

        behav.index = check_zeropadding_and_sort(behav.index.tolist())
        idx = np.array(behav) == target
        self.X = self.X[idx, :]
        self.common_subjects = [sub for i, sub in
                                enumerate(self.common_subjects) if idx[i]]

    def write_4D(self, path=None, name=None):
        # To do: method to write out 4D nifti(s)
        pass

    def _load_mask(self, mask, threshold):

        if isinstance(mask, list):
            msg = 'You can only pass one mask! To use custom masks for each ' \
                  'source entry, specify the mask-key in source.'
            raise ValueError(msg)

        self.mask_index = nib.load(mask).get_data() > threshold
        self.mask_shape = self.mask_index.shape
        self.mask_index = self.mask_index.ravel()

    def _load_4D_func(self, args):

        if args['atlas'] == 'aal':

            aal = fetch_atlas_aal()
            aal = {'file': aal.maps}
            atlas_filename = aal['file']
        else:
            harvard_oxford = fetch_atlas_harvard_oxford(
                'cort-maxprob-thr25-2mm')
            harvard_oxford = {'file': harvard_oxford.maps}
            atlas_filename = harvard_oxford['file']

        data = Parallel(n_jobs=args['n_cores'])(delayed(
             _parallelize_4D_func_loading)(f, atlas_filename, args['method'])
             for f in args['paths'])

        # Load last func for some meta-data
        data = np.concatenate(data, axis=0)

        func = nib.load(args['paths'][0])
        self.voxel_idx.append(np.arange(data.shape[1]))
        self.affine.append(func.affine)
        self.data_shape.append(func.shape)
        feature_ids = np.ones(data.shape[1], dtype=np.uint32) * len(self.X)
        self.featureset_id.append(feature_ids)
        self.X.append(data)

    def _load_4D_anat(self, args):

        tmp = nib.load(args['path'])
        data = tmp.get_data()

        voxel_idx = np.arange(np.prod(tmp.shape[:3]))

        if self.mask_shape == tmp.shape[:3]:
            data = data[self.mask_index.reshape(tmp.shape[:3])].T
            voxel_idx = voxel_idx[self.mask_index]
        else:
            data = data.reshape(-1, data.shape[-1]).T

        self.voxel_idx.append(voxel_idx)
        self.affine.append(tmp.affine)
        self.data_shape.append(tmp.shape[:3])
        feature_ids = np.ones(data.shape[1], dtype=np.uint32) * len(self.X)
        self.featureset_id.append(feature_ids)
        self.X.append(data)

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

        name = self.data_name[-1]
        self.data_name.pop()
        _ = [self.data_name.append(name + '_comp%i' % i) for i in final_comps]

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

            if '4D_anat' in data_type:
                continue

            args['paths'] = check_zeropadding_and_sort(glob(args['path']))

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

        if self.subject_list is not None:
            all_subjects.append(set(self.subject_list))

        self.common_subjects = list(set.intersection(*all_subjects))
        self.common_subjects = check_zeropadding_and_sort(self.common_subjects)

        print("Found a set of %i complete subjects for data-types: %r" % \
              (len(self.common_subjects), [key for key in self.source]))

        for data_type, args in self.source.iteritems():

            if '4D_anat' in data_type:
                continue

            args['paths'] = [p for p in args['paths']
                             if p.split(os.sep)[args['position']] in self.common_subjects]


def check_zeropadding_and_sort(lst):

    length = len(lst[0])
    zero_pad = all(len(i) == length for i in lst)

    if zero_pad:
        return sorted(lst)
    else:
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in
                                    re.split("([0-9]+)", key)]
        return sorted(lst, key=alphanum_key)

def _parallelize_4D_func_loading(f, atlas, method):
    print(f)
    func = nib.load(f)
    roi_masker = NiftiLabelsMasker(labels_img=atlas,
                                   standardize=True,
                                   resampling_target=None)

    time_series = roi_masker.fit_transform(func)

    if method == 'corr':
        conn = np.corrcoef(time_series.T)
    elif method == 'invcorr':
        graphlasso = GraphLassoCV()
        graphlasso.fit(time_series)
        conn = graphlasso.precision_
    else:
        raise ValueError('Specify either corr or invcorr')

    conn = conn[np.tril_indices(conn.shape[0], k=-1)].ravel()
    return(conn[np.newaxis, :])

if __name__ == '__main__':

    source = {}
    source['4D_func'] = {'path': '/media/lukas/piop/PIOP/PIOP_PREPROC_MRI/pi0*/*rs*/*mni.nii.gz',
                         'atlas': 'ho',
                         'method': 'corr',
                         'n_cores': 1}
    mvp = MvpBetween(source=source, subject_idf='pi????', remove_zeros=True)
    mvp.create()
    mvp.write('/home/lukas', name='harvard-oxford-conn')

