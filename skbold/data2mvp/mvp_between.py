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
    """
    Extracts and stores multivoxel pattern information across subjects.
    The MvpBetween class allows for the extraction and storage of multivoxel
    (MRI) pattern information across subjects. The MvpBetween class can handle
    various types of information, including functional contrasts, 3D
    (subject-specific) and 4D (subjects stacked) VBM and TBSS data,
    dual-regression data, and functional-connectivity data from resting-state
    scans (experimental).

    Parameters
    ----------
    source : dict
        Dictionary with types of data as keys and data-specific dictionaries
        as values. Keys can be 'Contrast_*' (indicating a 3D functional contrast),
        '4D_anat' (for 4D anatomical - VBM/TBSS - files), 'VBM', 'TBSS',
        and 'dual_reg' (a subject-spedific 4D file with components as fourth
        dimension). The dictionary passed as values must include, for each
        data-type, a path with wildcards to the corresponding (subject-specific)
        data-file. Other, optional, key-value pairs per data-type can be assigned,
        including 'mask': 'path', to use data-type-specific masks.

        An example:

        >>> source = {}
        >>> source['Contrast_emo'] = {'path': '~/data/sub0*/*.feat/stats/tstat1.nii.gz'}
        >>> vbm_mask = '~/vbm_mask.nii.gz'
        >>> source['VBM'] = {'path': '~/data/sub0*/*vbm.nii.gz', 'mask': vbm_mask}

    subject_idf : str
        Subject-identifier. This identifier is used to extract subject-names
        from the globbed directories in the 'path' keys in source, so that
        it is known which pattern belongs to which subject. This way,
        MvpBetween can check which subjects contain complete data!
    X : ndarray
        Not necessary to pass MvpWithin, but needs to be defined as it is
        needed in the super-constructor.
    y : ndarray or list
        Labels or targets corresponding to the samples in ``X``.
    mask : str
        Absolute path to nifti-file that will be used as a common mask.
        Note: this will only be applied if its shape corresponds to the
        to-be-indexed data. Otherwise, no mask is applied. Also, this mask
        is 'overridden' if source[data_type] contains a 'mask' key, which
        implies that this particular data-type has a custom mask.
    mask_threshold : int or float
        Minimum value to binarize the mask when it's probabilistic.

    Attributes
    ----------
    mask_shape : tuple
        Shape of mask that patterns will be indexed with.
    nifti_header : list of Nifti1Header objects
        Nifti-headers from original data-types.
    affine : list of ndarray
        Affines corresponding to nifti-masks of each data-type.
    X : ndarray
        The actual patterns (2D: samples X features)
    y : list or ndarray
        Array/list with labels/targets corresponding to samples in X.
    common_subjects : list
        List of subject-names that have complete data specified in source.
    featureset_id : ndarray
        Array with integers of size X.shape[1] (i.e. the amount of features in
        X). Each unique integer, starting at 0, refers to a different feature-set.
    voxel_idx : ndarray
        Array with integers of size X.shape[1]. Per feature-set, these voxel-
        indices allow the features to be mapped back to whole-brain space.
        For example, to map back the features in X from feature set 1 to MNI152
        (2mm) space, do:

        >>> mni_vol = np.zeros((91, 109, 91))
        >>> tmp_idx = mvp.featureset_id == 0
        >>> mni_vol[mvp.featureset_id[tmp_idx] = mvp.X[0, tmp_idx]

    data_shape : list of tuples
        Original (whole-brain) shape of the loaded data, per data-type.
    data_name : list of str
        List of names of data-types.
    """
    def __init__(self, source, subject_idf='sub0???', remove_zeros=True,
                 X=None, y=None, mask=None, mask_threshold=0, subject_list=None):

        super(MvpBetween, self).__init__(X=X, y=y, mask=mask,
                                         mask_threshold=mask_threshold)

        self.source = source
        self.remove_zeros = remove_zeros
        self.subject_idf = subject_idf
        self.subject_list = subject_list
        self.ref_space = 'mni'
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
        """ Extracts and stores data as specified in source.

        Raises
        ------
        ValueError
            If data-type is not one of ['VBM', 'TBSS', '4D_anat*', 'dual_reg', 'Contrast*']
        """
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
        """ Regresses out a confounding variable from X.

        Parameters
        ----------
        file_path : str
            Absolute path to spreadsheet-like file including the confounding variable.
        col_name : str
            Column name in spreadsheet containing the confouding variable
        backend : str
            Which algorithm to use to regress out the confound. The option 'numpy'
            uses np.linalg.lstsq() and 'sklearn' uses the LinearRegression estimator.
        sep : str
            Separator to parse the spreadsheet-like file.
        index_col : int
            Which column to use as index (should correspond to subject-name).
        """
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
        """ Sets ``y`` attribute to an outcome-variable (target).

        Parameters
        ----------
        file_path : str
            Absolute path to spreadsheet-like file including the outcome variable.
        col_name : str
            Column name in spreadsheet containing the outcome variable
        sep : str
            Separator to parse the spreadsheet-like file.
        index_col : int
            Which column to use as index (should correspond to subject-name).
        normalize : bool
            Whether to normalize (0 mean, unit std) the outcome variable.
        binarize : dict
            If not None, the outcome variable will be binarized along the
            key-value pairs in the binarize-argument. Options:

            >>> binarize = {'type': 'percentile', 'high': .75, 'low': .25} # binarizes along percentiles
            >>> binarize = {'type': 'zscore', 'std': 1} # binarizes according to x stdevs above and below mean
            >>> binarize = {'type': 'constant', 'cutoff': 10} # binarizes according to above and below constant
            >>> binarize = {'type': 'median'} # binarizes according to median-split.
        remove : int or float or str
            Removes instances in which y == remove from MvpBetween object.
        """

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
            self.common_subjects = [sub for i, sub in self.common_subjects
                                    if (self.y != remove)[i]]

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
        """ Splits an MvpBetween object based on some external index.

        Parameters
        ----------
        file_path : str
            Absolute path to spreadsheet-like file including the outcome variable.
        col_name : str
            Column name in spreadsheet containing the outcome variable
        target : str or int or float
            Target to which the data in col_name needs to be compared to, in
            order to create an index.
        sep : str
            Separator to parse the spreadsheet-like file.
        index_col : int
            Which column to use as index (should correspond to subject-name).
        """

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

    def write_4D(self, path=None):
        """ Writes a 4D nifti (subs = 4th dimension) of X.

        Parameters
        ----------
        path : str
            Absolute path to save nifti to.
        name : str
            Name to be given to nifti-file.
        """

        if path is None:
            path = os.getcwd()

        for i, fid in enumerate(np.unique(self.featureset_id)):
            s = self.data_shape[i]
            to_write = np.zeros((np.prod(s), self.X.shape[0]))
            X_to_write = self.X[:, self.featureset_id == fid]
            to_write[self.voxel_idx[self.featureset_id == fid]] = X_to_write.T
            to_write = to_write.reshape((s[0], s[1], s[2], to_write.shape[-1]))
            img = nib.Nifti1Image(to_write, self.affine[i])
            img.to_filename(op.join(path, self.data_name[i]) + '.nii.gz')

        if self.y is not None:
            np.savetxt(op.join(path, 'y_4D_nifti.txt'), self.y,
                       fmt='%1.4f', delimiter='\t')

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
