from __future__ import division, print_function, absolute_import

import os
import re
import json
import warnings
import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.stats as stat
from glob import glob
from fnmatch import fnmatch
from .mvp import Mvp
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from nilearn.decoding import SearchLight
except ImportError as e:
    print("Skbold's searchlight functionality not available.")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from ..preproc import MajorityUndersampler


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
        as values. Keys can be 'Contrast_*' (indicating a 3D functional
        contrast), '4D_anat' (for 4D anatomical - VBM/TBSS - files), 'VBM',
        'TBSS', and 'dual_reg' (a subject-spedific 4D file with components as
        fourth dimension).

        The dictionary passed as values must include, for each
        data-type, a path with wildcards to the corresponding
        (subject-specific) data-file. Other, optional, key-value pairs per
        data-type can be assigned, including 'mask': 'path', to use
        data-type-specific masks.

        An example:

        >>> source = {}
        >>> path_emo = '~/data/sub0*/*.feat/stats/tstat1.nii.gz'
        >>> source['Contrast_emo'] = {'path': path_emo}
        >>> vbm_mask = '~/vbm_mask.nii.gz'
        >>> path_vbm = '~/data/sub0*/*vbm.nii.gz'
        >>> source['VBM'] = {'path': path_vbm, 'mask': vbm_mask}

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
        X). Each unique integer, starting at 0, refers to a different
        feature-set.
    voxel_idx : ndarray
        Array with integers of size X.shape[1]. Per feature-set, these voxel-
        indices allow the features to be mapped back to whole-brain space.
        For example, to map back the features in X from feature set 1 to MNI152
        (2mm) space, do:

        >>> mni_vol = np.zeros((91, 109, 91))
        >>> tmp_idx = mvp.featureset_id == 0
        >>> mni_vol[mvp.featureset_id[tmp_idx]] = mvp.X[0, tmp_idx]

    data_shape : list of tuples
        Original (whole-brain) shape of the loaded data, per data-type.
    data_name : list of str
        List of names of data-types.
    """
    def __init__(self, source, subject_idf='sub0???', remove_zeros=True,
                 X=None, y=None, mask=None, mask_thres=0,
                 subject_list=None):

        super(MvpBetween, self).__init__(X=X, y=y, mask=mask,
                                         mask_thres=mask_thres)

        self.source = source
        self.remove_zeros = remove_zeros
        self.subject_idf = subject_idf
        self.subject_list = subject_list
        self.ref_space = 'mni'
        self.common_subjects = None
        self.y = None
        self.X = []
        self.fs_masks = []  # featureset-specific masks
        self.featureset_id = []
        self.affine = []  # This could be an array
        self.nifti_header = []
        self.voxel_idx = []
        self.data_shape = []  # This could be an array
        self.data_name = []
        self.binarize_params = None

        if not isinstance(source, dict):
            msg = "Source must be a dictionary with type (e.g. 'VBM') " \
                  "path ('path/to/VBM_file.nii.gz) mappings!"
            raise TypeError(msg)

    def create(self):
        """ Extracts and stores data as specified in source.

        Raises
        ------
        ValueError
            If data-type is not one of ['VBM', 'TBSS', '4D_anat*', 'dual_reg',
            'Contrast*']
        """
        # Globs all paths and checks for which subs there is complete data
        self._check_complete_data()

        # Loop over data-types as defined in source
        for data_type, args in self.source.items():

            print('Processing: %s ...' % data_type)
            self.data_name.append(data_type)

            if 'mask' in args.keys():
                maskl = nib.load(args['mask'])
                if 'mask_threshold' in args.keys():
                    th = args['mask_threshold']
                else:
                    th = 0

                self.fs_masks.append({'path': args['mask'],
                                      'threshold': th,
                                      'affine': maskl.affine,
                                      'shape': maskl.shape})
            else:
                self.fs_masks.append(self.common_mask)

            TYPES_3D = ['VBM', 'TBSS', 'Contrast*']
            if any(fnmatch(data_type, typ) for typ in TYPES_3D):
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
        all_vox = np.sum([self.voxel_idx[i].size
                          for i in range(len(self.voxel_idx))])
        assert(self.X.shape[1] == all_vox)

        print("Final size of array: %r" % list(self.X.shape))

    def _remove_zeros(self):

        # If remove_zeros, all columns with all zeros are removed to space
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

    def calculate_confound_weighting(self, file_path, col_name, sep='\t',
                                     index_col=0, estimator=None):
        """ Calculates inverse probability weighting for confounds.

        Note: should be moved to mvp-core

        Parameters
        ----------
        file_path : str
            Absolute path to spreadsheet-like file including the confounding
            variable.
        col_name : str
            Column name in spreadsheet containing the confouding variable
        sep : str
            Separator to parse the spreadsheet-like file.
        index_col : int
            Which column to use as index (should correspond to subject-name).
        estimator : scikit-learn estimator
            Estimator used to calculate p(y=1 | confound-array)

        Returns
        -------
        ipw : array
            Array with inverse probability weights for the samples, based on
            the confounds indicated by col_name.

        References
        ----------
        Linn, K.A., Gaonkar, B., Doshi, J., Davatzikos, C., & Shinohara, R.
        (2016). Addressing confounding in predictive models with an application
        to neuroimaging. Int. J. Biostat., 12(1): 31-44.

        Code adapted from https://github.com/kalinn/IPW-SVM.

        """

        if estimator is None:
            estimator = LogisticRegression()

        df = self._read_behav_file(file_path=file_path, sep=sep,
                                   index_col=index_col)

        df.index = check_zeropadding_and_sort(df.index.tolist())
        common_idx = df.index.isin(self.common_subjects)
        confound = np.array(df.loc[common_idx, col_name])

        # Fit confounds to y
        estimator.fit(confound, self.y)

        # Calculate p(y=1 | confounds)
        prob = estimator.predict_proba(confound)

        # Inverse of prob = inverse probability weighting (IPW) factor
        self.ipw = 1.0/prob
        return self.ipw

    def regress_out_confounds(self, file_path, col_name, backend='numpy',
                              sep='\t', index_col=0):
        """ Regresses out a confounding variable from X.

        Parameters
        ----------
        file_path : str
            Absolute path to spreadsheet-like file including the confounding
            variable.
        col_name : str
            Column name in spreadsheet containing the confouding variable
        backend : str
            Which algorithm to use to regress out the confound. The option
            'numpy' uses np.linalg.lstsq() and 'sklearn' uses the
            LinearRegression estimator.
        sep : str
            Separator to parse the spreadsheet-like file.
        index_col : int
            Which column to use as index (should correspond to subject-name).
        """
        df = self._read_behav_file(file_path=file_path, sep=sep,
                                   index_col=index_col)
        df.index = check_zeropadding_and_sort(df.index.tolist())
        common_idx = df.index.isin(self.common_subjects)
        confound = df.loc[common_idx, col_name]

        confound = np.array(confound)

        # normalize, just to be sure
        confound = StandardScaler().fit_transform(confound)

        if confound.ndim == 1:
            confound = confound[:, np.newaxis]

        if backend == 'sklearn':

            lr = LinearRegression(normalize=True, fit_intercept=True)

        elif backend == 'numpy':
            confound = np.hstack((np.ones((confound.shape[0], 1)), confound))

        if confound.shape[0] != self.X.shape[0]:
            msg = "Size of confound (%r) is not the same as X (%r)" % \
                  (confound.shape[0], self.X.shape[0])
            raise ValueError(msg)

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

    def update_sample(self, idx):
        """ Updates the data matrix and associated attributes."""

        self._update_common_subjects(idx)
        self.X = self.X[idx, :]

        if self.y is not None:
            self.y = self.y[idx]

    def _undersample_majority(self):

        if len(np.unique(self.y)) > 5:
            msg = ("Found >5 classes and attempting to perform majority "
                   "undersampling - are you sure the data is categorical?")
            warnings.warn(msg)

        self.y = LabelEncoder().fit(self.y).transform(self.y)
        mus = MajorityUndersampler(verbose=True)
        self.X, self.y = mus.fit().transform(self.X, self.y)
        self._update_common_subjects(mus.idx_)

    def _update_common_subjects(self, idx):
        """ Updates common_subjects after indexing. """
        self.common_subjects = [sub for i, sub in
                                enumerate(self.common_subjects) if idx[i]]

    def add_y(self, file_path, col_name, sep='\t', index_col=0,
              normalize=False, binarize=None, remove=None,
              save_binarization_params=None,
              apply_binarization_params=None,
              ensure_balanced=False):
        """ Sets ``y`` attribute to an outcome-variable (target).

        Parameters
        ----------
        file_path : str
            Absolute path to spreadsheet-like file including the outcome var.
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

            >>> binarize = {'type': 'percentile', 'high': .75, 'low': .25}
            >>> binarize = {'type': 'zscore', 'std': 1}
            >>> binarize = {'type': 'constant', 'cutoff': 10}
            >>> binarize = {'type': 'median'}

        save_binarization_params : str
            If not none, it refers to the path to save the binarization params
            to.
        apply_binarization_params : str
            If not none, it refers to the path to load the binarization params
            from and apply them to the loaded target variable.
        remove : int or float or str
            Removes instances in which y == remove from MvpBetween object.
        ensure_balanced : bool
            Whether to ensure balanced classes (if True, done by undersampling
            the majority class).
        """

        # Assumes index corresponds to self.common_subjects
        df = self._read_behav_file(file_path=file_path, sep=sep,
                                   index_col=index_col)

        df.index = check_zeropadding_and_sort(df.index.tolist())
        common_idx = df.index.isin(self.common_subjects)
        behav = df.loc[common_idx, col_name]

        if behav.empty:
            msg = ("Couldnt find any data common to .common_subjects in the "
                   "MvpBetween object!")
            raise ValueError(msg)

        self.y = np.array(behav)
        if remove is not None:
            idx = self.y != remove
            self.y = self.y[idx]
            self.X = self.X[self.y != remove, :]
            self._update_common_subjects(idx)

        if normalize:
            self.y = (self.y - self.y.mean()) / self.y.std()

        if apply_binarization_params is not None:

            with open(apply_binarization_params) as fin:
                params = json.load(fin)

            if params['type'] == 'zscore':
                y_norm = (self.y - params['mean']) / params['std']
                idx = np.abs(y_norm) > params['n_std']
                y = (y_norm[idx] > 0).astype(int)
            else:
                msg = ("Apply binarization params other than 'zscore is "
                       "not yet implemented.")
                raise ValueError(msg)

            self.y = y

            if idx is not None:
                self._update_common_subjects(idx)
                self.X = self.X[idx, :]

            if ensure_balanced:
                self._undersample_majority()

            return 0

        if binarize is None:

            if ensure_balanced:
                self._undersample_majority()
            return 0
        else:
            y = self.y

        if binarize['type'] == 'percentile':
            y_rank = [stat.percentileofscore(y, a, 'rank') for a in y]
            y_rank = np.array(y_rank)
            idx = (y_rank < binarize['low']) | (y_rank > binarize['high'])
            low = stat.scoreatpercentile(y, binarize['low'])
            high = stat.scoreatpercentile(y, binarize['high'])
            self.binarize_params = {'type': 'percentile',
                                    'low': low,
                                    'high': high}
            y = (y_rank[idx] > 50).astype(int)

        elif binarize['type'] == 'zscore':
            y_norm = (y - y.mean()) / y.std()  # just to be sure
            idx = np.abs(y_norm) > binarize['std']
            self.binarize_params = {'type': binarize['type'],
                                    'mean': y.mean(),
                                    'std': y.std(),
                                    'n_std': binarize['std']}
            y = (y_norm[idx] > 0).astype(int)

        elif binarize['type'] == 'constant':
            y = (y > binarize['cutoff']).astype(int)
            idx = None
            self.binarize_params = {'type': binarize['type'],
                                    'cutoff': binarize['cutoff']}
        elif binarize['type'] == 'median':  # median-split
            median = np.median(y)
            y = (y > median).astype(int)
            idx = None
            self.binarize_params = {'type': binarize['type'],
                                    'median': median}

        if save_binarization_params is not None:

            with open(op.join(save_binarization_params,
                              'binarization_params.json'), 'w') as fout:
                json.dump(self.binarize_params, fout)

        self.y = y

        if idx is not None:
            self._update_common_subjects(idx)
            self.X = self.X[idx, :]

        if ensure_balanced:
            self._undersample_majority()

    def split(self, file_path, col_name, target, sep='\t', index_col=0):
        """ Splits an MvpBetween object based on some external index.

        Parameters
        ----------
        file_path : str
            Absolute path to spreadsheet-like file including the outcome var.
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

        common_idx = df.index.isin(self.common_subjects)
        behav = df.loc[common_idx, col_name]

        if behav.empty:
            print('Couldnt find any data common to .common_subjects in '
                  ' the MvpBetween object!')
            return 0

        behav.index = check_zeropadding_and_sort(behav.index.tolist())
        idx = np.array(behav) == target

        if idx.sum() == 0:
            raise ValueError('Found 0 subjects for split with target: %s' %
                             str(target))
        else:
            print("Splitting mvp with target '%s', found %i subjects." %
                  (str(target), idx.sum()))

        self.X = self.X[idx, :]
        if self.y is not None:
            if len(self.y) > self.X.shape[0]:
                self.y = self.y[idx]

                self.common_subjects = [sub for i, sub in
                                        enumerate(self.common_subjects)
                                        if idx[i]]

    def run_searchlight(self, out_dir, name='sl_results', n_folds=10, radius=5,
                        mask=None, estimator=None, **kwargs):
        """ Runs a searchlight on the mvp object.

        Parameters
        ----------
        out_dir : str
            Path to which to save the searchlight results
        name : str
            Name for the searchlight-results-file (nifti)
        n_folds : int
            The amount of folds in sklearn's StratifiedKFold.
        radius : int/list
            Radius for the searchlight. If list, it iterates over radii.
        mask : str
            Path to mask to apply to mvp. If nothing is listed, it will use
            the masks applied when the mvp was created.
        estimator : sklearn estimator or pipeline
            Estimator to use in the classification process.
        **kwargs : misc
            Other arguments for initializing nilearn's searchlight object (see
           nilearn.github.io/decoding/searchlight.html).
        """

        # to do: implement import searchlight here (so skbold does not
        # necessarily depend on nilearn

        # NOT YET TESTED

        # Import OLD version
        from sklearn.cross_validation import StratifiedKFold
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from nilearn.decoding import SearchLight

        nimgs = self.write_4D(return_nimg=True)
        cv = StratifiedKFold(self.y, n_folds=10)

        if estimator is None:
            estimator = Pipeline([('scaler', StandardScaler()),
                                  ('svm', SVC(kernel='linear', C=1))])

        if isinstance(radius, (int, float)):
            radius = [radius]

        for i, nimg in enumerate(nimgs):

            if mask is None:
                mask_tmp = self.fs_masks[i]
                mask_bool = mask_tmp['idx'].astype(int)

            for r in radius:
                sl = SearchLight(mask_img=mask_bool, radius=r, n_jobs=-1,
                                 cv=cv, estimator=estimator,
                                 scoring='accuracy', **kwargs)
                sl.fit(nimg, y=self.y)
                sl_nifti = nib.Nifti1Image(sl.scores_, nimg.affine)

                nib.save(sl_nifti, op.join(out_dir, name + '_%imm.nii.gz' % r))

    def write_4D(self, path=None, return_nimg=False):
        """ Writes a 4D nifti (subs = 4th dimension) of X.

        Parameters
        ----------
        path : str
            Absolute path to save nifti to.
        return_nimg : bool
            Whether to actually return the Nifti1-image object.
        """

        if path is None:
            path = os.getcwd()

        fids = np.unique(self.featureset_id)

        nimgs = []
        for i, fid in enumerate(fids):
            pos_idx = np.where(i == fids)[0]
            s = self.data_shape[pos_idx]
            to_write = np.zeros((np.prod(s), self.X.shape[0]))
            X_to_write = self.X[:, self.featureset_id == fid]
            to_write[self.voxel_idx[self.featureset_id == fid]] = X_to_write.T
            to_write = to_write.reshape((s[0], s[1], s[2], to_write.shape[-1]))
            img = nib.Nifti1Image(to_write, self.affine[pos_idx])
            nimgs.append(img)

            if not return_nimg:
                img.to_filename(op.join(path,
                                        self.data_name[pos_idx]) + '.nii.gz')

        if return_nimg:
            if len(nimgs) == 1:
                return nimgs[0]
            else:
                return nimgs

        if self.y is not None:
            np.savetxt(op.join(path, 'y_4D_nifti.txt'), self.y,
                       fmt='%1.4f', delimiter='\t')

    def _load_4D_anat(self, args):

        # some checks
        tmp = nib.load(args['path'])

        if len(args['subjects']) != tmp.shape[-1]:
            msg = ("For 4D_anat, length of 'subjects' (%i) is different from "
                   "amount of vols in nifti (%i)." %
                   (len(args['subjects']), tmp.shape[-1]))
            raise ValueError(msg)

        args['subjects'] = check_zeropadding_and_sort(args['subjects'])
        idx = np.array([True if sub in self.common_subjects else
                        False for sub in args['subjects']])

        if tmp.shape[-1] > 500 and 'TBSS' in self.data_name[-1]:
            print('Loading TBSS data in two steps ...')
            # probably too large to load at once
            # cannot use idx, because nifti-slicing doesn't support that
            n_half = np.round(tmp.shape[-1] / 2.0).astype(int)
            data1 = tmp.dataobj[..., :n_half]
            data1 = data1[:, :, :, idx[:n_half]]
            data2 = tmp.dataobj[..., n_half:]
            data2 = data2[:, :, :, idx[n_half:]]
            data = np.concatenate((data1, data2), axis=-1)
        else:
            data = tmp.get_data()
            data = data[:, :, :, idx]

        voxel_idx = np.arange(np.prod(data.shape[:3]))

        tmp_mask = self.fs_masks[-1]
        if tmp_mask['shape'] == data.shape[:3]:
            data = data[tmp_mask['idx'].reshape(tmp.shape[:3])].T
            voxel_idx = voxel_idx[tmp_mask['idx']]
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

                tmp_mask = self.fs_masks[-1]
                if tmp_mask['shape'] == tmp_shape:
                    vol = vol[tmp_mask['idx']]

                vol = vol[np.newaxis, :]
                vols.append(vol)
            data.append(vols)

        # Hack to infer attributes by looking at last volume
        voxel_idx = np.arange(np.prod(tmp_shape))

        tmp_mask = self.fs_masks[-1]
        if tmp_mask['shape'] == tmp_shape:
            voxel_idx = voxel_idx[tmp_mask['idx']]

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

            tmp_mask = self.fs_masks[-1]
            if tmp_mask['shape'] == tmp.shape:
                tmp_data = tmp_data[tmp_mask['idx']]

            data.append(tmp_data[np.newaxis, :])

        voxel_idx = np.arange(np.prod(tmp.shape))

        if tmp_mask['shape'] == tmp.shape:
            voxel_idx = voxel_idx[tmp_mask['idx']]

        self.voxel_idx.append(voxel_idx)
        self.affine.append(tmp.affine)
        self.data_shape.append(tmp.shape)

        data = np.concatenate(data, axis=0)
        feature_ids = np.ones(data.shape[1], dtype=np.uint32) * len(self.X)
        self.featureset_id.append(feature_ids)
        self.X.append(data)

    def _check_complete_data(self):

        for data_type, args in self.source.items():

            if '4D_anat' in data_type:
                continue

            args['paths'] = check_zeropadding_and_sort(glob(args['path']))

            ex_path = args['paths'][0].split(os.sep)
            idx = [True if fnmatch(p, self.subject_idf) else False
                   for p in ex_path]
            position = list(np.arange(len(ex_path))[np.array(idx)])

            if len(position) > 1:
                msg = ("Couldn't resolve to which subject the file '%s' "
                       "belongs because subject-idf (%s) is ambiguous!" %
                       (ex_path, self.subject_idf))
                raise ValueError(msg)
            elif len(position) == 1:
                position = position[0]
            else:
                msg = ("Couldn't determine which subject belongs to which path"
                       "for data-type = '%s'" % data_type)
                raise ValueError(msg)

            args['position'] = position
            args['subjects'] = [p.split(os.sep)[position]
                                for p in args['paths']]

        all_subjects = [set(args['subjects'])
                        for args in self.source.values()]

        if self.subject_list is not None:
            all_subjects.append(set(self.subject_list))

        self.common_subjects = list(set.intersection(*all_subjects))
        self.common_subjects = check_zeropadding_and_sort(self.common_subjects)

        print("Found a set of %i complete subjects for data-types: %r" %
              (len(self.common_subjects), [key for key in self.source]))

        for data_type, args in self.source.items():

            if '4D_anat' in data_type:
                continue

            args['paths'] = [p for p in args['paths']
                             if p.split(os.sep)[args['position']]
                             in self.common_subjects]


def _check_if_number(text):

    if text.isdigit():
        return int(text)
    else:
        return text.lower()


def _alphanum_key(key):

    return [_check_if_number(c) for c in re.split("([0-9]+)", key)]


def check_zeropadding_and_sort(lst):

    length = len(lst[0])
    zero_pad = all(len(i) == length for i in lst)

    if zero_pad:

        return sorted(lst)
    else:

        return sorted(lst, key=_alphanum_key)
