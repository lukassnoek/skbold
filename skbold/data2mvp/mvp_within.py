import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
from skbold.core import Mvp, convert2epi, convert2mni
from skbold.utils import sort_numbered_list
from sklearn.preprocessing import LabelEncoder
from glob import glob


class MvpWithin(Mvp):

    def __init__(self, source, read_labels=True, remove_contrast=[],
                 invert_selection=None, ref_space='epi', beta2tstat=True,
                 remove_zeros=True, X=None, y=None, mask=None,
                 mask_threshold=0):

        super(MvpWithin, self).__init__(X=X, y=y, mask=mask,
                                        mask_threshold=mask_threshold)

        self.source = source
        self.read_labels = read_labels
        self.ref_space = ref_space
        self.beta2tstat = beta2tstat
        self.invert_selection = invert_selection
        self.remove_zeros = remove_zeros
        self.remove_contrast = remove_contrast
        self.remove_idx = None
        self.y = []
        self.contrast_labels = []
        self.X = []

    def create(self):

        if isinstance(self.source, str):
            self.source = [self.source]

        # Loop over sources
        for src in self.source:

            if '.feat' in src:
                self._load_fsl(src)
            else:
                msg = "Loading 'within-data' from other sources than " \
                      "FSL-feat directories is not yet implemented!"
                print(msg)

        self.X = np.concatenate(self.X, axis=0)

        if self.read_labels:
            self.y = LabelEncoder().fit_transform(self.contrast_labels)

        if self.remove_zeros:
            idx = np.invert((self.X == 0)).all(axis=0)
            self.X = self.X[:, idx]
            self.voxel_idx = self.voxel_idx[idx]

    def _load_fsl(self, src):

        if not op.isdir(src):
            msg = "The feat-directory '%s' doesn't seem to exist." % src
            raise ValueError(msg)

        if self.read_labels:
            design_file = op.join(src, 'design.con')
            contrast_labels_current = self._extract_labels(design_file=design_file)
            self.contrast_labels.extend(contrast_labels_current)

        if self.mask is not None:

            if self.ref_space == 'epi':
                reg_dir = op.join(src, 'reg')
                self.mask = convert2epi(self.mask, reg_dir, reg_dir)

                if self.voxel_idx is None:
                    self._update_mask_info(self.mask)

        if self.ref_space == 'epi':
            stat_dir = op.join(src, 'stats')
        elif self.ref_space == 'mni':
            stat_dir = op.join(src, 'reg_standard')
        else:
            raise ValueError('Specify valid reference-space (ref_space)')

        if self.ref_space == 'mni' and not op.isdir(stat_dir):
            stat_dir = op.join(src, 'stats')
            transform2mni = True
        else:
            transform2mni = False

        copes = sort_numbered_list(glob(op.join(stat_dir, 'cope*.gz')))
        varcopes = sort_numbered_list(glob(op.join(stat_dir, 'varcope*.gz')))

        # Transform (var)copes if ref_space is 'mni' but files are in 'epi'.
        if transform2mni:
            copes.extend(varcopes)
            out_dir = op.join(src, 'reg_standard')
            transformed_files = convert2mni(copes, reg_dir, out_dir)
            half = int(len(transformed_files) / 2)
            copes = transformed_files[:half]
            varcopes = transformed_files[half:]

        _ = [copes.pop(idx) for idx in sorted(self.remove_idx, reverse=True)]
        _ = [varcopes.pop(ix) for ix in sorted(self.remove_idx, reverse=True)]

        n_stat = len(copes)

        if not n_stat == len(contrast_labels_current):
            msg = 'The number of trials (%i) do not match the number of ' \
                  'class labels (%i)' % (n_stat, len(self.contrast_labels))
            raise ValueError(msg)

        if self.mask is None: # set attributes if no mask was given
            tmp = nib.load(copes[0])
            n_features = np.prod(tmp.shape)
            self.affine = tmp.affine
            self.nifti_header = tmp.header
            self.mask_shape = tmp.shape
            self.voxel_idx = np.arange(np.prod(tmp.shape))

        # Pre-allocate

        mvp_data = np.zeros((n_stat, self.voxel_idx.size))

        # Load in data (COPEs)
        for i, path in enumerate(copes):
            cope_img = nib.load(path)
            mvp_data[i, :] = cope_img.get_data().ravel()[self.voxel_idx]

        if self.beta2tstat:
            for i, varcope in enumerate(varcopes):
                var = nib.load(varcope).get_data()
                var_sq = np.sqrt(var.ravel()[self.voxel_idx])
                mvp_data[i, :] = np.divide(mvp_data[i, :], var_sq)

        mvp_data[np.isnan(mvp_data)] = 0
        self.X.append(mvp_data)

    def _read_design(self, design_file):

        if not op.isfile(design_file):
            raise IOError('There is no design.con file for %s' % design_file)

        # Find number of contrasts and read in accordingly
        contrasts = sum(1 if 'ContrastName' in line else 0
                        for line in open(design_file))

        n_lines = sum(1 for line in open(design_file))

        df = pd.read_csv(design_file, delimiter='\t', header=None,
                         skipfooter=n_lines - contrasts, engine='python')

        cope_labels = list(df[1].str.strip())  # remove spaces

        # Here, numeric extensions of labels (e.g. 'positive_003') are removed
        labels = []
        for c in cope_labels:
            parts = [x.strip() for x in c.split('_')]
            if parts[-1].isdigit():
                label = '_'.join(parts[:-1])
                labels.append(label)
            else:
                labels.append(c)

        return labels

    def _extract_labels(self, design_file):

        cope_labels = self._read_design(design_file)

        if isinstance(self.remove_contrast, str):
            self.remove_contrast = [self.remove_contrast]
        remove_contrast = self.remove_contrast

        if remove_contrast is None:
            self.remove_idx = []
            return cope_labels

        # Remove to-be-ignored contrasts (e.g. cues)
        remove_idx = np.zeros((len(cope_labels), len(remove_contrast)))

        for i, name in enumerate(remove_contrast):
            remove_idx[:, i] = np.array([name in lab for lab in cope_labels])

        self.remove_idx = np.where(remove_idx.sum(axis=1).astype(int))[0]

        if self.invert_selection:
            indices = np.arange(len(cope_labels))
            self.remove_idx = [x for x in indices if not x in self.remove_idx]

        _ = [cope_labels.pop(idx) for idx in np.sort(self.remove_idx)[::-1]]

        return cope_labels


if __name__ == '__main__':

    testdir = ['/home/lukas/pi0042/pi0042-20150706-0005-WIPpiopharriri.feat',
               '/home/lukas/pi0042/pi0042-20150706-0008-WIPpiopwm.feat']

    mask = '/home/lukas/pi0042/GrayMatter.nii.gz'

    mw = MvpWithin(source=testdir, read_labels=True, remove_contrast='emotion',
                   remove_zeros=True,
                   invert_selection=None, ref_space='epi', mask=mask,
                   mask_threshold=0)

    mw.create()
    mw.write(path='/home/lukas', name='within', backend='joblib')