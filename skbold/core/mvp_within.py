import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
from skbold.core import Mvp, convert2epi, convert2mni
from skbold.utils import sort_numbered_list
from sklearn.preprocessing import LabelEncoder
from glob import glob


class MvpWithin(Mvp):
    """
    Extracts and stores subject-specific single-trial multivoxel-patterns
    The MvpWithin class allows for the extraction of subject-specific
    single-trial, multivoxel fMRI patterns from a FSL feat-directory.

    Parameters
    ----------
    source : str
        An absolute path to a subject-specific first-level FEAT directory.
    read_labels : bool
        Whether to read the labels/targets (i.e. ``y``) from the contrast
        names defined in the design.con file.
    remove_contrast : list
        Given that all contrasts (COPEs) are loaded from the FEAT-directory,
        this argument can be used to remove irrelevant contrasts (e.g.
        contrasts of nuisance predictors). Entries in remove_contrast do
        not have to literal; they may be a substring of the full name of the
        contrast.
    invert_selection : bool
        Sometimes, instead of loading in all contrasts and excluding some,
        you might want to load only a single or a couple contrasts, and
        exclude all other. By setting invert_selection to True, it treats
        the remove_contrast variable as a list of contrasts to include.
    ref_space : str
        Indicates in which 'space' the patterns will be stored. The default
        is 'epi', indicating that the patterns will be loaded and stored
        in subject-specific (native) functional space. The other option is
        'mni', which indicates that MvpWithin will first transform contrasts
        to MNI152 (2mm) space before it loads them. This option assumes
        that a 'reg' directory is present in the .feat-directory, including
        warp-files from functional to mni space
        (i.e. example_func2standara.nii.gz).
    beta2tstat : bool
        Whether to convert beta-values from COPEs to t-statistics by
        dividing them by the square-root of the res4d.
    remove_zeros : bool
        Whether to remove features (i.e. voxels) which are 0 across all
        trials (due to, e.g., being located outside the brain).
    X : ndarray
        Not necessary to pass MvpWithin, but needs to be defined as it is
        needed in the super-constructor.
    y : ndarray or list
        Labels or targets corresponding to the samples in ``X``. This can
        be used when read_labels is set to False.
    mask : str
        Absolute path to nifti-file that will be used as mask.
    mask_threshold : int or float
        Minimum value to binarize the mask when it's probabilistic.

    Attributes
    ----------
    mask_shape : tuple
        Shape of mask that patterns will be indexed with.
    nifti_header : Nifti1Header object
        Nifti-header from corresponding mask.
    affine : ndarray
        Affine corresponding to nifti-mask.
    voxel_idx : ndarray
        Array with integer-indices indicating which voxels are used in the
        patterns relative to whole-brain space. In other words, it allows to map
        back the patterns to a whole-brain orientation.
    X : ndarray
        The actual patterns (2D: samples X features)
    y : list or ndarray
        Array/list with labels/targets corresponding to samples in X.
    contrast_labels : list
        List of names corresponding to the y-values.
    """

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
        self.directory = None
        self.voxel_idx = None
        self.y = []
        self.contrast_labels = []
        self.X = []

    def create(self):
        """ Extracts (meta-)data from FEAT-directory given appropriate settings
        during initialization.

        Raises
        ------
        ValueError
            If the 'source'-directory doesn't exist.
        ValueError
            If the number of loaded contrasts does not equal the number of
            extracted labels.
        """

        if isinstance(self.source, str):
            self.source = [self.source]

        # Loop over sources
        for src in self.source:

            if '.feat' in src:
                self.directory = src
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

        if not n_stat == len(contrast_labels_current) and self.read_labels:
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

        # The following attributes are added for compatibility with MvpResults
        self.data_shape = cope_img.shape
        self.data_name = ['MvpWithin']
        self.featureset_id = np.zeros(mvp_data.shape[1], dtype=np.uint32)

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