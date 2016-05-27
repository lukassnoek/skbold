# Dual regression stuff (not tested or anything)

# Author: Lukas Snoek [lukassnoek.github.io]
# Contact: lukassnoek@gmail.com
# License: 3 clause BSD

import nibabel as nib
import numpy as np
import os.path as op


class DualRegression(object):

    def __init__(self, reference_file, func_file, normalize=True):

        print('Initializing DualRegression ...')

        self.reference_file = reference_file
        self.func_file = func_file
        self.directory = op.dirname(self.func_file)
        self.normalize = normalize

        func = nib.load(func_file)
        self.func_affine = func.affine
        self.func_data = func.get_data()

        ref_data = nib.load(reference_file).get_data()
        self.ref_data = ref_data.reshape((np.prod(ref_data.shape[:-1]), ref_data.shape[-1]))
        self.spatial_X = self._add_bias_feature(self.ref_data)

    def _spatial_regression(self):

        print('Doing spatial regression ...')

        func = self.func_data
        X = self.spatial_X
        self.ts = np.zeros((self.ref_data.shape[-1], func.shape[-1]))

        for t in xrange(func.shape[-1]):
            betas, _, _, _ = np.linalg.lstsq(X, func[:, :, :, t].ravel())
            self.ts[:, t] = betas[1:]

    def _normalize(self):

        m = self.ts.mean(axis=1)
        std = self.ts.std(axis=1)
        self.ts = (self.ts - m[:, np.newaxis]) / std[:, np.newaxis]

    def _add_bias_feature(self, X):

        return np.c_[np.ones(X.shape[0]), X]

    def _temporal_regression(self):

        print('Doing temporal regression ...')

        s = self.func_data.shape[:-1]
        out = np.zeros((s[0], s[1], s[2], self.ts.shape[0]))
        func = self.func_data

        if self.normalize:
            self._normalize()

        X = self._add_bias_feature(self.ts.T)

        for x in xrange(s[0]):

            if x % 10 == 0:
                print('Slice %i / %i' % (x+1, s[0]))

            for y in xrange(s[1]):

                for z in xrange(s[2]):

                    tmp = func[x, y, z, :]
                    betas, _, _, _ = np.linalg.lstsq(X, tmp)
                    out[x, y, z, :] = betas[1:]

        return out

    def run(self):

        self._spatial_regression()
        out = self._temporal_regression()

        img = nib.Nifti1Image(out, affine=self.func_affine)
        nib.save(img, op.join(self.directory, 'output_dual_regression'))

if __name__ == '__main__':

    func_file = '/home/lukas/testica/func_mni.nii.gz'
    smith_masks = '/home/lukas/testica/SmithRSMasks/PNAS_Smith09_bm10.nii.gz'

    dg = DualRegression(smith_masks, func_file)
    dg.run()


