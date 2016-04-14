from nilearn.plotting import plot_stat_map
from skbold import harvardoxford_path
import os.path as op
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

mask = op.join(harvardoxford_path, 'bilateral', 'Amygdala.nii.gz')
mask = nib.load(mask)
affine = mask.affine
data = mask.get_data()

n_vox = np.sum(data > 20)
rand = np.random.rand(n_vox)
data[data>20] = rand*100

img = nib.Nifti1Image(np.ma.array(data, mask=None), affine)
plot_stat_map(img)
plt.savefig('/home/lukas/test.png')