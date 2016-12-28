import os
import nibabel as nib
import os.path as op

cwd = os.getcwd()
fdata = op.join(cwd, 'test_data', 'test_func.nii.gz')
fdata = nib.load(fdata)
npts = fdata.shape[-1] * fdata.header['pixdim'][4]

n_events = 10
n_conditions = 2
dur = 2

t_con_a = range(0, 200, 20)
t_con_b = range(10, 200, 20)

for i, (ta, tb) in enumerate(zip(t_con_a, t_con_b)):

    f = open(op.join(cwd, 'conditionA_trial%i.bfsl' % (i+1)), 'wb')
    f.write('%.3f\t%.3f\t1.000' % (ta, dur))
    f.close()

    f = open(op.join(cwd, 'conditionB_trial%i.bfsl' % (i + 1)), 'wb')
    f.write('%.3f\t%.3f\t1.000' % (tb, dur))
    f.close()
