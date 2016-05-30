import numpy as np
import nibabel as nib
import pandas as pd
import os
import glob


def create_MR_params_csv(toprocess_dir, output_dir):

    os.chdir(toprocess_dir)
    subDirs = glob.glob('pi*')

    df = pd.DataFrame(columns=['Sub', 'Task', 'TR', 'zero_voxels', 'n_slices', 'n_dynamics', 'dim1', 'dim2'])
    row_n = 0

    for curSubDir in subDirs:
        os.chdir(os.path.join(toprocess_dir, curSubDir))
        taskDirs = glob.glob('201*piop*')
        print('Processing subject %s' %(curSubDir))

        for curTaskDir in taskDirs:
            os.chdir(os.path.join(toprocess_dir, curSubDir, curTaskDir))

            task_name = curTaskDir.split("piop", 1)[1]
            print('Task %s\n' % (task_name))

            img = nib.load('func_data_mcst_sg_ss5.nii.gz')
            tr = img.header.get_zooms()[-1]
            (dim1, dim2, n_slices, n_dynamics) = img.header['dim'][1:5]
            zero_voxels = np.sum(img.get_data().flatten()==0)/n_dynamics  #this takes a while, set zero_voxels to 0 for debug /checking everything else
            df.loc[row_n] = [curSubDir, task_name, tr, zero_voxels, n_slices, n_dynamics, dim1, dim2]
            row_n += 1

    df.to_csv(output_dir, columns=['Sub', 'Task', 'TR', 'zero_voxels', 'n_slices', 'n_dynamics', 'dim1', 'dim2'])

if __name__ == '__main__':
    create_MR_params_csv(toprocess_dir='/home/c6076769/ToProcess', output_dir='/home/c6076769/MultimodalDecoding/mr_params.csv')
