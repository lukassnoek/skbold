"""
This module contains functions to set-up the analysis directory structure
for preprocessing. It assumes that there is a project directory (project_dir)
containing subject-specific subdirectories with all corresponding files. A
working directory is created with subject-specific subdirecties, which in turn
contain directories for separate "runs" (e.g. T1, func_A, func_B1, func_B2).

Lukas Snoek, University of Amsterdam
"""

from __future__ import division, print_function
import os
import glob
import shutil
import cPickle
import nibabel as nib


class DataOrganizer():
    "Organizes data into a sensible directory structure"

    def __init__(self, run_names, project_dir=os.getcwd(), subject_stem='sub', already_converted=False):

        self.run_names = run_names
        self.project_dir = project_dir
        self.working_dir = os.path.join(project_dir, 'working_directory')
        self.subject_stem = subject_stem
        self.subject_dirs = glob.glob(os.path.join(project_dir, '%s*' % subject_stem))

        if not self.subject_dirs and not already_converted:
            raise ValueError('Could not find valid subject directories!')

        if already_converted:
            self.subject_dirs = glob.glob(os.path.join(self.working_dir, '*%s*' % subject_stem))

    def create_scaninfo(self, par_filepath):
        """Create pickle with scan-params."""
        fID = open(par_filepath)
        scaninfo = nib.parrec.parse_PAR_header(fID)[0]
        fID.close()

        to_save = os.path.join(os.getcwd(), par_filepath[:-4] + '_scaninfo' + '.cPickle')

        with open(to_save,'wb') as handle:
            cPickle.dump(scaninfo, handle)

    def convert_parrec2nifti(self, remove_nifti=True, backup=True):

        if not os.path.isdir(self.working_dir):
            os.makedirs(self.working_dir)

        new_sub_dirs = []
        for sub_dir in self.subject_dirs:
            REC_files = glob.glob(os.path.join(sub_dir, '*.REC'))
            PAR_files = glob.glob(os.path.join(sub_dir, '*.PAR'))

            # Create scaninfo from PAR and convert .REC to nifti
            for REC, PAR in zip(REC_files, PAR_files):

                self.create_scaninfo(PAR)
                REC_name = REC[:-4]

                if not os.path.exists(REC_name + '.nii'):
                    print("Processing file %s ..." % REC_name, end="")
                    PR_obj = nib.parrec.load(REC)
                    nib.nifti1.save(PR_obj,REC_name)
                    print(" done.")

                else:
                    print("File %s was already converted." % REC_name)

            os.system("gzip %s" % os.path.join(sub_dir, '*.nii'))

            if remove_nifti:
                os.system('rm %s' % os.path.join(sub_dir, '*.nii'))

            if backup:
                self.backup_parrec(sub_dir)

            new_dir = os.path.join(self.working_dir, os.path.basename(sub_dir))
            shutil.copytree(sub_dir, new_dir)
            shutil.rmtree(sub_dir)
            new_sub_dirs.append(new_dir)

        self.subject_dirs = new_sub_dirs
        print("Done with conversion of par/rec to nifti.")

    def backup_parrec(self, sub_dir):

        backup_dir = os.path.join(self.project_dir, 'rawdata_backup')
        if not os.path.isdir(backup_dir):
            os.makedirs(backup_dir)

        backup_subdir = os.path.join(backup_dir, os.path.basename(sub_dir))
        if not os.path.isdir(backup_subdir):
            os.makedirs(backup_subdir)

        PAR_files = glob.glob(os.path.join(sub_dir, '*.PAR'))
        REC_files = glob.glob(os.path.join(sub_dir, '*.REC'))

        for PAR,REC in zip(PAR_files, REC_files):
            shutil.move(PAR, backup_subdir)
            shutil.move(REC, backup_subdir)

    def create_project_tree(self):
        """
        Moves files to subject specific directories and creates run-specific
        subdirectories
        """

        for sub_dir in self.subject_dirs:

            for func_run in self.run_names['func']:

                func_dir = os.path.join(sub_dir, 'func_%s' % func_run)
                if not os.path.isdir(func_dir):
                    os.makedirs(func_dir)

                # kinda ugly, but it works (accounts for different spellings)
                run_files = glob.glob(os.path.join(sub_dir, '*%s*' % func_run))
                run_files.extend(glob.glob(os.path.join(sub_dir, '*%s*' % func_run.upper())))
                run_files.extend(glob.glob(os.path.join(sub_dir, '*%s*' % func_run.capitalize())))

                _ = [shutil.move(f, func_dir) for f in run_files]

            struc_run = self.run_names['struc']
            struc_dir = os.path.join(sub_dir, struc_run)
            if not os.path.isdir(struc_dir):
                os.makedirs(struc_dir)

            struc_files = glob.glob(os.path.join(sub_dir, '*%s*' % struc_run))

            _ = [shutil.move(f, struc_dir) for f in struc_files]

            unallocated = glob.glob(os.path.join(sub_dir, '*'))

            for f in unallocated:
                if not os.path.isdir(f):
                    print('Unallocated file: %s' % f)

    def reset_pipeline(self):
        """
        Resets to analysis set-up to raw files aggregated in the project_dir.
        Retrieves log/phy files from ToProcess and PAR/REC from backup_dir.
        Subsequently removes all subdirectories of the project_dir.
        """

        for sub in self.subject_dirs:
            dirs = glob.glob(os.path.join(sub, '*'))

            for d in dirs:
                if os.path.isdir(d):

                    files = glob.glob(os.path.join(d, '*'))
                    _ = [shutil.move(to_move, sub) for to_move in files]

            shutil.copytree(sub, os.path.join(self.project_dir, os.path.basename(sub)))
            shutil.rmtree(sub)

        self.subject_dirs = glob.glob(os.path.join(self.project_dir, '*%s*' % self.subject_stem))

    def create_dir_structure_full(self):

        self.convert_parrec2nifti().create_project_tree()

    def get_filepaths(keyword, directory):
        """
        Given a certain keyword (including wildcards), this function walks
        through subdirectories relative to arg directory (i.e. root) and returns
        matches (absolute paths) and filenames_total (filenames).
        """

        matches = []
        filenames_total = []

        for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if keyword in filename:
                    matches.append(root + '/' + filename)
                    filenames_total.append(filename)
        return matches, filenames_total
