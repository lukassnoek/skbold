import os
import os.path as op
from glob import glob
import nibabel as nib
import multiprocessing


class FsfCrawler(object):
    """ Class to create subject-specific .fsf FEAT files.

    Given an fsf-template, this crawler creates subject-specific fsf-FEAT files,
    assuming that appropriate .bfsl files exist.
    """
    def __init__(self, template, preproc_dir, run_idf, output_dir=None,
                 subject_idf='sub', func_idf='sg_ss?', n_cores=1):
        """ Initializes FsfCrawler object. """
        self.template = template
        self.preproc_dir = preproc_dir

        if output_dir is None:
            output_dir = op.join(op.dirname(template), 'Firstlevels')

        if not op.isdir(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.run_idf = run_idf
        self.func_idf = func_idf
        self.subject_idf = subject_idf

        if n_cores == -1:
            n_cores = multiprocessing.cpu_count() - 1

        self.n_cores = n_cores
        self.clean_fsf = None
        self.out_fsf = []

    def crawl(self):
        """ Crawls subject-directories. """
        self._read_fsf()
        run_paths = op.join(preproc_dir, '%s*' % self.subject_idf,
                            '*%s*' % self.run_idf)
        self.sub_dirs = sorted(glob(run_paths))
        fsf_paths = [self._write_fsf(sub) for sub in self.sub_dirs]

        with open(op.join(op.dirname(self.template), 'batch_fsf.sh'), 'wb') as fout:

            for i, fsf in enumerate(fsf_paths):

                fout.write('feat %s &\n' % fsf)
                if (i+1) % self.n_cores == 0:
                    fout.write('wait\n')

    def _read_fsf(self):

        with open(self.template, 'rb') as f:
            template = f.readlines()

        template = [txt for txt in template if txt != '\n']
        template = [txt.replace('\n', '') for txt in template]
        template = [txt for txt in template if txt[0] != '#']

        self.clean_fsf = template

    def _write_fsf(self, sub_dir):

        func_file = glob(op.join(sub_dir, '*%s*.nii.gz' % self.func_idf))

        if len(func_file) == 0:
            msg = "Found no func-file for sub %s" % sub_dir
            raise IOError(msg)
        elif len(func_file) > 1:
            msg = "Found more than one func-file for sub %s: %r" % (sub_dir,
                                                                    func_file)
            raise IOError(msg)
        else:
            func_file = func_file[0]

        hdr = nib.load(func_file).header

        arg_dict = {}
        arg_dict['tr'] = hdr['pixdim'][4]
        arg_dict['npts'] = hdr['dim'][4]
        arg_dict['custom'] = glob(op.join(sub_dir, '*.bfsl'))
        arg_dict['feat_files'] = "\"%s\"" % func_file

        out_dir = op.join(self.output_dir, op.basename(op.dirname(sub_dir)))
        if not op.isdir(out_dir):
            os.makedirs(out_dir)

        feat_dir = op.join(out_dir, '%s.feat' % self.run_idf)
        arg_dict['outputdir'] = "\"%s\"" % feat_dir
        fsf_out = []

        for line in self.clean_fsf:

            if any(key in line for key in arg_dict.keys()):
                parts = [txt for txt in line.split(' ') if txt]
                keys = [key for key in arg_dict.keys() if key in line][0]
                values = arg_dict[keys]

                if not isinstance(values, list):
                    parts[-1] = values
                elif len(values) > 1:
                    ev = line.split(os.sep)[-1].replace("\"", '')
                    bfsls = glob(op.join(sub_dir, '*.bfsl'))
                    to_set = [bfsl for bfsl in bfsls if ev in bfsl]

                    if len(to_set) == 1:
                        parts[-1] = "\"%s\"" % to_set[0]

                parts = [str(p) for p in parts]
                fsf_out.append(" ".join(parts))
            else:
                fsf_out.append(line)

        with open(op.join(sub_dir, 'design.fsf'), 'wb') as fsfout:
            print("Writing fsf to %s" % sub_dir)
            fsfout.write("\n".join(fsf_out))

        return op.join(sub_dir, 'design.fsf')

if __name__ == '__main__':

    base_dir = '/media/lukas/piop/'
    preproc_dir = op.join(base_dir, 'PIOP', 'PIOP_PREPROC_MRI')
    testfile = op.join(base_dir, 'AnticipatieStage', 'FullFactorial.fsf')

    fsf = FsfCrawler(testfile, preproc_dir, run_idf='piopanticipatie',
                     subject_idf='pi', n_cores=-1)
    fsf.crawl()