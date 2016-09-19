import skbold
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

REQUIREMENTS = [
    'scipy>=0.17',
    'numpy>=1.10',
    'scikit-learn>=0.17',
    'pandas>=0.17',
    'nibabel>=2.0',
    'matplotlib',
    'nipype>=0.12',
    'joblib>=0.9',
    'seaborn',
    'nilearn'
]

VERSION = skbold.__version__

def readme():
    with open('README.rst') as f:
        return f.read()

class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import tox
        import shlex
        args = self.tox_args
        if args:
            args = shlex.split(self.tox_args)
        tox.cmdline(args=args)

setup(
    name='skbold',
    version=VERSION,
    description='Utilities and tools for machine learning and other ' \
                'multivoxel pattern analyses of fMRI data.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
    keywords="fMRI MVPA decoding machine learning",
    url='http://skbold.readthedocs.io/en/latest/',
    author='Lukas Snoek',
    author_email='lukassnoek@gmail.com',
    license='MIT',
    platforms='Linux',
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    scripts=['bin/check_mc_output'],
    include_package_data=True,
    tests_require=['tox'],
    cmdclass={'test': Tox},
    zip_safe=False)
