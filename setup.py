import skbold
from setuptools import setup, find_packages

with open('requirements.txt') as rf:
    requirements = rf.readlines()

VERSION = skbold.__version__

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='skbold',
    version=VERSION,
    description='Utilities and tools for machine learning ' \
                'on BOLD-fMRI data.',
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
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False)
