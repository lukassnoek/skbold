from setuptools import setup

install_requires = [
	'scikit-learn',
	'numpy',
	'nibabel',
	'nilearn'
	]

def readme():
	with open('README.rst') as f:
		return f.read()

setup(
	name='scikit-bold',
	version='0.1',
    description='Tools to convert and transform first-level fMRI data to scikit-learn compatible data-structures',
    long_description=readme(),
    classifiers=[
    	'Development Status :: 1 - Planning',
    	'Intended Audience :: Science/Research',
    	'Operating System :: POSIX :: Linux',
    	'Programming Language :: Python :: 2.7',
    	'Topic :: Scientific/Engineering :: Bio-Informatics'],
    url='https://github.com/lukassnoek/scikit-bold',
    author='Lukas Snoek',
    author_email='lukassnoek@gmail.com',
    license='MIT',
    packages=['scikit-bold'],
    zip_safe=False)
