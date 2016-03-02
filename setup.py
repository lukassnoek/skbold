from setuptools import setup, find_packages

install_requires = [
    'scikit-learn',
    'numpy',
    'nibabel',
    'nilearn',
    'joblib',
    'nipype'
]

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='skbold',
    version='0.1.7',
    description='Tools to convert and transform first-level fMRI data to scikit-learn compatible data-structures',
    long_description=readme(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
    keywords="fMRI scikit-learn RSA representational simililarity analysis",
    url='https://github.com/lukassnoek/skbold',
    author='Lukas Snoek',
    author_email='lukassnoek@gmail.com',
    license='MIT',
    platforms='Linux',
    packages=find_packages(),
    scripts=['bin/glm2mvp'],
    include_package_data=True,
    zip_safe=False)
