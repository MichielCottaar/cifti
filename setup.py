#!/usr/bin/env python

from setuptools import setup

setup(name='cifti',
      version='1.1',
      description='Interface to handle CIFTI files',
      author='Michiel Cottaar',
      author_email='Michiel.Cottaar@ndcn.ox.ac.uk',
      url='https://github.com/MichielCottaar/cifti',
      packages=['cifti', 'cifti.tests'],
      install_requires=['numpy', 'nibabel>=2.2', 'six'],
      license='MIT',
      )
