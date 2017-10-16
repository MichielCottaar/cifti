#!/usr/bin/env python

from distutils.core import setup

setup(name='cifti',
      version='1.0',
      description='Interface to handle CIFTI files',
      author='Michiel Cottaar',
      author_email='Michiel.Cottaar@ndcn.ox.ac.uk',
      url='https://git.fmrib.ox.ac.uk/ndcn0236/cifti',
      packages=['cifti', 'cifti.tests'],
      install_requires=['numpy', 'nibabel>=2.2', 'six'],
      )
