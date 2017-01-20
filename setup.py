#!/usr/bin/env python

from distutils.core import setup

setup(name='cifti',
      version='0.0',
      description='Creates a simplified interface to handle CIFTI files',
      author='Michiel Cottaar',
      author_email='Michiel.Cottaar@ndcn.ox.ac.uk',
      url='https://git.fmrib.ox.ac.uk/ndcn0236/gyralcoord',
      packages=['cifti', 'cifti.tests'],
      install_requires=['numpy', 'nibabel', 'six'],
      )
