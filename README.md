[![DOI](https://zenodo.org/badge/80036201.svg)](https://zenodo.org/badge/latestdoi/80036201)
[![Build Status](https://travis-ci.org/MichielCottaar/cifti.svg?branch=master)](https://travis-ci.org/MichielCottaar/cifti)

<aside class="warning">
This package is deprecated in favor of the Axis implementation in nibabel 2.4.0 
([documentation](https://nipy.org/nibabel/reference/nibabel.cifti2.html#module-nibabel.cifti2.cifti2_axes)).
</aside>

With respect to the implementation here there are a few changes of note:
- The Axis objects is now created by calling `to_axes` on the `Cifti2Header` object. A new header can be created using the `from_axes` class method in `Cifti2Header`. This replaces the interface of loading/saving axes objects directly from/to filenames.
- All classes have been renamed to append "Axis", so `Scalar` is now `ScalarAxis`.
- The class constructors have been made more useful. These constructors now replace some factory functions included in this package:
  - `Scalar.from_names`
  - `Scalar.to_label`
- `BrainModel.is_surface` is now called `BrainModel.surface_mask` and an opposite `BrainModel.volume_mask` has also been defined.
- Axis objects are no longer described under the hood by a typed numpy array. This has very little practical effect.

This module allows for straight-forward creation of CIFTI files and the reading and manipulating of existing ones

The CIFTI format is used in brain imaging to store data acquired across the brain volume (in voxels) and/or 
the brain surface (in vertices). The format is unique in that it can store data from both volume and 
surface as opposed to NIftI, which only covers the brain volume, and GIftI, which only covers the brain surface. 
See http://www.nitrc.org/projects/cifti for specification of the CIFTI format.

Each type of CIFTI axes describing the rows/columns in a CIFTI matrix is given a unique class:
- `BrainModel`: each row/column is a voxel or vertex
- `Parcels`: each row/column is a group of voxels and/or vertices
- `Series`: each row/column is a timepoint, which increases monotonically
- `Scalar`: each row/column has a unique name (with optional meta-data)
- `Label`: each row/column has a unique name and label table (with optional meta-data)
All of these classes are derived from `Axis`

Reading a CIFTI file (through `read`) will return a matrix and a pair of axes describing the rows and columns of the matrix.
Similarly to write a CIFTI file (through `write`) requires a matrix and a pair of axes.

CIFTI axes of the same type can be concatenated by adding them together. 
Numpy indexing also works on them (except for Series objects, which have to remain monotonically increasing or decreasing)

Installation
------------
This package can be installed directly from pypi using:
```shell
pip install cifti
```

Creating new CIFTI axes
-----------------------
Several helper functions exist to create new CIFTI objects:
- BrainModel.from_mask creates a new BrainModel volume covering the non-zero values of a mask
- BrainModel.from_surface creates a new BrainModel surface covering the provided indices of a surface
- Scalar.from_names creates a CIFTI scalar axis based on a list of names (with no meta-data)
- Scalar.to_label converts a CIFTI scalar axis into a CIFTI Label axis given a label table

Examples
--------
So we can create brain models covering the left cortex and left thalamus using:
```python
import cifti
bm_cortex = cifti.BrainModel.from_mask(cortex_mask, brain_structure='cortex_left')
bm_thal = cifti.BrainModel.from_mask(thalamus_mask, affine=affine, brain_structure='thalamus_left')
```
A 1-dimensional mask will be automatically interpreted as a surface element and a 3-dimensional mask as a volume element.

These can be concatenated in a single brain model covering the left cortex and thalamus by simply adding them together
```python
bm_full = bm_cortex + bm_thal
```
Brain models covering the full HCP grayordinate space can be constructed by adding all the volumetric and 
surface brain models together like this (or by reading one from an already existing HCP file)

Getting a specific brain region from the full brain model is as simple as:
```python
assert bm_full[bm_full.struc == 'CortexLeft'] == bm_cortex
assert bm_full[bm_full.struc == 'ThalamusLeft'] == bm_thal
```

You can also iterate over all brain structures in a brain model:
```python
for brain_model, structure in bm_full.iter_structures(): ...
```
In this case there will be two iterations, namely: (bm_cortex, 'CortexLeft') and (bm_thal, 'ThalamusLeft')

Parcels can be constructed from selections of these brain models:
```python
parcel = cifti.Parcels.from_brain_models([('surface_parcel', bm_cortex[:100]),  # parcel containing first 100 vertices of the left cortex
                                          ('volume_parcel', bm_thal),  # parcel containing the full left thalamus
                                          ('combined_parcel', bm_full[[1, 8, 10, 19, 50, 120, 127])  # parcel containing specific indices of the full brain model
                                         ])
```

Time series are represented by their starting time (typically 0), step size (i.e. sampling time or TR), and number of elements:
```python
series = cifti.Series(start=0, step=100, size=5000)
```

So fMRI data with a TR of 100 ms covering the left cortex and thalamus with 5000 timepoints could be stored with
```python
cifti.write('Lcortex_thal.dtseries.nii', fMRI_data, (series, bm_full))
```

Similarly the curvature and cortical thickness on the left cortex can be stored in a single CIFTI file using
```python
cifti.write('Lgeometry.dscalar.nii', [curvature, thickness], (cifti.Scalar.from_names(['curvature', 'thickness']), bm_full))
```

Any CIFTI file can be read using
```python
arr, (axis1, axis2, ...) = cifti.read('test_file.nii')
```
If the file is not zipped (default for CIFTI) `arr` will be a memory-mapped array, so it should be fast even for a dense connectome. 
If the CIFTI file is zipped the full data will be loaded into memory, which might take a long time. In that case the `get_axes` function can be used to extract the axes from the header without reading the data.
