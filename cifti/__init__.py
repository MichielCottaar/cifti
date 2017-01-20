"""Contains helper objects to easily create, read, write, and manipulate CIFTI files

Functions
---------
load : Loads a CIFTI file returning the matrix and two Axis objects describing the row/columns
save : Saves a matrix and two Axis objects into a CIFTI file

Classes
-------
Axis : Parent class of all cifti axes
BrainModel : cifti axis, where each row/column is a voxel/vertex
Parcels : cifti axis, where each row/column is a parcel
Series : cifti axis describing a time series
Scalar : cifti axis where each row/column has its own name
Label : cifti axis where each row/column has its own name and label table
"""
from .io import load, save
from .axis import Axis, Series, Parcels, BrainModel, Label, Scalar
