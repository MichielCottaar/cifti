from . import axis
import nibabel
from nibabel import cifti2
import numpy as np


def load(file):
    """
    Loads a CIFTI file

    Parameters
    ----------
    file : str
        filename or an already opened file

    Returns
    -------
    tuple with:
    - memory-mapped vector/matrix with the actual data
    - two Axis describing each of the axes
    The type of the axes will be determined by the information in the CIFTI file itself, not the extension of the CIFTI file
    """
    img = cifti2.Cifti2Image.from_filename(file)
    arr = img.get_data()
    if arr.ndim == 1:
        arr = arr[None, :]
    axes = tuple(axis.from_mapping(img.header.matrix.get_index_map(idx)) for idx in range(2))
    if arr.shape[0] != len(axes[0]):
        raise ValueError("CIFTI header expects %i elements along first axis, but %i were found" %
                         (arr.shape[0], len(axes[0])))
    if arr.shape[1] != len(axes[1]):
        raise ValueError("CIFTI header expects %i elements along second axis, but %i were found" %
                         (arr.shape[1], len(axes[1])))
    return arr, axes


def save(filename, arr, axes):
    """
    Saves a CIFTI file

    Parameters
    ----------
    filename : str
        name of output CIFTI file
    arr : array
        data to be stored as vector or matrix
    axes : list[axis.Axis]
        axis explaining each of the dimensions in the arr
    """
    img = cifti2.Cifti2Image(np.asarray(arr), axis.to_header(axes))
    img.to_filename(filename)


