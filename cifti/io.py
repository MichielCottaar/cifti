from . import axis
from nibabel import cifti2
import numpy as np


def get_axes(file):
    """
    Extracts the axes form a CIFTI file header

    Parameters
    ----------
    file : str
        filename or an already opened Cifti2Image

    Returns
    -------
    tuple of N Axes describing the rows/columns along each of the dimensions
    The type of the axes will be determined by the information in the CIFTI file itself,
    not the extension of the CIFTI file
    """
    if isinstance(file, cifti2.Cifti2Image):
        img = file
    else:
        img = cifti2.Cifti2Image.from_filename(file)
    dim = 0

    shape = img.shape

    only_size_1_dimensions = True
    axes = []
    while dim != len(shape):
        axes.append(axis.from_mapping(img.header.matrix.get_index_map(dim)))
        if len(axes[-1]) != 1:
            only_size_1_dimensions = False
        if len(axes[-1]) != shape[dim]:
            if only_size_1_dimensions:
                shape = (1, ) + shape
            else:
                raise ValueError("CIFTI header expects %i elements along dimension %i, but %i were found" %
                                 (len(axes[-1]), dim, shape[dim]))
        dim += 1
    try:
        img.header.matrix.get_index_map(dim)
        raise ValueError("CIFTI header contains definition for dimension %i, but array is only %i-dimensional" %
                         (dim, dim))
    except cifti2.Cifti2HeaderError:
        pass
    return tuple(axes)



def read(file):
    """
    Loads a CIFTI file

    Parameters
    ----------
    file : str
        filename or an already opened file

    Returns
    -------
    - memory-mapped N-dimensional array with the actual data
    - tuple of N Axes describing the rows/columns along each of the dimensions
    The type of the axes will be determined by the information in the CIFTI file itself,
    not the extension of the CIFTI file
    """
    if isinstance(file, cifti2.Cifti2Image):
        img = file
    else:
        img = cifti2.Cifti2Image.from_filename(file)
    axes = get_axes(img)
    arr = img.get_data()
    while arr.ndim < len(axes):
        arr = arr[None, ...]
    return arr, axes


def write(filename, arr, axes):
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
    arr = np.asarray(arr)
    if len(axes) != arr.ndim:
        raise ValueError("Number of defined CIFTI axes (%i) does not match dimensionality of array (%i)" %
                         (len(axes), arr.ndim))
    for dim, ax, len_arr in zip(range(arr.ndim), axes, arr.shape):
        if len(ax) != len_arr:
            raise ValueError("Size of CIFTI axes (%i) does not match array size (%i) for dimension %i" %
                             (len(ax), len_arr, dim))
    img = cifti2.Cifti2Image(arr, axis.to_header(axes))
    img.to_filename(filename)
