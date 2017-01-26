import numpy as np
from nibabel import cifti2
from . import structure
from six import string_types


def from_mapping(mim):
    """
    Parses the MatrixIndicesMap to find the appropriate CIFTI axis describing the rows or columns

    Parameters
    ----------
    mim : cifti2.Cifti2MatrixIndicesMap

    Returns
    -------
    subtype of Axis
    """
    return_type = {'CIFTI_INDEX_TYPE_SCALARS': Scalar,
                   'CIFTI_INDEX_TYPE_LABELS': Label,
                   'CIFTI_INDEX_TYPE_SERIES': Series,
                   'CIFTI_INDEX_TYPE_BRAIN_MODELS': BrainModel,
                   'CIFTI_INDEX_TYPE_PARCELS': Parcels}
    return return_type[mim.indices_map_to_data_type].from_mapping(mim)


def to_header(axes):
    """
    Converts the axes describing the rows/columns of a CIFTI vector/matrix to a Cifti2Header

    Parameters
    ----------
    axes : iterable[Axis]
        one or more axes describing each dimension in turn

    Returns
    -------
    cifti2.Cifti2Header
    """
    axes = list(axes)
    mims_all = []
    matrix = cifti2.Cifti2Matrix()
    for dim, ax in enumerate(axes):
        if ax in axes[:dim]:
            dim_prev = axes.index(ax)
            mims_all[dim_prev].applies_to_matrix_dimension.append(dim)
            mims_all.append(mims_all[dim_prev])
        else:
            mim = ax.to_mapping(dim)
            mims_all.append(mim)
            matrix.append(mim)
    return cifti2.Cifti2Header(matrix)


class Axis(object):
    """
    Generic object describing the rows or columns of a CIFTI vector/matrix

    Attributes
    ----------
    arr : np.ndarray
        (N, ) typed array with the actual information on each row/column
    """
    _use_dtype = None
    arr = None

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=self._use_dtype)

    def get_element(self, index):
        """
        Extracts a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        Description of the row/column
        """
        return self.arr[index]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_element(item)
        if isinstance(item, string_types):
            raise IndexError("Can not index an Axis with a string (except for Parcels)")
        return type(self)(self.arr[item])

    @property
    def size(self, ):
        return self.arr.size

    def __len__(self):
        return self.size

    def __eq__(self, other):
        return (type(self) == type(other) and
                len(self) == len(other) and
                (self.arr == other.arr).all())

    def __add__(self, other):
        """
        Concatenates two Axes of the same type

        Parameters
        ----------
        other : Axis
            axis to be appended to the current one

        Returns
        -------
        Axis of the same subtype as self and other
        """
        if type(self) == type(other):
            return type(self)(np.append(self.arr, other.arr))


class BrainModel(Axis):
    """
    Each row/column in the CIFTI vector/matrix represents a single vertex or voxel

    This Axis describes which vertex/voxel is represented by each row/column.

    Attributes
    ----------
    voxel : np.ndarray
        (N, 3) array with the voxel indices
    vertex :  np.ndarray
        (N, ) array with the vertex indices
    struc : np.ndarray
        (N, ) array with the brain structure objects
    """
    _use_dtype = np.dtype([('surface', 'bool'), ('vertex', 'i4'), ('voxel', ('i4', 3)), ('struc', 'object')])

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new BrainModel axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        BrainModel
        """
        nbm = np.sum([bm.index_count for bm in mim.brain_models])
        arr = np.zeros(nbm, dtype=cls._use_dtype)
        for bm in mim.brain_models:
            index_end = bm.index_offset + bm.index_count
            is_surface = bm.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
            bs = structure.from_string(bm.brain_structure, is_surface=is_surface)
            if is_surface:
                arr['vertex'][bm.index_offset: index_end] = bm.vertex_indices
                arr['surface'][bm.index_offset: index_end] = True
                bs.nvertex = bm.surface_number_of_vertices
            else:
                arr['voxel'][bm.index_offset: index_end, :] = bm.voxel_indices_ijk
                arr['surface'][bm.index_offset: index_end] = False
                bs.shape = mim.volume.volume_dimensions
                bs.affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
            arr['struc'][bm.index_offset: index_end] = bs
        return cls(arr)

    @classmethod
    def from_mask(cls, mask, affine, brain_structure='other'):
        """
        Creates a new BrainModel axis describing the provided mask

        Parameters
        ----------
        mask : np.ndarray
            all non-zero voxels will be included in the BrainModel axis
        affine : np.ndarray
            (4, 4) array with the voxel to mm transformation
        brain_structure : str
            Name of the brain structure (e.g. 'thalamus_left' or 'brain_stem')

        Returns
        -------
        BrainModel which covers the provided mask
        """
        voxels = np.array(np.where(mask != 0)).T
        arr = np.zeros(len(voxels), dtype=cls._use_dtype)
        arr['voxel'] = voxels
        bs = structure.from_string(brain_structure, is_surface=False)
        bs.affine = affine
        bs.shape = mask.shape
        arr['struc'] = bs
        return cls(arr)

    @classmethod
    def from_surface(cls, vertices, nvertex, brain_structure='Cortex'):
        """
        Creates a new BrainModel axis describing the vertices on a surface

        Parameters
        ----------
        vertices : np.ndarray
            indices of the vertices on the surface
        nvertes : int
            total number of vertices on the surface
        brain_structure : str
            Name of the brain structure (e.g. 'CortexLeft' or 'CortexRight')

        Returns
        -------
        BrainModel which covers (part of) the surface
        """
        arr = np.zeros(len(vertices), dtype=cls._use_dtype)
        arr['vertex'] = vertices
        arr['surface'] = True
        bs = structure.from_string(brain_structure, is_surface=True)
        bs.nvertex = nvertex
        arr['struc'] = bs
        return cls(arr)

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 3 elements
        - boolean, which is True if it is a surface element
        - vertex index if it is a surface element, otherwise array with 3 voxel indices
        - structure.BrainStructure object describing the brain structure the element was taken from
        """
        elem = self.arr[index]
        name = 'vertex' if elem['surface'] else 'voxel'
        return elem['surface'], elem[name], elem['struc']

    def to_mapping(self, dim):
        """
        Converts the brain model axis to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
        for bm, struc in self.iter_structures():
            is_surface = struc.model_type == 'surface'
            if is_surface:
                voxels = None
                vertices = cifti2.Cifti2VertexIndices(bm.vertex)
                nsurf = struc.size
            else:
                voxels = cifti2.Cifti2VoxelIndicesIJK(bm.voxel)
                vertices = None
                nsurf = None
                if mim.volume is None:
                    affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, matrix=struc.affine)
                    mim.volume = cifti2.Cifti2Volume(struc.shape, affine)
            idx_start = np.where(self.struc == struc)[0].min()
            cifti_bm = cifti2.Cifti2BrainModel(idx_start, len(bm),
                                               'CIFTI_MODEL_TYPE_SURFACE' if is_surface else 'CIFTI_MODEL_TYPE_VOXELS',
                                               struc.cifti, nsurf, voxels, vertices)
            mim.append(cifti_bm)
        return mim

    def iter_structures(self, ):
        """
        Iterates over all brain structures in the order that they appear along the axis

        Yields
        ------
        tuple with
        - brain model covering a specific brain structure
        - brain structure
        """
        idx_start = 0
        start_struc = self.struc[idx_start]
        for idx_current, struc in enumerate(self.struc):
            if start_struc != struc:
                yield self[idx_start: idx_current], start_struc
                idx_start = idx_current
                start_struc = self.struc[idx_start]
        yield self[idx_start: idx_current], start_struc

    @property
    def voxel(self, ):
        """The voxel represented by each row or column
        """
        return self.arr['voxel']

    @voxel.setter
    def voxel(self, values):
        self.arr['voxel'] = values

    @property
    def vertex(self, ):
        """The vertex represented by each row or column
        """
        return self.arr['vertex']

    @vertex.setter
    def vertex(self, values):
        self.arr['vertex'] = values

    @property
    def struc(self, ):
        """The brain structure to which the voxel/vertices of belong
        """
        return self.arr['struc']

    @struc.setter
    def struc(self, values):
        self.arr['struc'] = values


class Parcels(Axis):
    """
    Each row/column in the CIFTI vector/matrix represents a parcel of voxels/vertices

    This Axis describes which parcel is represented by each row/column.

    Attributes
    ----------
    name : np.ndarray
        (N, ) string array with the parcel names
    parcel :  np.ndarray
        (N, ) array with the actual parcels (each of which is a BrainModel object)

    Individual parcels can also be accessed based on their name, using
    >>> parcel = parcel_axis[name]
    """
    _use_dtype = np.dtype([('name', 'U60'), ('parcel', 'object')])

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new Parcels axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Parcels
        """
        nparcels = len(list(mim.parcels))
        arr = np.zeros(nparcels, dtype=cls._use_dtype)
        volume_shape = None if mim.volume is None else mim.volume.volume_dimensions
        affine = None if mim.volume is None else mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        for idx_parcel, parcel in enumerate(mim.parcels):
            nvoxels = 0 if parcel.voxel_indices_ijk is None else len(parcel.voxel_indices_ijk)
            total_size = nvoxels + np.sum([len(vertex) for vertex in parcel.vertices])
            parcel_arr = np.zeros(total_size, BrainModel._use_dtype)
            if nvoxels != 0:
                parcel_arr['surface'][:nvoxels] = False
                parcel_arr['voxel'][:nvoxels, :] = parcel.voxel_indices_ijk
                struc = structure.from_string('CIFTI_STRUCTURE_OTHER')
                struc.affine = affine
                struc.volume_shape = volume_shape
            idx_current = nvoxels
            for vertex in parcel.vertices:
                idx_next = idx_current + len(vertex)
                parcel_arr['surface'][idx_current:idx_next] = True
                parcel_arr['vertex'][idx_current:idx_next] = vertex
                parcel_arr['struc'][idx_current:idx_next] = structure.from_string(vertex.brain_structure)
                idx_current = idx_next
            arr[idx_parcel]['parcel'] = BrainModel(parcel_arr)
            arr[idx_parcel]['name'] = parcel.name
        return cls(arr)

    def to_mapping(self, dim):
        """
        Converts the parcels to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_PARCELS')
        for name, parcel in self.arr:
            voxels = cifti2.Cifti2VoxelIndicesIJK(parcel.arr['voxel'][~parcel.arr['surface']])
            if len(voxels) > 0:
                struc = parcel.arr['struc'][~parcel.arr['surface']][0]
                if mim.volume is None:
                    affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, matrix=struc.affine)
                    mim.volume = cifti2.Cifti2Volume(struc.shape, affine)
            element = cifti2.Cifti2Parcel(name, voxels)
            for bm, struc in parcel.iter_structures():
                if struc.model_type == 'surface':
                    element.vertices.append(cifti2.Cifti2Vertices(struc.cifti, bm.arr['vertex']))
            mim.append(element)
        return mim

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the parcel
        - BrainModel describing the voxels/vertices in the parcel
        """
        return self.arr['name'][index], self.arr['parcel'][index]

    @property
    def name(self, ):
        return self.arr['name']

    @name.setter
    def name(self, values):
        self.arr['name'] = values

    @property
    def parcel(self, ):
        return self.arr['parcel']

    @parcel.setter
    def parcel(self, values):
        self.arr['parcel'] = values

    def __getitem__(self, item):
        if isinstance(item, string_types):
            idx = np.where(self.name == item)[0]
            if len(idx) == 0:
                raise IndexError("Parcel %s not found" % item)
            if len(idx) > 1:
                raise IndexError("Multiple parcels with name %s found" % item)
            return self.parcel[idx[0]]
        super(Parcels, self).__getitem__(item)

    def __eq__(self, other):
        return (type(self) == type(other) and
                len(self) == len(other) and
                (self.name == other.name).all() and
                all(parc1 == parc2 for parc1, parc2 in zip(self.parcel, other.parcel)))


class Scalar(Axis):
    """
    Along this axis of the CIFTI vector/matrix each row/column has been given a unique name and optionally metadata

    Attributes
    ----------
    name : np.ndarray
        (N, ) string array with the parcel names
    meta :  np.ndarray
        (N, ) array with a dictionary of metadata for each row/column
    """
    _use_dtype = np.dtype([('name', 'U60'), ('meta', 'object')])

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new scalar axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Scalar
        """
        res = np.zeros(len(list(mim.named_maps)), dtype=cls._use_dtype)
        res['name'] = [nm.map_name for nm in mim.named_maps]
        res['meta'] = [{} if nm.metadata is None else dict(nm.metadata) for nm in mim.named_maps]
        return cls(res)

    @classmethod
    def from_names(cls, names):
        """
        Creates a new scalar axis with the given row/column names

        Parameters
        ----------
        names : List[str]
            gives a unique name to every row/column in the matrix

        Returns
        -------
        Scalar
        """
        res = np.zeros(len(names), dtype=cls._use_dtype)
        res['name'] = names
        res['meta'] = [{} for _ in names]
        return cls(res)

    def to_mapping(self, dim):
        """
        Converts the hcp_labels to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SCALARS')
        for elem in self.arr:
            meta = None if len(elem['meta']) == 0 else elem['meta']
            named_map = cifti2.Cifti2NamedMap(elem['name'], cifti2.Cifti2MetaData(meta))
            mim.append(named_map)
        return mim

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the scalar
        - dictionary with the element metadata
        """
        return self.arr['name'][index], self.arr['meta'][index]

    def to_label(self, labels):
        """
        Creates a new Label axis based on the Scalar axis

        Parameters
        ----------
        labels : list[dict]
            mapping from integers to (name, (R, G, B, A)), where `name` is a string and R, G, B, and A are floats
            between 0 and 1 giving the colour and alpha (transparency)

        Returns
        -------
        Label
        """
        res = np.zeros(self.size, dtype=Label._use_dtype)
        res['name'] = self.arr['name']
        res['meta'] = self.arr['meta']
        res['label'] = labels
        return Label(res)

    @property
    def name(self, ):
        return self.arr['name']

    @name.setter
    def name(self, values):
        self.arr['name'] = values

    @property
    def meta(self, ):
        return self.arr['meta']

    @meta.setter
    def meta(self, values):
        self.arr['meta'] = values


class Label(Axis):
    """
    Along this axis of the CIFTI vector/matrix each row/column has been given a unique name,
    label table, and optionally metadata

    Attributes
    ----------
    name : np.ndarray
        (N, ) string array with the parcel names
    meta :  np.ndarray
        (N, ) array with a dictionary of metadata for each row/column
    label : sp.ndarray
        (N, ) array with dictionaries mapping integer values to label names and RGBA colors
    """
    _use_dtype = np.dtype([('name', 'U60'), ('label', 'object'), ('meta', 'object')])

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new scalar axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Scalar
        """
        tables = [{key: (value.label, value.rgba) for key, value in nm.label_table.items()}
                  for nm in mim.named_maps]
        return Scalar.from_mapping(mim).to_label(tables)

    def to_mapping(self, dim):
        """
        Converts the hcp_labels to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_LABELS')
        for elem in self.arr:
            label_table = cifti2.Cifti2LabelTable()
            for key, value in elem['label'].items():
                label_table[key] = (value[0], ) + tuple(value[1])
            meta = None if len(elem['meta']) == 0 else elem['meta']
            named_map = cifti2.Cifti2NamedMap(elem['name'], cifti2.Cifti2MetaData(meta),
                                              label_table)
            mim.append(named_map)
        return mim

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 2 elements
        - unicode name of the scalar
        - dictionary with the label table
        - dictionary with the element metadata
        """
        return self.arr['name'][index], self.arr['label'][index], self.arr['meta'][index]

    @property
    def name(self, ):
        return self.arr['name']

    @name.setter
    def name(self, values):
        self.arr['name'] = values

    @property
    def meta(self, ):
        return self.arr['meta']

    @meta.setter
    def meta(self, values):
        self.arr['meta'] = values

    @property
    def label(self, ):
        return self.arr['label']

    @label.setter
    def label(self, values):
        self.arr['label'] = values


class Series(Axis):
    """
    Along this axis of the CIFTI vector/matrix the rows/columns increase monotonously in time

    This Axis describes the time point of each row/column.

    Attributes
    ----------
    start : float
        starting time point
    step :  float
        sampling time (TR)
    size : int
        number of time points
    """
    size = None

    def __init__(self, start, step, size):
        self.start = start
        self.step = step
        self.size = size

    @property
    def arr(self, ):
        return np.arange(self.size) * self.step + self.start

    @classmethod
    def from_mapping(cls, mim):
        """
        Creates a new Series axis based on a CIFTI dataset

        Parameters
        ----------
        mim : cifti2.Cifti2MatrixIndicesMap

        Returns
        -------
        Series
        """
        start = mim.series_start * 10 ** mim.series_exponent
        step = mim.series_step * 10 ** mim.series_exponent
        return cls(start, step, mim.number_of_series_points)

    def to_mapping(self, dim):
        """
        Converts the series to a MatrixIndicesMap for storage in CIFTI format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        cifti2.Cifti2MatrixIndicesMap
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SERIES')
        mim.series_exponent = 0
        mim.series_start = self.start
        mim.series_step = self.step
        mim.number_of_series_points = self.size
        return mim

    def extend(self, other_axis):
        """
        Concatenates two series

        Note: this will ignore the start point of the other axis

        Parameters
        ----------
        other_axis : Series
            other axis

        Returns
        -------
        Series
        """
        if other_axis.step != self.step:
            raise ValueError('Can only concatenate series with the same step size')
        return Series(self.start, self.step, self.size + other_axis.size)

    def __getitem__(self, item):
        if isinstance(item, slice):
            step = 1 if item.step is None else item.step
            idx_start = ((self.size - 1 if step < 0 else 0)
                         if item.start is None else
                         (item.start if item.start >= 0 else self.size + item.start))
            idx_end = ((-1 if step < 0 else self.size)
                       if item.stop is None else
                       (item.stop if item.stop >= 0 else self.size + item.stop))
            if idx_start > self.size:
                idx_start = self.size - 1
            if idx_end > self.size:
                idx_end = self.size
            nelements = (idx_end - idx_start) // step
            if nelements < 0:
                nelements = 0
            return Series(idx_start * self.step + self.start, self.step * step, nelements)
        elif isinstance(item, int):
            return self.get_element(item)
        raise IndexError('Series can only be indexed with integers or slices without breaking the regular structure')

    def get_element(self, index):
        """
        Gives the time point of a specific row/column

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        float
        """
        if index < 0:
            index = self.size + index
        if index >= self.size:
            raise IndexError("index %i is out of range for series with size %i" % (index, self.size))
        return self.start + self.step * index

    def __add__(self, other):
        """
        Concatenates two Series

        Parameters
        ----------
        other : Series
            Time series to append at the end of the current time series.
            Note that the starting time of the other time series is ignored.

        Returns
        -------
        Series
            New time series with the concatenation of the two

        Raises
        ------
        ValueError
            raised if the repetition time of the two time series is different
        """
        if isinstance(other, Series):
            if other.step != self.step:
                raise ValueError("Can not concatenate Series with different step sizes (i.e. different repetition times)")
            return Series(self.start, self.step, self.size + other.size)
        return NotImplemented
