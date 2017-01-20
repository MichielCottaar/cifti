from six import string_types


def from_string(value, issurface=None):
    if '_' in value:
        items = [val.lower() for val in value.split('_')]
        if items[-1] in ['left', 'right', 'both']:
            orientation = items[-1]
            others = items[:-1]
        else:
            orientation = 'both'
            others = items
        if others[0] in ['nifti', 'cifti', 'gifti']:
            others = others[2:]
        primary = '_'.join(others)
    else:
        low = value.lower()
        if 'left' == low[-4:]:
            orientation = 'left'
            primary = low[:-4]
        elif 'right' == low[-5:]:
            orientation = 'right'
            primary = low[:-5]
        elif 'both' == low[-4:]:
            orientation = 'both'
            primary = low[:-4]
        else:
            orientation = 'both'
            primary = low
    if issurface is None:
        issurface = primary == 'cortex'
    if primary == '':
        primary = 'all'
    if primary == 'all':
        return BrainStructure(primary, orientation)
    if issurface:
        return BrainSurface(primary, orientation)
    else:
        return BrainVolume(primary, orientation)


class BrainStructure(object):
    model_type = None
    def __init__(self, primary, orientation='both'):
        self.primary = primary.lower()
        self.orientation = orientation.lower()

    def __eq__(self, other):
        if isinstance(other, string_types):
            other = from_string(other)
        return (self.primary == other.primary and self.orientation == other.orientation)

    @property
    def gifti(self, ):
        return str(self)

    def __str__(self, ):
        return self.primary.capitalize() + self.orientation.capitalize()

    def cifti(self, ):
        return 'CIFTI_STRUCTURE_' + self.primary.upper() + ('' if self.orientation == 'both' else ('_' + self.orientation.upper()))


class BrainSurface(BrainStructure):
    model_type='surface'
    def __init__(self, primary, orientation='both', nvertex=None, points=None, triangles=None):
        super(BrainSurface, self).__init__(primary, orientation)
        self.nvertex = nvertex
        self.points = points
        self.triangles = triangles

    @property
    def size(self, ):
        if self.points is not None:
            return self.points.shape[0]
        else:
            return self.nvertex

class BrainVolume(BrainStructure):
    model_type='volume'
    def __init__(self, primary, orientation='both', shape=None, affine=None):
        super(BrainVolume, self).__init__(primary, orientation)
        self.shape = shape
        self.affine = affine
