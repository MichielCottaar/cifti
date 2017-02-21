from .. import axis
import numpy as np
from nose.tools import assert_raises
from .test_io import check_rewrite


rand_affine = np.random.randn(4, 4)
vol_shape = (5, 10, 3)


def brain_models():
    mask = np.zeros(vol_shape)
    mask[0, 1, 2] = 1
    mask[0, 4, 2] = True
    mask[0, 4, 0] = True
    yield axis.BrainModel.from_mask(mask, rand_affine, 'ThalamusRight')
    mask[0, 0, 0] = True
    yield axis.BrainModel.from_mask(mask, rand_affine)

    yield axis.BrainModel.from_surface([0, 5, 10], 15, 'CortexLeft')
    yield axis.BrainModel.from_surface([0, 5, 10, 13], 15)

    surface_mask = np.zeros(15, dtype='bool')
    surface_mask[[2, 9, 14]] = True
    yield axis.BrainModel.from_mask(surface_mask, brain_structure='CortexRight')


def parcels():
    bml = list(brain_models())
    return axis.Parcels.from_brain_models([('mixed', bml[0] + bml[2]), ('volume', bml[1]), ('surface', bml[3])])


def scalar():
    return axis.Scalar.from_names(['one', 'two', 'three'])

def label():
    return axis.Scalar.from_names(['one', 'two', 'three']).to_label({0: ('something', (0.2, 0.4, 0.1, 0.5)),
                                                                     1: ('even better', (0.3, 0.8, 0.43, 0.9))})

def series():
    yield axis.Series(3, 10, 4)
    yield axis.Series(8, 10, 3)
    yield axis.Series(3, 2, 4)


def axes():
    yield parcels()
    yield scalar()
    yield label()
    for elem in brain_models():
        yield elem
    for elem in series():
        yield elem


def test_brain_models():
    bml = list(brain_models())
    assert len(bml[0]) == 3
    assert (bml[0].vertex == -1).all()
    assert (bml[0].voxel == [[0, 1, 2], [0, 4, 0], [0, 4, 2]]).all()
    assert bml[0][1][0] == False
    assert (bml[0][1][1] == [0, 4, 0]).all()
    assert bml[0][1][2] == axis.BrainModel.to_cifti_brain_structure_name('thalamus_right')
    assert len(bml[1]) == 4
    assert (bml[1].vertex == -1).all()
    assert (bml[1].voxel == [[0, 0, 0], [0, 1, 2], [0, 4, 0], [0, 4, 2]]).all()
    assert len(bml[2]) == 3
    assert (bml[2].voxel == -1).all()
    assert (bml[2].vertex == [0, 5, 10]).all()
    assert bml[2][1] == (True, 5, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    assert len(bml[3]) == 4
    assert (bml[3].voxel == -1).all()
    assert (bml[3].vertex == [0, 5, 10, 13]).all()
    print(bml[4][1], bml[4][1][-1])
    assert bml[4][1] == (True, 9, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    assert len(bml[4]) == 3
    assert (bml[4].voxel == -1).all()
    assert (bml[4].vertex == [2, 9, 14]).all()

    for bm, label in zip(bml, ['ThalamusRight', 'Other', 'cortex_left', 'cortex']):
        structures = list(bm.iter_structures())
        assert len(structures) == 1
        struc = structures[0][0]
        assert struc == axis.BrainModel.to_cifti_brain_structure_name(label)
        if 'CORTEX' in struc:
            assert bm.nvertices[struc] == 15
        else:
            assert struc not in bm.nvertices
            assert (bm.affine == rand_affine).all()
            assert bm.volume_shape == vol_shape

    bmt = bml[0] + bml[1] + bml[2] + bml[3]
    assert len(bmt) == 14
    structures = list(bmt.iter_structures())
    assert len(structures) == 4
    for bm, (name, bm_split) in zip(bml, structures):
        assert bm == bm_split
        assert (bm_split.name == name).all()
        assert bm == bmt[bmt.name == bm.name[0]]
        assert bm == bmt[np.where(bmt.name == bm.name[0])]

    bmt = bmt + bml[3]
    assert len(bmt) == 18
    structures = list(bmt.iter_structures())
    assert len(structures) == 4
    assert len(structures[-1][1]) == 8


def test_parcels():
    prc = parcels()
    assert isinstance(prc, axis.Parcels)
    assert prc['mixed'][0].shape == (3, 3)
    assert len(prc['mixed'][1]) == 1
    assert prc['mixed'][1]['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (3, )

    assert prc['volume'][0].shape == (4, 3)
    assert len(prc['volume'][1]) == 0

    assert prc['surface'][0].shape == (0, 3)
    assert len(prc['surface'][1]) == 1
    assert prc['surface'][1]['CIFTI_STRUCTURE_CORTEX'].shape == (4, )


def test_scalar():
    sc = scalar()
    assert len(sc) == 3
    assert isinstance(sc, axis.Scalar)
    assert (sc.name == ['one', 'two', 'three']).all()
    assert sc[1] == ('two', {})


def test_series():
    sr = list(series())
    assert (sr[0].arr == np.arange(4) * 10 + 3).all()
    assert (sr[1].arr == np.arange(3) * 10 + 8).all()
    assert (sr[2].arr == np.arange(4) * 2 + 3).all()
    assert ((sr[0] + sr[1]).arr == np.arange(7) * 10 + 3).all()
    assert ((sr[1] + sr[0]).arr == np.arange(7) * 10 + 8).all()
    assert ((sr[1] + sr[0] + sr[0]).arr == np.arange(11) * 10 + 8).all()
    assert sr[1][2] == 28
    assert sr[1][-2] == sr[1].arr[-2]
    assert_raises(ValueError, lambda: sr[0] + sr[2])
    assert_raises(ValueError, lambda: sr[2] + sr[1])

    # test slicing
    assert (sr[0][1:3].arr == sr[0].arr[1:3]).all()
    assert (sr[0][1:].arr == sr[0].arr[1:]).all()
    assert (sr[0][:-2].arr == sr[0].arr[:-2]).all()
    assert (sr[0][1:-1].arr == sr[0].arr[1:-1]).all()
    assert (sr[0][1:-1:2].arr == sr[0].arr[1:-1:2]).all()
    assert (sr[0][::2].arr == sr[0].arr[::2]).all()
    assert (sr[0][:10:2].arr == sr[0].arr[::2]).all()
    assert (sr[0][10::-1].arr == sr[0].arr[::-1]).all()
    assert (sr[0][3:1:-1].arr == sr[0].arr[3:1:-1]).all()
    assert (sr[0][1:3:-1].arr == sr[0].arr[1:3:-1]).all()


def test_writing():
    for ax1 in axes():
        for ax2 in axes():
            arr = np.random.randn(len(ax1), len(ax2))
            print(ax1, ax2)
            check_rewrite(arr, (ax1, ax2))
