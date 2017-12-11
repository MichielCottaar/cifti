from .. import io, axis
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
import os
import numpy as np
import tempfile

dirname = os.path.join(get_nibabel_data(), 'nitest-cifti2')

hcp_labels = ['CortexLeft', 'CortexRight', 'AccumbensLeft', 'AccumbensRight', 'AmygdalaLeft', 'AmygdalaRight',
              'brain_stem', 'CaudateLeft', 'CaudateRight', 'CerebellumLeft', 'CerebellumRight',
              'Diencephalon_ventral_left', 'Diencephalon_ventral_right', 'HippocampusLeft', 'HippocampusRight',
              'PallidumLeft', 'PallidumRight', 'PutamenLeft', 'PutamenRight', 'ThalamusLeft', 'ThalamusRight']

hcp_n_elements = [29696, 29716, 135, 140, 315, 332, 3472, 728, 755, 8709, 9144, 706,
                  712, 764, 795, 297, 260, 1060, 1010, 1288, 1248]

hcp_affine = np.array([[  -2.,    0.,    0.,   90.],
                       [   0.,    2.,    0., -126.],
                       [   0.,    0.,    2.,  -72.],
                       [   0.,    0.,    0.,    1.]])


def check_hcp_grayordinates(brain_model):
    """Checks that a BrainModel matches the expected 32k HCP grayordinates
    """
    assert isinstance(brain_model, axis.BrainModel)
    structures = list(brain_model.iter_structures())
    assert len(structures) == len(hcp_labels)
    idx_start = 0
    for idx, (name, _, bm), label, nel in zip(range(len(structures)), structures, hcp_labels, hcp_n_elements):
        if idx < 2:
            assert name in bm.nvertices.keys()
            assert (bm.voxel == -1).all()
            assert (bm.vertex != -1).any()
            assert bm.nvertices[name] == 32492
        else:
            assert name not in bm.nvertices.keys()
            assert (bm.voxel != -1).any()
            assert (bm.vertex == -1).all()
            assert (bm.affine == hcp_affine).all()
            assert bm.volume_shape == (91, 109, 91)
        assert name == axis.BrainModel.to_cifti_brain_structure_name(label)
        assert len(bm) == nel
        assert (bm.arr == brain_model.arr[idx_start:idx_start + nel]).all()
        idx_start += nel
    assert idx_start == len(brain_model)

    assert (brain_model.arr[:5]['vertex'] == np.arange(5)).all()
    assert structures[0][2].vertex[-1] == 32491
    assert structures[1][2].vertex[0] == 0
    assert structures[1][2].vertex[-1] == 32491
    assert (structures[-1][2].arr[-1] == brain_model.arr[-1]).all()
    assert (brain_model.arr[-1]['voxel'] == [38, 55, 46]).all()
    assert (brain_model.arr[70000]['voxel'] == [56, 22, 19]).all()


def check_Conte69(brain_model):
    """Checks that the BrainModel matches the expected Conte69 surface coordinates
    """
    assert isinstance(brain_model, axis.BrainModel)
    structures = list(brain_model.iter_structures())
    assert len(structures) == 2
    assert structures[0][0] == 'CIFTI_STRUCTURE_CORTEX_LEFT'
    assert structures[0][2].is_surface.all()
    assert structures[1][0] == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    assert structures[1][2].is_surface.all()
    assert (brain_model.voxel == -1).all()

    assert (brain_model.arr[:5]['vertex'] == np.arange(5)).all()
    assert structures[0][2].vertex[-1] == 32491
    assert structures[1][2].vertex[0] == 0
    assert structures[1][2].vertex[-1] == 32491


def check_rewrite(arr, axes, extension='.nii'):
    (fd, name) = tempfile.mkstemp(extension)
    io.write(name, arr, axes)
    arr2, axes2 = io.read(name)
    assert (arr == arr2).all()
    assert (axes == axes2)
    assert (axes == io.get_axes(name))
    return arr2, axes2


@needs_nibabel_data('nitest-cifti2')
def test_read_ones():
    arr, axes = io.read(os.path.join(dirname, 'ones.dscalar.nii'))
    assert (arr == 1).all()
    assert isinstance(axes[0], axis.Scalar)
    assert len(axes[0]) == 1
    assert axes[0].name[0] == 'ones'
    assert axes[0].meta[0] == {}
    check_hcp_grayordinates(axes[1])
    arr2, axes2 = check_rewrite(arr, axes)
    check_hcp_grayordinates(axes2[1])


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dscalar():
    arr, axes = io.read(os.path.join(dirname, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.dscalar.nii'))
    assert isinstance(axes[0], axis.Scalar)
    assert len(axes[0]) == 2
    assert axes[0].name[0] == 'MyelinMap_BC_decurv'
    assert axes[0].name[1] == 'corrThickness'
    assert axes[0].meta[0] == {'PaletteColorMapping': '<PaletteColorMapping Version="1">\n   <ScaleMode>MODE_AUTO_SCALE_PERCENTAGE</ScaleMode>\n   <AutoScalePercentageValues>98.000000 2.000000 2.000000 98.000000</AutoScalePercentageValues>\n   <UserScaleValues>-100.000000 0.000000 0.000000 100.000000</UserScaleValues>\n   <PaletteName>ROY-BIG-BL</PaletteName>\n   <InterpolatePalette>true</InterpolatePalette>\n   <DisplayPositiveData>true</DisplayPositiveData>\n   <DisplayZeroData>false</DisplayZeroData>\n   <DisplayNegativeData>true</DisplayNegativeData>\n   <ThresholdTest>THRESHOLD_TEST_SHOW_OUTSIDE</ThresholdTest>\n   <ThresholdType>THRESHOLD_TYPE_OFF</ThresholdType>\n   <ThresholdFailureInGreen>false</ThresholdFailureInGreen>\n   <ThresholdNormalValues>-1.000000 1.000000</ThresholdNormalValues>\n   <ThresholdMappedValues>-1.000000 1.000000</ThresholdMappedValues>\n   <ThresholdMappedAvgAreaValues>-1.000000 1.000000</ThresholdMappedAvgAreaValues>\n   <ThresholdDataName></ThresholdDataName>\n   <ThresholdRangeMode>PALETTE_THRESHOLD_RANGE_MODE_MAP</ThresholdRangeMode>\n</PaletteColorMapping>'}
    check_Conte69(axes[1])
    check_rewrite(arr, axes)


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dtseries():
    arr, axes = io.read(os.path.join(dirname, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.dtseries.nii'))
    assert isinstance(axes[0], axis.Series)
    assert len(axes[0]) == 2
    assert axes[0].start == 0
    assert axes[0].step == 1
    assert axes[0].size == arr.shape[0]
    assert (axes[0].arr == [0, 1]).all()
    check_Conte69(axes[1])
    check_rewrite(arr, axes)


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dlabel():
    arr, axes = io.read(os.path.join(dirname, 'Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii'))
    assert isinstance(axes[0], axis.Label)
    assert len(axes[0]) == 3
    assert (axes[0].name == ['Composite Parcellation-lh (FRB08_OFP03_retinotopic)',
                             'Brodmann lh (from colin.R via pals_R-to-fs_LR)', 'MEDIAL WALL lh (fs_LR)']).all()
    assert axes[0].label[1][70] == ('19_B05', (1.0, 0.867, 0.467, 1.0))
    assert (axes[0].meta == [{}] * 3).all()
    check_Conte69(axes[1])
    check_rewrite(arr, axes)


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_ptseries():
    arr, axes = io.read(os.path.join(dirname, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.ptseries.nii'))
    assert isinstance(axes[0], axis.Series)
    assert len(axes[0]) == 2
    assert axes[0].start == 0
    assert axes[0].step == 1
    assert axes[0].size == arr.shape[0]
    assert (axes[0].arr == [0, 1]).all()

    assert len(axes[1]) == 54
    voxels, vertices = axes[1]['ER_FRB08']
    assert voxels.shape == (0, 3)
    assert len(vertices) == 2
    assert vertices['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (206 // 2, )
    assert vertices['CIFTI_STRUCTURE_CORTEX_RIGHT'].shape == (206 // 2, )
    check_rewrite(arr, axes)
