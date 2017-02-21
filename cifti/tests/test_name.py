from .. import axis

equivalents = [('CIFTI_STRUCTURE_CORTEX_LEFT', ('CortexLeft', 'LeftCortex', 'left_cortex', 'Left Cortex',
                                                'Cortex_Left', 'cortex left', 'CORTEX_LEFT', 'LEFT CORTEX',
                                                ('cortex', 'left'), ('CORTEX', 'Left'), ('LEFT', 'coRTEX'))),
               ('CIFTI_STRUCTURE_CORTEX', ('Cortex', 'CortexBOTH', 'Cortex_both', 'both cortex',
                                           'BOTH_CORTEX', 'cortex', 'CORTEX', ('cortex', ),
                                           ('COrtex', 'Both'), ('both', 'cortex')))]


def test_name_conversion():
    func = axis.BrainModel.to_cifti_brain_structure_name
    for base_name, input_names in equivalents:
        assert base_name == func(base_name)
        for name in input_names:
            assert base_name == func(name)