"""Test for rocket parts."""

import pytest
import numpy as np

from mass_prop.rocket_parts import (
    DENSITY_KEY,
    HEIGHT_KEY,
    MASS_KEY,
    MOI_KEY,
    NAME_KEY,
    RADIUS_KEY,
    REL_POS_KEY,
    THICKNESS_KEY,
    CylindricalShellPart,
    Part
)

class TestPart:
    """Test Part object."""

    @pytest.mark.parametrize(
        "moi, rel_pos, valid_moi_at",
        [
            pytest.param(
                np.array([[0, 0, 0], [0, 1/12, 0], [0, 0, 1/12]]),
                np.array([[1/2], [0], [0]]),
                np.array([[0, 0, 0], [0, 1/3, 0], [0, 0, 1/3]]),
                id='1d-rod-moi'
            ),
            pytest.param(
                np.array([[1/4, 0, 0], [0, 1/4, 0], [0, 0, 1/2]]),
                np.array([[1], [0], [0]]),
                np.array([[1/4, 0, 0], [0, 5/4, 0], [0, 0, 3/2]]),
                id='2d-flat-circle-moi'
            )
        ]
    )
    def test_parallel_axis_theorem(self, moi, rel_pos, valid_moi_at):
        """Verify parallel axis theorem calculation."""
        test_mass_prop = {MASS_KEY: 1, MOI_KEY: moi, NAME_KEY: 'TEST-PART'}
        test_part = Part.from_mass_prop_dict(test_mass_prop)
        mass_prop_at = test_part.to_dict_at(rel_pos_from_cg=rel_pos)
        assert np.allclose(mass_prop_at[MOI_KEY], valid_moi_at)

    def test_add_child_parts_cg_shift(self):
        """Verify child parts are correctly added with cg shift."""
        # Create the root part object
        test_root_mass_prop = {MASS_KEY: 1, MOI_KEY: np.zeros(shape=(3, 3)), NAME_KEY: 'TEST-ROOT'}
        root_part = Part.from_mass_prop_dict(test_root_mass_prop)

        # Create child part object
        test_child_mass_prop_1 = {
            MASS_KEY: 1,
            MOI_KEY: np.zeros(shape=(3, 3)),
            NAME_KEY: 'TEST-FIRST-LAYER-1',
            REL_POS_KEY: np.array([[1], [0], [0]]),
        }
        test_child_part_1 = Part.from_mass_prop_dict(test_child_mass_prop_1)

        test_child_mass_prop_2 = {
            MASS_KEY: 0.5,
            MOI_KEY: np.zeros(shape=(3, 3)),
            NAME_KEY: 'TEST-FIRST-LAYER-2',
            REL_POS_KEY: np.array([[0], [1], [0]]),
        }
        test_child_part_2 = Part.from_mass_prop_dict(test_child_mass_prop_2)

        test_child_mass_prop_3 = {
            MASS_KEY: 1.5,
            MOI_KEY: np.zeros(shape=(3, 3)),
            NAME_KEY: 'TEST-SECOND-LAYER-1',
            REL_POS_KEY: np.array([[0], [0], [1]]),
        }
        test_child_part_3 = Part.from_mass_prop_dict(test_child_mass_prop_3)

        # Add all child parts and verify the updated mass prop
        root_part.add_child_part(child_part=test_child_part_1)
        assert root_part.mass == test_root_mass_prop[MASS_KEY] + test_child_mass_prop_1[MASS_KEY]
        assert np.allclose(root_part.moi, np.array([[0, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]))
        assert np.allclose(root_part._subassembly_cg_shift, np.array([[0.5], [0], [0]]))

        root_part.add_child_part(child_part=test_child_part_2)
        assert root_part.mass == test_root_mass_prop[MASS_KEY] + test_child_mass_prop_1[MASS_KEY] + test_child_mass_prop_2[MASS_KEY]
        assert np.allclose(root_part.moi, np.array([[0.4, 0.2, 0], [0.2, 0.6, 0], [0, 0, 1]]))
        assert np.allclose(root_part._subassembly_cg_shift, np.array([[0.4], [0.2], [0]]))

        test_child_part_1.add_child_part(child_part=test_child_part_3)
        assert root_part.mass == test_root_mass_prop[MASS_KEY] + test_child_mass_prop_1[MASS_KEY] + test_child_mass_prop_2[MASS_KEY] + test_child_mass_prop_3[MASS_KEY]
        assert np.allclose(root_part.moi, np.array([[1.375, 0.3125, -0.5625], [0.3125, 1.875, 0.1875], [-0.5625, 0.1875, 1.375]]))
        assert np.allclose(root_part._subassembly_cg_shift, np.array([[0.625], [0.125], [0.375]]))


class TestCylindricalShellPart:
    """Test CylindricalShellPart object."""

    def test_mass_prop_from_shape(self):
        """Verify that mass prop is correctly computed."""
        test_shape_prop = {HEIGHT_KEY: 1, RADIUS_KEY: 2, THICKNESS_KEY: 0.2, DENSITY_KEY: 0.5}
        mass_prop = CylindricalShellPart.compute_mass_prop_from_shape(test_shape_prop)
        # Verify with the correct mass prop
        assert MASS_KEY in mass_prop
        assert MOI_KEY in mass_prop
        assert np.isclose(mass_prop[MASS_KEY], 1.2566370614359177)
        assert np.allclose(mass_prop[MOI_KEY], np.array([[2.9384363286576543, 0, 0], [0, 2.9384363286576543, 0], [0, 0, 5.03911461635803]]))
