"""Define rocket part objects."""

import abc

import numpy as np
from dataclasses import dataclass
from rocket_util.logconfig import create_logger

LOG = create_logger(__name__)

# Constants keynames for mass prop extraction
MASS_KEY = 'mass'
MOI_KEY = 'moi'
NAME_KEY = 'name'
REL_POS_KEY = 'rel_origin_from_parent_cg'

# Constants keynames for shape prop extraction
RADIUS_KEY = 'radius'
HEIGHT_KEY = 'height'
THICKNESS_KEY = 'thickness'
DENSITY_KEY = 'density'


class AlreadyHaveParentError(ValueError):
    """Raise when trying to reassign a parent part."""

class MissingRequiredMassPropertyError(ValueError):
    """Raise when a required mass property not provided."""


@dataclass
class Part:
    """Define a generic point mass part object with MoI.

    Input mass prop assumptions:
    0. The origin refers to the self part's body frame origin or its original center of mass
    1. Mass and MoI is always with respect to the subassembly center of mass
    2. Part mass and MoI refers to the original part mass and MoI wrt to self origin
    """

    # Subassembly mass properties
    mass: float
    moi: np.ndarray
    # Part/subassembly name
    name: str
    # Record self center of mass position vector in the parent's body frame
    rel_pos_from_parent_origin: np.ndarray

    def __post_init__(self):
        """Set necessary data field default values."""
        # Auto save a copy of the initialization mass and MoI value
        self.part_mass: float = self.mass
        self.part_moi: np.ndarray = self.moi

        # Initialize inheritance information (disallow setup at construction)
        self.parent_part: 'Part' = None
        self.child_part_tree: list['Part'] = []

        ## Additionally define useful private fields for internal operations
        self._description: str = ''
        self._subassembly_cg_shift: np.ndarray = np.zeros(shape=(3, 1))
        # Record the total children mass and moment of inertia wrt the self origin
        self._children_total_mass: float = 0
        self._children_total_moi_to_origin: np.ndarray = np.zeros(shape=(3, 3))

    @staticmethod
    def check_prop_dict(input_prop: dict, required_keys: list):
        """Check if a mass prop dictionary is valid."""
        missing_prop = [key for key in required_keys if key not in input_prop]
        if missing_prop:
            raise MissingRequiredMassPropertyError(
                f'The following required property keys are missing {", ".join(missing_prop)} '
                f'in supplied input property dictionary {input_prop}.'
            )

    @classmethod
    def from_mass_prop_dict(cls, mass_prop: dict = {}) -> 'Part':
        """Construct a part from mass property dictionary."""
        Part.check_prop_dict(input_prop=mass_prop, required_keys=[MASS_KEY, MOI_KEY, NAME_KEY])
        rel_pos_specified = mass_prop.get(REL_POS_KEY)
        # Ensure the moment of inertia is alwasy 2D array
        return cls(
            mass=mass_prop[MASS_KEY],
            moi=np.atleast_2d(mass_prop[MOI_KEY]),
            name=mass_prop[NAME_KEY],
            rel_pos_from_parent_origin=rel_pos_specified if rel_pos_specified is not None else np.array([[0], [0], [0]]),
        )

    def _delta_moi_parallel_axis(self, rel_pos_from_cg: np.ndarray):
        """Return the additive term of 3D parallel axis theorem to mass evaluated at center of mass."""
        # Ensure numpy array size are at least 2D
        rel_pos_from_cg = np.atleast_2d(rel_pos_from_cg)
        identity = np.eye(rel_pos_from_cg.size)
        return self.mass * (rel_pos_from_cg.T @ rel_pos_from_cg * identity - rel_pos_from_cg @ rel_pos_from_cg.T)

    def add_child_part(self, child_part: 'Part'):
        """Compute with the added child part."""
        # Ensure that child does not already have a parent part
        if child_part.parent_part is not None:
            raise AlreadyHaveParentError(f'Part {child_part} has an existing parent {child_part.parent_part}')

        # Build the inheritance relations
        self.child_part_tree.append(child_part)
        child_part.parent_part = self

        # Update the mass property tree with the new part, set original mass prop to zero for addition
        self.update_mass_prop(child_part=child_part, original_mass_prop={
            MASS_KEY: 0, MOI_KEY: np.zeros(shape=(3, 3), dtype=np.float64), REL_POS_KEY: np.array([[0], [0], [0]]),
        })

    def update_mass_prop(self, child_part: 'Part', original_mass_prop: dict):
        """Update recursively the final mass property for the combination of all available decendences."""
        # Record the self original mass prop dictionary
        self_mass_prop = self.to_dict_at(rel_pos_from_cg=self.rel_pos_from_parent_origin)
        # Create child mass prop dictionary
        child_mass_prop = child_part.to_dict_at(rel_pos_from_cg=child_part.rel_pos_from_parent_origin)

        # Update the children total mass prop impact wrt to self origin
        self._children_total_mass += child_mass_prop[MASS_KEY] - original_mass_prop[MASS_KEY]
        self._children_total_moi_to_origin += child_mass_prop[MOI_KEY] - original_mass_prop[MOI_KEY]

        # Update the self subassembly mass
        new_total_mass = self.part_mass + self._children_total_mass

        # Compute new total mass and moi for parent-child combination wrt the origin of parent body frame
        new_cg_rel_to_origin = (
            self.mass * self._subassembly_cg_shift +
            child_mass_prop[MASS_KEY] * child_mass_prop[REL_POS_KEY] -
            original_mass_prop[MASS_KEY] * original_mass_prop[REL_POS_KEY]
        ) / new_total_mass
        self.mass = new_total_mass

        # Update self cg relative to the parent of self origin shift
        self.rel_pos_from_parent_origin = self.rel_pos_from_parent_origin + new_cg_rel_to_origin - self._subassembly_cg_shift
        self._subassembly_cg_shift = new_cg_rel_to_origin

        # Update the self subassembly moment of inertia
        new_total_moi_to_origin = self.part_moi + self._children_total_moi_to_origin
        self.moi = new_total_moi_to_origin - self._delta_moi_parallel_axis(rel_pos_from_cg=self._subassembly_cg_shift)

        # Update the parent of self part mass prop if exists
        if self.parent_part is not None:
            self.parent_part.update_mass_prop(child_part=self, original_mass_prop=self_mass_prop)

    def to_dict_at(self, rel_pos_from_cg: np.ndarray = np.array([[0], [0], [0]])) -> dict:
        """Output a mass prop dictionary with a relative reference position vector."""
        # Apply parallel axis theorem to moment of inertia term
        return {
            MASS_KEY: self.mass,
            MOI_KEY: self.moi + self._delta_moi_parallel_axis(rel_pos_from_cg=rel_pos_from_cg),
            NAME_KEY: self.name,
            REL_POS_KEY: self.rel_pos_from_parent_origin,
        }


@dataclass
class ShapedPart(metaclass=abc.ABCMeta):
    """Define a generic shape."""

    @staticmethod
    @abc.abstractmethod
    def compute_mass_prop_from_shape(shape_prop: dict):
        """Compute mass properties from the defined shape."""

    @classmethod
    def from_shape(cls, shape_prop: dict):
        """Part object factory from a shape property dictionary."""
        Part.check_prop_dict(input_prop=shape_prop, required_keys=[DENSITY_KEY])
        mass_prop = cls.compute_mass_prop_from_shape(shape_prop=shape_prop)
        # Accept a mass override if provided
        mass_prop[MASS_KEY] = shape_prop.get(MASS_KEY, mass_prop[MASS_KEY])
        mass_prop[NAME_KEY] = shape_prop[NAME_KEY]
        # Use mass property generated to construct a part object
        return cls.from_mass_prop_dict(mass_prop=mass_prop)


@dataclass
class CylindricalShellPart(ShapedPart, Part):
    """Define a cylindrical shell part object.

    Assume the component body frame align with the principle frame
    x, y - orthogonal directions along radial direction; z - axial direction 
    """

    @staticmethod
    def compute_mass_prop_from_shape(shape_prop: dict):
        """Compute moment of inertia for a cylindrical shell."""
        # Compute the volumn of the shape
        volumn = 2 * np.pi * shape_prop[HEIGHT_KEY] * shape_prop[THICKNESS_KEY] * shape_prop[RADIUS_KEY]
        # Compute the mass of the shape by uniform density
        mass = volumn * shape_prop[DENSITY_KEY]

        height = shape_prop[HEIGHT_KEY]
        inner_radius = shape_prop[RADIUS_KEY] - 1/2 * shape_prop[THICKNESS_KEY]
        outer_radius = shape_prop[RADIUS_KEY] + 1/2 * shape_prop[THICKNESS_KEY]
        # Compute moment of inertia in principle directions
        insq_plus_outsq = inner_radius ** 2 + outer_radius ** 2
        moi_rad = 1/12 * mass * (3 * insq_plus_outsq + 4 * height ** 2)
        moi_axi = 1/2 * mass * insq_plus_outsq

        # Return pure mass prop as a dictionary
        return {
            MASS_KEY: mass,
            MOI_KEY: np.array([[moi_rad, 0, 0], [0, moi_rad, 0], [0, 0, moi_axi]]),
            REL_POS_KEY: shape_prop.get(REL_POS_KEY, np.array([[0], [0], [0]])),
        }
    
    @classmethod
    def from_shape(cls, shape_prop):
        """Formulate a cylindrical shell from shape with shape properties checked."""
        Part.check_prop_dict(input_prop=shape_prop, required_keys=[RADIUS_KEY, HEIGHT_KEY, THICKNESS_KEY, NAME_KEY])
        return super().from_shape(shape_prop)


@dataclass
class DiskPart(ShapedPart, Part):
    """Define a horizontal disk part object.

    Assume the component body frame align with the principle frame
    x, y - orthogonal directions along radial direction; z - axial direction 
    """

    @staticmethod
    def compute_mass_prop_from_shape(shape_prop: dict):
        """Compute moment of inertia for a disk."""
        # Compute the volumn of the shape
        volumn = np.pi * shape_prop[THICKNESS_KEY] * shape_prop[RADIUS_KEY] ** 2
        # Compute the mass of the shape by uniform density
        mass = volumn * shape_prop[DENSITY_KEY]

        rsq = shape_prop[RADIUS_KEY] ** 2
        # Compute moment of inertia in principle directions
        moi_rad = 1/12 * mass * (3 * rsq + shape_prop[THICKNESS_KEY] ** 2)
        moi_axi = 1/2 * mass * rsq

        # Return pure mass prop as a dictionary
        return {
            MASS_KEY: mass,
            MOI_KEY: np.array([[moi_rad, 0, 0], [0, moi_rad, 0], [0, 0, moi_axi]]),
            REL_POS_KEY: shape_prop.get(REL_POS_KEY, np.array([[0], [0], [0]])),
        }

    @classmethod
    def from_shape(cls, shape_prop):
        """Formulate a cylindrical shell from shape with shape properties checked."""
        Part.check_prop_dict(input_prop=shape_prop, required_keys=[RADIUS_KEY, THICKNESS_KEY, NAME_KEY])
        return super().from_shape(shape_prop)


@dataclass
class CylinderTankPart(ShapedPart, Part):
    """Define a cylinder tank component part object.

    Assume the component body frame align with the principle frame
    x, y - orthogonal directions along radial direction; z - axial direction 
    """

    @classmethod
    def from_shape(cls, shape_prop):
        """Define a cylinder shaped tank part."""
        Part.check_prop_dict(
            input_prop=shape_prop, required_keys=[RADIUS_KEY, HEIGHT_KEY, THICKNESS_KEY, DENSITY_KEY, NAME_KEY]
        )
        # Instantiate an empty part and add children parts
        empty_mass_prop = {
            MASS_KEY: 0,
            MOI_KEY: np.zeros(shape=(3, 3), dtype=np.float64),
            NAME_KEY: shape_prop[NAME_KEY]
        }
        tank_part = cls.from_mass_prop_dict(mass_prop=empty_mass_prop)

        # Unpack shape properties
        radius = shape_prop[RADIUS_KEY]
        height = shape_prop[HEIGHT_KEY]
        thickness = shape_prop[THICKNESS_KEY]
        density = shape_prop[DENSITY_KEY]

        # Create the child part - upper dome
        upper_dome_name = f'{shape_prop[NAME_KEY]}_upper_dome'
        upper_dome = DiskPart.from_shape(shape_prop={
            RADIUS_KEY: radius,
            THICKNESS_KEY: thickness,
            NAME_KEY: upper_dome_name,
            REL_POS_KEY: np.array([[0], [0], [height / 2]]),
            DENSITY_KEY: density,
        })
        tank_part.add_child_part(upper_dome)

        # Create the child part - lower dome
        lower_dome_name = f'{shape_prop[NAME_KEY]}_lower_dome'
        lower_dome = DiskPart.from_shape(shape_prop={
            RADIUS_KEY: radius,
            THICKNESS_KEY: thickness,
            NAME_KEY: lower_dome_name,
            REL_POS_KEY: np.array([[0], [0], [-height / 2]]),
            DENSITY_KEY: density,
        })
        tank_part.add_child_part(lower_dome)

        # Create the child part - cylindrial wall
        cylinder_wall_name = f'{shape_prop[NAME_KEY]}_cylindrical_wall'
        cylinder_wall = CylindricalShellPart.from_shape(shape_prop={
            RADIUS_KEY: radius,
            HEIGHT_KEY: height,
            THICKNESS_KEY: thickness,
            NAME_KEY: cylinder_wall_name,
            DENSITY_KEY: density,
        })
        tank_part.add_child_part(cylinder_wall)

        return tank_part

    @staticmethod
    def compute_mass_prop_from_shape(shape_prop: dict):
        """Compute moment of inertia for a cylindrical tank."""
        tank = CylinderTankPart.from_shape(shape_prop=shape_prop)
        return {MASS_KEY: tank.mass, MOI_KEY: tank.moi, REL_POS_KEY: tank.rel_pos_from_parent_origin}
