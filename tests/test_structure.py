import unittest

from perovskite_utils.structure import *

# triclinic lattice vec
# TODO: check cross product condition and comment here
TRICLINIC_LATTICE_VEC = (
    (1.00000000, 0.00000000, 0.00000000),
    (0.28905639, 0.56564332, 0.00000000),
    (0.84583030, 0.57577482, 0.34274498)
)

TRICLINIC_LATTICE_PARAMS = {
    "a": 1.00000000,
    "b": 0.63522119,
    "c": 1.07908278,
    "alpha": 0.58841866,
    "beta": 0.66996741,
    "gamma": 1.09836975
}

TRICLINIC_CELL_VOLUME = 0.19387140

TEST_ATOMS_CART = [
    StructureAtom(pos=(0.00000000, 0.00000000, 0.00000000), label="A"),
    StructureAtom(pos=(0.84955539, 0.81170342, 0.73985779), label="B")
]

TEST_ATOMS_FRACT = [
    StructureAtom(pos=(0.00000000, 0.00000000, 0.00000000), label="A"),
    StructureAtom(pos=(-0.75593306, -0.76227958, 2.15862476), label="B")
]


class TestCoordinateModes(unittest.TestCase):
    def test_coordinate_modes(self):
        self.assertTrue(CoordinateModes.FRACTIONAL in CoordinateModes)
        self.assertTrue(CoordinateModes.CARTESIAN in CoordinateModes)


class TestStructure(unittest.TestCase):
    def test_constructor(self):
        example_struct = Structure(
            name="example_struct",
            struct_type="example",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )
        self.assertEqual(example_struct.name, "example_struct")
        self.assertEqual(example_struct.struct_type, "example")
        self.assertEqual(example_struct.lattice_vec, TRICLINIC_LATTICE_VEC)
        self.assertEqual(example_struct.atoms, TEST_ATOMS_CART)
        self.assertEqual(example_struct.coordinate_mode, CoordinateModes.CARTESIAN)

    def test_translate(self):
        translation = (0.00091469, 0.70593067, 0.04667093)
        translated_atoms = [
            StructureAtom(pos=(0.00091469, 0.70593067, 0.04667093), label="A"),
            StructureAtom(pos=(0.85047008, 1.51763409, 0.78652872), label="B")
        ]

        original_struct = Structure(
            name="example_struct",
            struct_type="example_type",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )

        translated_struct = original_struct.translate(trans=translation)

        self.assertEqual(original_struct.name, "example_struct")
        self.assertEqual(original_struct.struct_type, "example_type")
        self.assertEqual(original_struct.lattice_vec, TRICLINIC_LATTICE_VEC)
        self.assertEqual(original_struct.atoms, TEST_ATOMS_CART)
        self.assertEqual(original_struct.coordinate_mode, CoordinateModes.CARTESIAN)

        self.assertEqual(translated_struct.name, "example_struct")
        self.assertEqual(translated_struct.struct_type, "example_type")
        self.assertEqual(translated_struct.lattice_vec, TRICLINIC_LATTICE_VEC)
        self.assertEqual(translated_struct.atoms, translated_atoms)
        self.assertEqual(translated_struct.coordinate_mode, CoordinateModes.CARTESIAN)

    def test_to_fract(self):
        cartesian_struct = Structure(
            name="cartesian_struct",
            struct_type="example_type",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )

        fract_struct = Structure(
            name="fract_struct",
            struct_type="example_type",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_FRACT),
            coordinate_mode=CoordinateModes.FRACTIONAL
        )

        cart2fract_struct = cartesian_struct.to_fract()
        fract2fract_struct = fract_struct.to_fract()

        self.assertEqual(cart2fract_struct.name, "cartesian_struct")
        self.assertEqual(cart2fract_struct.struct_type, "example_type")
        self.assertEqual(cart2fract_struct.lattice_vec, TRICLINIC_LATTICE_VEC)
        for atom_index, atom in enumerate(cart2fract_struct.atoms):
            self.assertEqual(atom.label, TEST_ATOMS_FRACT[atom_index].label)
            for coordinate_index, coordinate in enumerate(atom.pos):
                self.assertAlmostEqual(coordinate, TEST_ATOMS_FRACT[atom_index].pos[coordinate_index])
        self.assertEqual(cart2fract_struct.coordinate_mode, CoordinateModes.FRACTIONAL)

        self.assertEqual(fract2fract_struct.name, "fract_struct")
        self.assertEqual(fract2fract_struct.struct_type, "example_type")
        self.assertEqual(fract2fract_struct.lattice_vec, TRICLINIC_LATTICE_VEC)
        self.assertEqual(fract2fract_struct.atoms, TEST_ATOMS_FRACT)
        self.assertEqual(fract2fract_struct.coordinate_mode, CoordinateModes.FRACTIONAL)

    def test_set_lattice_vec_from_cell_params(self):
        example_struct = Structure(
            name="example_struct",
            struct_type="example",
            lattice_vec=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )

        example_struct.set_lattice_vec_from_cell_params(
            a=TRICLINIC_LATTICE_PARAMS["a"],
            b=TRICLINIC_LATTICE_PARAMS["b"],
            c=TRICLINIC_LATTICE_PARAMS["c"],
            alpha=TRICLINIC_LATTICE_PARAMS["alpha"],
            beta=TRICLINIC_LATTICE_PARAMS["beta"],
            gamma=TRICLINIC_LATTICE_PARAMS["gamma"],
        )

        self.assertEqual(example_struct.name, "example_struct")
        self.assertEqual(example_struct.struct_type, "example")
        for vec_index, vec in enumerate(example_struct.lattice_vec):
            for component_index, component in enumerate(vec):
                self.assertAlmostEqual(component, TRICLINIC_LATTICE_VEC[vec_index][component_index],
                                       msg=f"Compared vector {vec_index + 1}, component {component_index + 1}")
        self.assertEqual(example_struct.atoms, TEST_ATOMS_CART)
        self.assertEqual(example_struct.coordinate_mode, CoordinateModes.CARTESIAN)

    def test_get_cell_params_from_lattice_vec(self):
        example_struct = Structure(
            name="example_struct",
            struct_type="example",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )

        a, b, c, alpha, beta, gamma = example_struct.get_cell_params_from_lattice_vec()
        self.assertAlmostEqual(a, TRICLINIC_LATTICE_PARAMS["a"])
        self.assertAlmostEqual(b, TRICLINIC_LATTICE_PARAMS["b"])
        self.assertAlmostEqual(c, TRICLINIC_LATTICE_PARAMS["c"])
        self.assertAlmostEqual(np.deg2rad(alpha), TRICLINIC_LATTICE_PARAMS["alpha"])
        self.assertAlmostEqual(np.deg2rad(beta), TRICLINIC_LATTICE_PARAMS["beta"])
        self.assertAlmostEqual(np.deg2rad(gamma), TRICLINIC_LATTICE_PARAMS["gamma"])

        self.assertEqual(example_struct.name, "example_struct")
        self.assertEqual(example_struct.struct_type, "example")
        self.assertEqual(example_struct.atoms, TEST_ATOMS_CART)
        self.assertEqual(example_struct.coordinate_mode, CoordinateModes.CARTESIAN)

    def test_get_cell_volume(self):
        example_struct = Structure(
            name="example_struct",
            struct_type="example",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )

        self.assertEqual(example_struct.name, "example_struct")
        self.assertEqual(example_struct.struct_type, "example")
        self.assertEqual(example_struct.lattice_vec, TRICLINIC_LATTICE_VEC)
        self.assertEqual(example_struct.atoms, TEST_ATOMS_CART)
        self.assertEqual(example_struct.coordinate_mode, CoordinateModes.CARTESIAN)

        self.assertAlmostEqual(TRICLINIC_CELL_VOLUME, example_struct.get_cell_volume())

    def test_add_atoms(self):
        example_struct = Structure(
            name="example_struct",
            struct_type="example",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )

        new_atoms = [
            StructureAtom(pos=(1.0, 0.0, 0.0), label="C"),
            StructureAtom(pos=(0.0, 1.0, 0.0), label="D")
        ]
        example_struct.add_atoms(new_atoms)

        self.assertEqual(example_struct.name, "example_struct")
        self.assertEqual(example_struct.struct_type, "example")
        self.assertEqual(example_struct.lattice_vec, TRICLINIC_LATTICE_VEC)
        self.assertEqual(example_struct.atoms, [
            StructureAtom(pos=(0.00000000, 0.00000000, 0.00000000), label="A"),
            StructureAtom(pos=(0.84955539, 0.81170342, 0.73985779), label="B"),
            StructureAtom(pos=(1.0, 0.0, 0.0), label="C"),
            StructureAtom(pos=(0.0, 1.0, 0.0), label="D")
        ])
        self.assertEqual(example_struct.coordinate_mode, CoordinateModes.CARTESIAN)

    def test_add_atoms_error(self):
        example_struct = Structure(
            name="example_struct",
            struct_type="example",
            lattice_vec=copy.deepcopy(TRICLINIC_LATTICE_VEC),
            atoms=copy.deepcopy(TEST_ATOMS_CART),
            coordinate_mode=CoordinateModes.CARTESIAN
        )

        new_wrong_type_atoms = [
            "wrong_type"
        ]

        with self.assertRaises(ValueError) as cm:
            example_struct.add_atoms(new_wrong_type_atoms)

    def test_create_empty_structure(self):
        empty_struct = Structure.create_empty_structure()

        self.assertEqual(empty_struct.name, "")
        self.assertEqual(empty_struct.struct_type, "")
        self.assertEqual(empty_struct.lattice_vec, [])  # TODO: change to tuple
        self.assertEqual(empty_struct.atoms, []) #TOOD: change to tuple
        self.assertEqual(empty_struct.coordinate_mode, CoordinateModes.FRACTIONAL)

if __name__ == '__main__':
    unittest.main()
