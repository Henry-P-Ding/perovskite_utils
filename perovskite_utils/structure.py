import copy
from enum import Enum, auto

import numpy as np


class CoordinateModes(Enum):
    CARTESIAN = auto()
    FRACTIONAL = auto()


class Structure:
    _UC_EXPAND_LAT = np.array(
        [
            [0, 0, 0],  # origin
            [1, 0, 0],  # faces
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],  # edges
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [1, 1, 1],  # corners
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
        ]
    )

    def __init__(self, name, struct_type, lattice_vec, atoms, coordinate_mode):
        self.name = name
        self.struct_type = struct_type
        self.lattice_vec = lattice_vec
        self.recip_vec = []  # TODO: refactor to include recip vec
        self.atoms = atoms
        self.coordinate_mode = coordinate_mode

    def __repr__(self):
        return f"{self.name} {self.struct_type}"

    @property
    def coordinate_mode(self):
        return self._coordinate_mode

    @coordinate_mode.setter
    def coordinate_mode(self, new_mode):
        if not isinstance(new_mode, CoordinateModes):
            raise ValueError("new_mode argument is not of type CoordinateModes")
        self._coordinate_mode = new_mode

    def translate(self, trans):
        trans_atoms = [
            StructureAtom(
                pos=[
                    atom.pos[0] + trans[0],
                    atom.pos[1] + trans[1],
                    atom.pos[2] + trans[2],
                ],
                label=atom.label,
                bonds=copy.deepcopy(atom.bonds),
            )
            for atom in self.atoms
        ]
        return Structure(
            self.name,
            self.struct_type,
            copy.deepcopy(self.lattice_vec),
            trans_atoms,
            self.coordinate_mode,
        )

    def to_fract(self):
        if self.coordinate_mode == CoordinateModes.FRACTIONAL:
            return Structure(
                self.name,
                self.struct_type,
                self.lattice_vec,
                self.atoms,
                self.coordinate_mode,
            )
        elif self.coordinate_mode == CoordinateModes.CARTESIAN:
            cart_pos = [atom.pos for atom in self.atoms]
            # cart to fract transformation matrix
            trans_mat = np.linalg.inv(np.array(self.lattice_vec).T)
            fract_pos = list(np.matmul(trans_mat, np.array(cart_pos).T).T)
            fract_atoms = []
            for i, atom in enumerate(self.atoms):
                fract_atoms.append(StructureAtom(fract_pos[i], atom.label))
            return Structure(
                self.name,
                self.struct_type,
                self.lattice_vec,
                fract_atoms,
                CoordinateModes.FRACTIONAL,
            )

    def set_lattice_vec_from_cell_params(self, a, b, c, alpha, beta, gamma):
        a_vec = [a, 0, 0]
        b_vec = [b * np.cos(gamma), b * np.sin(gamma), 0]
        c_vec = [
            c * np.cos(beta),
            c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
            c
            * np.sqrt(
                1
                - np.square(np.cos(beta))
                - np.square((np.cos(alpha) - np.cos(beta) * np.cos(gamma)))
            ),
        ]
        self.lattice_vec = [a_vec, b_vec, c_vec]

    def get_cell_params_from_lattice_vec(self):
        a_vec, b_vec, c_vec = list(np.array(self.lattice_vec))
        a, b, c = map(np.linalg.norm, [a_vec, b_vec, c_vec])
        alpha = np.rad2deg(np.arccos(np.dot(b_vec, c_vec) / b / c))
        beta = np.rad2deg(np.arccos(np.dot(a_vec, c_vec) / a / c))
        gamma = np.rad2deg(np.arccos(np.dot(a_vec, b_vec) / a / b))
        return a, b, c, alpha, beta, gamma

    def get_cell_volume(self):
        return np.abs(
            np.dot(
                self.lattice_vec[0], np.cross(self.lattice_vec[1], self.lattice_vec[2])
            )
        )

    def detect_bonds(self, possible_bonds, bond_vec_predicate, check_boundaries=True):
        checked_atoms = []
        original_indices = []
        if check_boundaries:
            if self.coordinate_mode == CoordinateModes.FRACTIONAL:
                for translation in self._UC_EXPAND_LAT:
                    checked_atoms += [
                        atom.translate(translation) for atom in self.atoms
                    ]
                    original_indices += [i for i in range(len(self.atoms))]
            elif self.coordinate_mode == CoordinateModes.CARTESIAN:
                for translation in self._UC_EXPAND_LAT:
                    cart_trans = np.matmul(self.lattice_vec.T, translation.T).T
                    checked_atoms += [atom.translate(cart_trans) for atom in self.atoms]
                    original_indices += [i for i in range(len(self.atoms))]

        for n1, in_cell_atom in enumerate(self.atoms):
            for n2, all_atoms in enumerate(checked_atoms):
                if n1 > original_indices[n2]:
                    continue

                label_pair = frozenset([in_cell_atom.label, all_atoms.label])
                if label_pair not in possible_bonds:
                    continue

                bond_vec = np.array(in_cell_atom.pos) - np.array(all_atoms.pos)
                if self.coordinate_mode == CoordinateModes.FRACTIONAL:
                    trans_mat = np.array(self.lattice_vec).T
                    cart_bond_vec = np.matmul(trans_mat, bond_vec.T).T
                elif self.coordinate_mode == CoordinateModes.CARTESIAN:
                    cart_bond_vec = bond_vec
                else:
                    raise RuntimeError(
                        f"Invalid coordinate mode of {self.coordinate_mode}"
                    )
                if bond_vec_predicate(cart_bond_vec, label_pair):
                    in_cell_atom.bonds.append(
                        Bond(cart_bond_vec, in_cell_atom, all_atoms)
                    )
                    all_atoms.bonds.append(Bond(cart_bond_vec, in_cell_atom, all_atoms))

    def add_atoms(self, new_atoms):
        if not all(map(lambda atom: isinstance(atom, StructureAtom), new_atoms)):
            raise ValueError("Not all elements of new_atoms are instances of StructureAtom.")
        self.atoms.extend(new_atoms)

    @staticmethod
    def create_empty_structure():
        return Structure("", "", [], [], CoordinateModes.FRACTIONAL)


class StructureAtom:
    def __init__(self, pos, label, bonds=None):
        self.pos = pos
        self.label = label
        if bonds is None:
            self.bonds = []
        else:
            self.bonds = bonds

    def __repr__(self):
        return f"{self.label}: {self.pos}"

    def translate(self, translation):
        new_pos = list(map(lambda e1, e2: e1 + e2, self.pos, translation))
        return StructureAtom(new_pos, self.label, self.bonds)


class Bond:
    def __init__(self, bond_vec, atom1, atom2):
        self._bond_vec = bond_vec
        self._bond_length = None
        self._atoms = set([atom1, atom2])

    @property
    def bond_length(self):
        if self._bond_length is None:
            self._bond_length = np.sqrt(np.sum(np.square(np.array([self._bond_vec]))))

        return self._bond_length

    @property
    def bond_vec(self):
        return self._bond_vec

    @property
    def atoms(self):
        return self._atoms

    def __repr__(self):
        labels = [atom.label for atom in self._atoms]
        labels.sort()
        return f"{labels[0]}-{labels[1]}: {self.bond_length:.3f}"
