"""Utility classes for storing structural information
"""
import copy
from enum import Enum, auto
from typing import Iterable, Callable

import numpy as np


class CoordinateModes(Enum):
    """
    Enumeration for the cartesian and fractional coordinate modes

    :attr:`CoordinateModes.CARTESIAN` is written in a Cartesian basis, while :attr:`CoordinateModes.FRACTIONAL` is
    written in a lattice basis.
    """
    CARTESIAN = auto()
    FRACTIONAL = auto()


class Structure:
    """
    Stores structural information for a crystal

    :attr:`Structure._UC_EXPAND_LAT` is a list of the 27 nearest-neighbor lattice-symmetric translation in a fractional
    basis.
    """
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

    def __init__(self, name: str, struct_type: str, lattice_vec: Iterable[
        Iterable[float, float, float], Iterable[float, float, float], Iterable[float, float, float]],
                 atoms: list['StructureAtom'],
                 coordinate_mode: CoordinateModes) -> None:
        """:class:`~Structure` constructor

        :param name: name of the structure
        :type name: str
        :param struct_type: type of structure, such as 'relaxed' or 'experimental'
        :type struct_type: str
        :param lattice_vec: crystal lattice vectors specified in a Cartesian basis
        :type lattice_vec: Iterable[Iterable[float, float, float], Iterable[float, float, float], Iterable[float, float, float]]
        :param atoms: collection of atoms in the structure
        :type atoms: list[StructureAtom]
        :param coordinate_mode: coordinate basis used to specify the atomic positions
        :type coordinate_mode: CoordinateModes
        """
        self.name = name
        self.struct_type = struct_type
        self.lattice_vec = lattice_vec
        self.recip_vec = []  # TODO: refactor to include recip vec
        self.atoms = atoms
        self._coordinate_mode = coordinate_mode

    def __repr__(self):
        return f"{self.name} {self.struct_type}"

    @property
    def coordinate_mode(self):
        """
        Gets and sets the coordinate basis of the structure

        :param new_mode: new coordinate basis for the structure
        :type new_mode: CoordinateModes
        :return: the coordinate mode of the structure
        :rtype: CoordinateModes
        :raises ValueError: if the new_mode parameter is part of the CoordinateModes enumeration
        """
        return self._coordinate_mode

    @coordinate_mode.setter
    def coordinate_mode(self, new_mode: CoordinateModes):
        if not isinstance(new_mode, CoordinateModes):
            raise ValueError("new_mode argument is not of type CoordinateModes")
        self._coordinate_mode = new_mode

    def translate(self, trans: Iterable[float, float, float]) -> 'Structure':
        """
        Translates all atoms in the structure, returning a new structure with the translated atoms

        :param trans: translation
        :type trans: Iterable[float, float, float]
        :return: new structure with the translated atoms
        :rtype: Structure
        """
        # TODO: use atom translate method
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

    def to_fract(self) -> 'Structure':
        """
        Converts atomic positions in a Cartesian basis to a fractional (lattice vector) basis, returning a new structure

        :return: new structure with atomic positions in a fractional basis
        :rtype: Structure
        """
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
            # change of basis transformation
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

    def set_lattice_vec_from_cell_params(self, a: float, b: float, c: float, alpha: float, beta: float,
                                         gamma: float) -> None:
        """
        Sets lattice vectors, calculated from the crystallographic cell parameters

        :param a: "a" cell length
        :type a: float
        :param b: "b" cell length
        :type b: float
        :param c: "c" cell length
        :type c: float
        :param alpha: alpha cell angle in radians
        :type alpha: float
        :param beta: beta cell angle in radians
        :type beta: float
        :param gamma: gamma cell angle in radians
        :type gamma: float
        """
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

    def get_cell_params_from_lattice_vec(self) -> tuple[float, float, float, float, float, float]:
        """
        Calculates crystallographic cell parameters from the lattice vectors

        :return: a, b, c cell lengths and alpha, beta, gamma cell angles in degrees
        :rtype: tuple[float, float, float, float, float, float]
        """
        a_vec, b_vec, c_vec = list(np.array(self.lattice_vec))
        a, b, c = map(np.linalg.norm, [a_vec, b_vec, c_vec])
        alpha = np.rad2deg(np.arccos(np.dot(b_vec, c_vec) / b / c))
        beta = np.rad2deg(np.arccos(np.dot(a_vec, c_vec) / a / c))
        gamma = np.rad2deg(np.arccos(np.dot(a_vec, b_vec) / a / b))
        return a, b, c, alpha, beta, gamma

    def get_cell_volume(self) -> float:
        """
        Gets the cell volume in cubic angstroms

        :return: cell volume in cubic angstroms
        :rtype: float
        """
        return np.abs(
            np.dot(
                self.lattice_vec[0], np.cross(self.lattice_vec[1], self.lattice_vec[2])
            )
        )

    def detect_bonds(self, possible_bonds: Iterable[frozenset[str, str]],
                     bond_vec_predicate: Callable[[Iterable[float, float, float], frozenset[str, str]], bool],
                     check_boundaries=True) -> None:
        """
        Detects bonds between two atoms in the structure and adds Bond objects to each atom.

        :param possible_bonds: collection of frozenset containing str pairs of atom labels. These pairs of atom types are checked for bonds.
        :type possible_bonds: Iterable[frozenset[str, str]]
        :param bond_vec_predicate: predicate that accepts space vector joining two atoms and a frozenset with their atomic labels and returns whether these two atoms are bonded
        :type bond_vec_predicate: Callable[[Iterable[float, float, float], frozenset[str, str]], bool]
        :param check_boundaries: check additional atoms in the structure beyond the unit cell due periodic boundary conditions, defaults to True
        :type check_boundaries: bool
        """
        checked_atoms = []
        original_indices = []
        # add additional atoms on all surrounding unit cells due to periodic boundary conditions
        if check_boundaries:
            if self.coordinate_mode == CoordinateModes.FRACTIONAL:
                for translation in self._UC_EXPAND_LAT:
                    checked_atoms += [
                        atom.translate(translation) for atom in self.atoms
                    ]
                    original_indices += [i for i in range(len(self.atoms))]
            elif self.coordinate_mode == CoordinateModes.CARTESIAN:
                for translation in self._UC_EXPAND_LAT:
                    # need to convert the unit cell translations to Cartesian coordinates before adding new atoms
                    cart_trans = np.matmul(self.lattice_vec.T, translation.T).T
                    checked_atoms += [atom.translate(cart_trans) for atom in self.atoms]
                    original_indices += [i for i in range(len(self.atoms))]

        # iterate through pairs of atoms checking for bonds
        for n1, in_cell_atom in enumerate(self.atoms):
            for n2, all_atoms in enumerate(checked_atoms):
                # this avoids double counting since only need to iterate through each unique pair of atoms
                if n1 > original_indices[n2]:
                    continue

                label_pair = frozenset([in_cell_atom.label, all_atoms.label])
                if label_pair not in possible_bonds:
                    continue

                bond_vec = np.array(in_cell_atom.pos) - np.array(all_atoms.pos)
                # convert bond vector between two vectors into Cartesian coordinates
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

    def add_atoms(self, new_atoms: Iterable['StructureAtom']) -> None:
        """
        Adds new atoms to the structure

        :param new_atoms: collection of StructureAtom objects
        :type new_atoms: Iterable['StructureAtom']
        :raises ValueError: if not all elements of new_atoms are StructureAtom instances
        """
        if not all(map(lambda atom: isinstance(atom, StructureAtom), new_atoms)):
            raise ValueError("Not all elements of new_atoms are instances of StructureAtom.")
        self.atoms.extend(new_atoms)

    @staticmethod
    def create_empty_structure() -> 'Structure':
        """
        Utility method to create an empty structure with no atoms, no lattice vectors, an empty string name,
        an empty string struct_type, and in CoordinateModes.FRACTIONAL coordinates.

        :return: empty structure
        :rtype: Structure
        """
        return Structure("", "", [], [], CoordinateModes.FRACTIONAL)


class StructureAtom:
    """Stores the position, labels, and bonds of an atom"""
    def __init__(self, pos: Iterable[float, float, float], label: str, bonds: Iterable['Bond'] = None) -> None:
        """:class:`~StructureAtom` constructor

        :param pos: position of the atom
        :type pos: Iterable[float, float, float]
        :param label: label of the atom
        :type label: str
        :param bonds: collection of bonds for the atom participates in
        :type bonds: Iterable[Bond]
        """
        self.pos = pos
        self.label = label
        if bonds is None:
            self.bonds = []
        else:
            self.bonds = bonds

    def __repr__(self):
        return f"{self.label}: {self.pos}"

    def translate(self, translation: Iterable[float, float, float]) -> 'StructureAtom':
        """
        Translates atom, returning a new StructureAtom object

        :param translation: translation vector
        :type translation: Iterable[float, float, float]
        :return: translated atom
        :rtype: StructureAtom
        """
        new_pos = list(map(lambda e1, e2: e1 + e2, self.pos, translation))
        return StructureAtom(new_pos, self.label, self.bonds)


class Bond:
    """Stores information about a bond between two atoms"""
    def __init__(self, bond_vec: Iterable[float, float, float], atom1: StructureAtom, atom2: StructureAtom) -> None:
        """:class:`~Bond` constructor

        :param bond_vec: vector joining the two atoms in the bond
        :type bond_vec: Iterable[float, float, float]
        :param atom1: atom involved in the bond
        :type atom1: StructureAtom
        :param atom2: atom involved in the bond
        :type atom2: StructureAtom
        """
        self._bond_vec = bond_vec
        self._bond_length = None
        self._atoms = {atom1, atom2}

    @property
    def bond_length(self) -> float:
        """
        Gets the length of the bond. The bond length is calculated if not already.

        :return: bond length
        :rtype: float
        """
        if self._bond_length is None:
            self._bond_length = np.sqrt(np.sum(np.square(np.array([self._bond_vec]))))

        return self._bond_length

    @property
    def bond_vec(self) -> Iterable[float, float, float]:
        """
        Gets the bond vector.

        :return: bond vector
        :rtype: Iterable[float, float, float]
        """
        return self._bond_vec

    @property
    def atoms(self) -> set[StructureAtom, Structure]:
        """
        Gets the set of the pair of atoms involved in the bond

        :return: set of the pair of atoms involved in the bond
        :rtype: set[StructureAtom, Structure]
        """
        return self._atoms

    def __repr__(self):
        labels = [atom.label for atom in self._atoms]
        labels.sort()
        return f"{labels[0]}-{labels[1]}: {self.bond_length:.3f}"
