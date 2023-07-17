from enum import Enum, auto
import numpy as np


class CoordinateModes(Enum):
    CARTESIAN = auto()
    FRACTIONAL = auto()


class Structure:
    def __init__(self, name, struct_type, lattice_vec, atoms, coordinate_mode):
        self.name = name
        self.struct_type = struct_type
        self.lattice_vec = lattice_vec
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

    def to_fract(self):
        if self.coordinate_mode == CoordinateModes.FRACTIONAL:
            return Structure(self.name, self.struct_type, self.lattice_vec, self.atoms, self.coordinate_mode)
        elif self.coordinate_mode == CoordinateModes.CARTESIAN:
            cart_pos = [atom.pos for atom in self.atoms]
            # cart to fract transformation matrix
            trans_mat = np.linalg.inv(np.array(self.lattice_vec).T)
            fract_pos = list(np.matmul(trans_mat, np.array(cart_pos)).T.T)
            fract_atoms = []
            for i, atom in enumerate(self.atoms):
                fract_atoms.append(StructureAtom(fract_pos[i], atom.label))
            return Structure(self.name, self.struct_type, self.lattice_vec, fract_atoms, CoordinateModes.FRACTIONAL)
            
    def set_lattice_vec_from_cell_params(self, a, b, c, alpha, beta, gamma):
        a_vec = [a, 0, 0]
        b_vec = [b * np.cos(gamma), b * np.sin(gamma), 0]
        c_vec = [c * np.cos(beta), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma), c * np.sqrt(1 - np.square(np.cos(beta)) - np.square((np.cos(alpha) - np.cos(beta) * np.cos(gamma))))]
        self.lattice_vec = [a_vec, b_vec, c_vec]


    def get_cell_params_from_lattice_vec(self):
        a_vec, b_vec, c_vec = list(np.array(self.lattice_vec))
        a, b, c = map(np.linalg.norm, [a_vec, b_vec, c_vec])
        alpha = np.rad2deg(np.arccos(np.dot(b_vec, c_vec)
                                                / b / c))
        beta  = np.rad2deg(np.arccos(np.dot(a_vec, c_vec)
                                                / a / c))
        gamma = np.rad2deg(np.arccos(np.dot(a_vec, b_vec)
                                                / a / b))
        return a, b, c, alpha, beta, gamma

    def get_cell_volume(self):
        return np.abs(np.dot(self.lattice_vec[0], np.cross(self.lattice_vec[1], 
                                                                        self.lattice_vec[2])))

    @staticmethod
    def create_empty_structure():
        return Structure("", "", [], [], CoordinateModes.FRACTIONAL)


class StructureAtom:
    def __init__(self, pos, label):
        self.pos = pos
        self.label = label

    def __repr__(self):
        return f"{self.label}: {self.pos}"