"""Generates chiral perovskite structures
"""
from typing import Iterable

import numpy as np

from perovskite_utils.structure import Structure, StructureAtom, CoordinateModes


class PerovskiteStructure(Structure):
    """Extends :class:`~.Structure` to store chiral perovskite structural information"""

    def __init__(self, name: str, lattice_vec: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
                 atoms: list[StructureAtom], coordinate_mode: CoordinateModes) -> None:
        """:class:`~PerovskiteStructure` constructor

        :param name: name of the structure
        :type name: str
        :param lattice_vec: crystal lattice vectors specified in a Cartesian basis
        :type lattice_vec: Iterable[Iterable[float, float, float], Iterable[float, float, float], Iterable[float, float, float]]
        :param atoms: collection of atoms in the structure
        :type atoms: list[StructureAtom]
        :param coordinate_mode: coordinate basis used to specify the atomic positions
        :type coordinate_mode: CoordinateModes
        """
        super().__init__(name, "generated", lattice_vec, atoms, coordinate_mode)

    @staticmethod
    def from_param(
            name: str,
            lattice_vec: tuple[
                tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
            cart_M_disp: tuple[float, float],
            cart_layer_trans: tuple[float, float, float],
            theta_d: float,
            ax_bl: float,
            beta_bl_ratio: float,
            betap_bl_ratio: float,
            beta_d: float,
            betap_d: float,
            M_label: str,
            X_label: str,
    ) -> 'PerovskiteStructure':
        """
        Utility for generating the inorganic layers of a chiral perovskite structure from parameters

        :param name: name of the structure
        :type name: str
        :param lattice_vec: Cartesian lattice vectors for the structure in angstroms
        :type lattice_vec: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        :param cart_M_disp: in-plane cartesian displacement of the metal atom relative to the other metal atom in the unit cell
        :type cart_M_disp: tuple[float, float]
        :param cart_layer_trans: Cartesian displacement of an inorganic layer relative to the other inorganic layer in the unit cell in angstroms
        :type cart_layer_trans:  tuple[float, float, float]
        :param theta_d: angle in degrees relative to the inorganic-plane normal for axial halogen atoms
        :type theta_d: float
        :param ax_bl: bond length for the axial halogen atoms
        :type ax_bl: float
        :param beta_bl_ratio: bond length ratio for the bonds forming the beta bond angle
        :type beta_bl_ratio: float
        :param betap_bl_ratio: bond length ratio for the bonds forming the beta prime bond angle
        :type betap_bl_ratio: float
        :param beta_d: beta bond angle in degrees
        :type beta_d: float
        :param betap_d: beta prime bond angle in degrees
        :type betap_d: float
        :param M_label: label used by the metal atoms
        :type M_label: str
        :param X_label: label used by the halogen atoms
        :type X_label: str
        :return: a chiral perovskite structure generated from the parameters
        :rtype: PerovskiteStructure
        """
        a, b, c = np.linalg.norm(lattice_vec, axis=1)  # gets the cell length parameters a, b, c
        x, y = cart_M_disp
        trans_x, trans_y, trans_z = cart_layer_trans
        # convert angles to radians
        theta_r = np.deg2rad(theta_d)
        beta_r = np.deg2rad(beta_d)
        betap_r = np.deg2rad(betap_d)

        # generate the first inorganic layer
        layer1_M_pos, layer1_X_pos = PerovskiteStructure.generate_layer(
            a, b, x, y, theta_r, ax_bl, beta_bl_ratio, betap_bl_ratio, beta_r, betap_r
        )
        # the second inorganic layer is the first layer reflected about the xz, yz planes
        layer2_M_pos, layer2_X_pos = layer1_M_pos.copy(), layer1_X_pos.copy()
        layer2_M_pos[:, 0:2] = -layer2_M_pos[:, 0:2]  # reflect about xz, yz, plane
        layer2_X_pos[:, 0:2] = -layer2_X_pos[:, 0:2]  # reflect about xz, yz, plane

        # translates second inorganic layer
        layer2_M_pos = layer2_M_pos + np.tile(
            np.array([trans_x, trans_y, trans_z]), (layer2_M_pos.shape[0], 1)
        )
        layer2_X_pos = layer2_X_pos + np.tile(
            np.array([trans_x, trans_y, trans_z]), (layer2_X_pos.shape[0], 1)
        )

        # combine both inorganic layers into a single inorganic layer
        bilayer_M_pos = np.append(layer1_M_pos, layer2_M_pos, axis=0)
        bilayer_X_pos = np.append(layer1_X_pos, layer2_X_pos, axis=0)
        atoms = [StructureAtom(pos, M_label) for pos in bilayer_M_pos]
        atoms += [StructureAtom(pos, X_label) for pos in bilayer_X_pos]
        return PerovskiteStructure(name, lattice_vec, atoms, CoordinateModes.CARTESIAN)

    @staticmethod
    def generate_layer(
            a, b, x, y, theta_r, ax_bl, beta_bl_ratio, betap_bl_ratio, beta_r, betap_r
    ):
        """ #TODO: implement non-orthogonal axes
        Generates a single inorganic layer in a chiral perovskite structure using structure parameters

        :param a: in-plane cell length a
        :param b: float
        :type a: in-plane cell length b
        :type b: float
        :param x: displacement of one metal atom relative to the other along the x-direction
        :type x: float
        :param y: displacement of one metal atom relative to the other along the y-direction
        :type y: float
        :param theta_r: angle in radians of the bond joining an axial halogen atom and the metal atom relative to the noraml of the inorganic plane
        :type theta_r: float
        :param ax_bl: length of the axial halogen and metal bond
        :type ax_bl: float
        :param beta_bl_ratio: bond length ratio for the bonds forming the beta bond angle
        :type beta_bl_ratio: float
        :param betap_bl_ratio: bond length ratio for the bonds forming the beta prime bond angle
        :type betap_bl_ratio: float
        :param beta_r: beta bond angle in radians
        :type beta_r: float
        :param betap_r: beta prime bond angle in radians
        :type betap_r: float
        :return: positions of the metal and halogen atoms
        :rtype: tuple[Iterable[tuple[float, float, float]], Iterable[tuple[float, float, float]]]
        """
        # M atom positions
        M_pos = np.array([[x, y, 0], [0, 0, 0]])

        # translate M atom positions with periodic boundary conditions
        xy_M_pos = np.array([[x, y], [x - a, y], [x, y - b], [x - a, y - b]])
        M_angles_r = np.arctan2(xy_M_pos[:, 1], xy_M_pos[:, 0])

        recip_beta_bl_ratio = 1 / beta_bl_ratio
        recip_beta_p_bl_ratio = 1 / betap_bl_ratio

        phi_beta_r_1 = np.arcsin(
            beta_bl_ratio
            * np.sin(beta_r)
            / np.sqrt(
                beta_bl_ratio * beta_bl_ratio - 2 * beta_bl_ratio * np.cos(beta_r) + 1
            )
        )
        phi_beta_r_2 = np.arcsin(
            recip_beta_bl_ratio
            * np.sin(beta_r)
            / np.sqrt(
                recip_beta_bl_ratio * recip_beta_bl_ratio
                - 2 * recip_beta_bl_ratio * np.cos(beta_r)
                + 1
            )
        )

        phi_betap_r_1 = np.arcsin(
            betap_bl_ratio
            * np.sin(betap_r)
            / np.sqrt(
                betap_bl_ratio * betap_bl_ratio
                - 2 * betap_bl_ratio * np.cos(betap_r)
                + 1
            )
        )
        phi_betap_r_2 = np.arcsin(
            recip_beta_p_bl_ratio
            * np.sin(betap_r)
            / np.sqrt(
                recip_beta_p_bl_ratio * recip_beta_p_bl_ratio
                - 2 * recip_beta_p_bl_ratio * np.cos(betap_r)
                + 1
            )
        )

        phi_r = np.array([phi_beta_r_1, phi_beta_r_2, phi_betap_r_1, phi_betap_r_2])

        X_angles_r = M_angles_r - phi_r
        M_dists = np.sqrt(np.sum(np.square(xy_M_pos), axis=1))
        denom_beta_1 = np.sqrt(
            beta_bl_ratio * beta_bl_ratio - 2 * beta_bl_ratio * np.cos(beta_r) + 1
        )
        denom_beta_2 = np.sqrt(
            recip_beta_bl_ratio * recip_beta_bl_ratio
            - 2 * recip_beta_bl_ratio * np.cos(beta_r)
            + 1
        )

        denom_betap_1 = np.sqrt(
            betap_bl_ratio * betap_bl_ratio - 2 * betap_bl_ratio * np.cos(betap_r) + 1
        )
        denom_betap_2 = np.sqrt(
            recip_beta_p_bl_ratio * recip_beta_p_bl_ratio
            - 2 * recip_beta_p_bl_ratio * np.cos(betap_r)
            + 1
        )
        denom = [denom_beta_1, denom_beta_2, denom_betap_1, denom_betap_2]
        X_mags = M_dists / denom

        # polar magnitudes
        X_equat_pos = np.array(
            [
                X_mags * np.cos(X_angles_r),
                X_mags * np.sin(X_angles_r),
                np.zeros(shape=X_mags.shape),
            ]
        ).T

        xy_atom_axial_vec = X_equat_pos[0, :] - M_pos[0, :]
        origin_atom_axial_vec = X_equat_pos[1, :] - M_pos[1, :]
        # normalize
        xy_mag = np.linalg.norm(xy_atom_axial_vec)
        origin_mag = np.linalg.norm(origin_atom_axial_vec)
        xy_atom_axial_vec /= xy_mag
        origin_mag /= origin_mag

        X_axial_displacements = np.array(
            [
                [
                    ax_bl * np.sin(theta_r) * xy_atom_axial_vec[0],
                    ax_bl * np.sin(theta_r) * xy_atom_axial_vec[1],
                    ax_bl * np.cos(theta_r),
                ],
                [
                    ax_bl * np.sin(theta_r) * xy_atom_axial_vec[0],
                    ax_bl * np.sin(theta_r) * xy_atom_axial_vec[1],
                    -ax_bl * np.cos(theta_r),
                ],
                [
                    ax_bl * np.sin(theta_r) * origin_atom_axial_vec[0],
                    ax_bl * np.sin(theta_r) * origin_atom_axial_vec[1],
                    ax_bl * np.cos(theta_r),
                ],
                [
                    ax_bl * np.sin(theta_r) * origin_atom_axial_vec[0],
                    ax_bl * np.sin(theta_r) * origin_atom_axial_vec[1],
                    -ax_bl * np.cos(theta_r),
                ],
            ]
        )
        X_axial_pos = np.repeat(M_pos, 2, axis=0) + X_axial_displacements
        X_pos = np.append(X_equat_pos, X_axial_pos, axis=0)

        return M_pos, X_pos
