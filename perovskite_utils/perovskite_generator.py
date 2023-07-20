import numpy as np
from perovskite_utils.structure import Structure, StructureAtom, CoordinateModes


class PerovskiteStructure(Structure):
    def __init__(self, name, lattice_vec, atoms, coordinate_mode):
        super().__init__(name, "generated", lattice_vec, atoms, coordinate_mode)

    @staticmethod
    def from_param(name, lattice_vec, cart_M_disp, cart_layer_trans, theta_d, ax_bl, beta_bl_ratio, betap_bl_ratio, beta_d, betap_d, M_label, X_label):
        a, b, c = np.linalg.norm(lattice_vec, axis=1)
        x, y = cart_M_disp
        trans_x, trans_y, trans_z = cart_layer_trans
        theta_r = np.deg2rad(theta_d)
        beta_r = np.deg2rad(beta_d)
        betap_r = np.deg2rad(betap_d)

        layer1_M_pos, layer1_X_pos = PerovskiteStructure.generate_layer(a, b, x, y, theta_r, ax_bl, beta_bl_ratio, betap_bl_ratio, beta_r, betap_r)
        layer2_M_pos, layer2_X_pos = layer1_M_pos.copy(), layer1_X_pos.copy()

        layer2_M_pos[:, 0:2] = -layer2_M_pos[:, 0:2]   # reflect about xz, yz, plane
        layer2_X_pos[:, 0:2] = -layer2_X_pos[:, 0:2]   # reflect about xz, yz, plane

        # translates second layer
        layer2_M_pos = layer2_M_pos + np.tile(np.array([trans_x, trans_y, trans_z]), (layer2_M_pos.shape[0], 1))
        layer2_X_pos = layer2_X_pos + np.tile(np.array([trans_x, trans_y, trans_z]), (layer2_X_pos.shape[0], 1))

        bilayer_M_pos = np.append(layer1_M_pos, layer2_M_pos, axis=0)
        bilayer_X_pos = np.append(layer1_X_pos, layer2_X_pos, axis=0)
        atoms = [StructureAtom(pos, M_label) for pos in bilayer_M_pos]
        atoms += [StructureAtom(pos, X_label) for pos in bilayer_X_pos]
        return PerovskiteStructure(name, lattice_vec, atoms, CoordinateModes.CARTESIAN)

    @staticmethod
    def generate_layer(a, b, x, y, theta_r, ax_bl, beta_bl_ratio, betap_bl_ratio, beta_r, betap_r):
        # M atom positions
        M_pos = np.array([
            [x, y, 0],
            [0, 0, 0]
        ])

        xy_M_pos = np.array([
            [x, y],
            [x-a, y],
            [x, y-b],
            [x-a, y-b]
        ])
        M_angles_r = np.arctan2(xy_M_pos[:, 1], xy_M_pos[:, 0])

        recip_beta_bl_ratio = 1 / beta_bl_ratio
        recip_beta_p_bl_ratio = 1 / betap_bl_ratio

        phi_beta_r_1  = np.arcsin(beta_bl_ratio * np.sin(beta_r ) / np.sqrt(beta_bl_ratio * beta_bl_ratio - 2 * beta_bl_ratio * np.cos(beta_r ) + 1))
        phi_beta_r_2  = np.arcsin(recip_beta_bl_ratio * np.sin(beta_r ) / np.sqrt(recip_beta_bl_ratio * recip_beta_bl_ratio - 2 * recip_beta_bl_ratio * np.cos(beta_r ) + 1))

        phi_betap_r_1 = np.arcsin(betap_bl_ratio * np.sin(betap_r) / np.sqrt(betap_bl_ratio * betap_bl_ratio - 2 * betap_bl_ratio * np.cos(betap_r) + 1))
        phi_betap_r_2 = np.arcsin(recip_beta_p_bl_ratio * np.sin(betap_r) / np.sqrt(recip_beta_p_bl_ratio * recip_beta_p_bl_ratio - 2 * recip_beta_p_bl_ratio * np.cos(betap_r) + 1))

        phi_r = np.array([phi_beta_r_1, phi_beta_r_2, phi_betap_r_1, phi_betap_r_2])

        X_angles_r = M_angles_r - phi_r
        M_dists = np.sqrt(np.sum(np.square(xy_M_pos), axis=1))
        denom_beta_1  = np.sqrt(beta_bl_ratio * beta_bl_ratio - 2 * beta_bl_ratio * np.cos(beta_r ) + 1)
        denom_beta_2  = np.sqrt(recip_beta_bl_ratio * recip_beta_bl_ratio - 2 * recip_beta_bl_ratio * np.cos(beta_r ) + 1)

        denom_betap_1 = np.sqrt(betap_bl_ratio * betap_bl_ratio - 2 * betap_bl_ratio * np.cos(betap_r) + 1)
        denom_betap_2 = np.sqrt(recip_beta_p_bl_ratio * recip_beta_p_bl_ratio - 2 * recip_beta_p_bl_ratio * np.cos(betap_r) + 1)
        denom = [denom_beta_1, denom_beta_2, denom_betap_1, denom_betap_2]
        X_mags = M_dists / denom

        # polar magnitudes
        X_equat_pos = np.array([
            X_mags * np.cos(X_angles_r),
            X_mags * np.sin(X_angles_r),
            np.zeros(shape=X_mags.shape),
        ]).T

        xy_atom_axial_vec = X_equat_pos[0, :] - M_pos[0, :]
        origin_atom_axial_vec = X_equat_pos[1, :] - M_pos[1, :]
        # normalize
        xy_mag     = np.linalg.norm(xy_atom_axial_vec)
        origin_mag = np.linalg.norm(origin_atom_axial_vec)
        xy_atom_axial_vec /= xy_mag
        origin_mag        /= origin_mag

        X_axial_displacements = np.array([
            [
                ax_bl * np.sin(theta_r) * xy_atom_axial_vec[0],
                ax_bl * np.sin(theta_r) * xy_atom_axial_vec[1],
                ax_bl * np.cos(theta_r)
            ],
            [
                ax_bl * np.sin(theta_r) * xy_atom_axial_vec[0],
                ax_bl * np.sin(theta_r) * xy_atom_axial_vec[1],
                -ax_bl * np.cos(theta_r)
            ],
            [
                ax_bl * np.sin(theta_r) * origin_atom_axial_vec[0],
                ax_bl * np.sin(theta_r) * origin_atom_axial_vec[1],
                ax_bl * np.cos(theta_r)
            ],
            [
                ax_bl * np.sin(theta_r) * origin_atom_axial_vec[0],
                ax_bl * np.sin(theta_r) * origin_atom_axial_vec[1],
                -ax_bl * np.cos(theta_r)
            ]
        ])
        X_axial_pos = np.repeat(M_pos, 2, axis=0) + X_axial_displacements
        X_pos = np.append(X_equat_pos, X_axial_pos, axis=0)

        return M_pos, X_pos
