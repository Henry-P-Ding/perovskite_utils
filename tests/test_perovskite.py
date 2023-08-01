import unittest
import numpy as np
from perovskite_utils.perovskite_utils.file_read_write import PWscfInputReader, CIFFileWriter
from perovskite_utils.perovskite_utils.perovskite_generator import PerovskiteStructure


class TestPerovskiteStructure(unittest.TestCase):
    def test_perovskite_structure(self):
        qe_reader = PWscfInputReader("perovskite_utils/tests/SMBA2PbI4_150.000_150.000.relax.in")
        qe_reader.read_file()
        comparison_file_writer = CIFFileWriter("perovskite_utils/tests/comparison_structure.cif", qe_reader.structure)
        comparison_file_writer.write_file()

        a, b, c = 8.431827, 8.76139, 27.07124
        x, y = 0.5 * a, 0.5 * b
        trans_x = -0.29590 * a
        trans_y = 0.01328 * b
        trans_z = -(c / 2)
        theta_d = (180 - 167.1194) / 2
        ax_bl = 3.12991
        final_uc_trans = np.array([
            0.14795 * a,
            0.24336 * b,
            0.57472 * c
        ])
        pv_structure = PerovskiteStructure.from_param(
            name="SMBA2PbI4_150.000_150.000",
            lattice_vec=np.array([
                [a, 0, 0],
                [0, b, 0],
                [0, 0, c]
            ]),
            cart_M_disp=np.array([x, y]),
            cart_layer_trans=np.array([trans_x, trans_y, trans_z]),
            theta_d=theta_d,
            ax_bl=ax_bl,
            beta_bl_ratio=1,
            betap_bl_ratio=1,
            beta_d=150,
            betap_d=150,
            M_label = "Pb",
            X_label = "I"
        )
        pv_structure = pv_structure.translate(final_uc_trans)
        pv_structure = pv_structure.to_fract()
        cif_writer = CIFFileWriter("perovskite_utils/tests/gen_structure.cif", pv_structure)
        cif_writer.write_file()

if __name__ == '__main__':
    unittest.main()