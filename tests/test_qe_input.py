import unittest
from perovskite_utils.perovskite_utils.file_read_write import PWscfInputReader, CIFFileWriter, PWscfRelaxOutputReader, CIFReader, PWscfCalculation, PWscfInputWriter


class TestSystemNamelist(unittest.TestCase):
    def test_write_to_cif(self):
        qe_reader = PWscfInputReader("perovskite_utils/tests/test_qe.relax.in")
        qe_reader.read_file()
        self.assertEqual(qe_reader.structure.name, "SMBA2PbI4_150.000_150.000")
        self.assertEqual(qe_reader.structure.lattice_vec, [
            [ 8.4318200000,  0.0000000000,  0.0000000000],
            [-0.0000810450,  8.7613899996,  0.0000000000],
            [-0.0034538454, -0.0116041962, 27.0712372926]
        ])
        cif_writer = CIFFileWriter("perovskite_utils/tests/test_qe.cif", qe_reader.structure)
        cif_writer.write_file()

    def test_relax_read(self):
        qe_reader = PWscfRelaxOutputReader("perovskite_utils/tests/test_relax_output.out")
        qe_reader.read_file()
        cif_writer = CIFFileWriter("perovskite_utils/tests/test_relax_output.cif", qe_reader.structure)
        cif_writer.write_file()

    def test_cif_read(self):
        cif_reader = CIFReader("perovskite_utils/tests/test_qe.cif")
        cif_reader.read_file()
        cif_writer = CIFFileWriter("perovskite_utils/tests/test_qe_read_then_write.cif", cif_reader.structure)
        cif_writer.write_file()

    def test_relax_input(self):
        cif_reader = CIFReader("perovskite_utils/tests/test_qe.cif")
        cif_reader.read_file()
        calculation = PWscfCalculation(
            structure=cif_reader.structure,
            calculation="relax",
            pseudo_dir="./pseudo/",
            etot_conv_thr=1.0e-5,
            forc_conv_thr=1.0e-4,
            max_seconds=13800,
            ecut_rho=4.0e2,
            ecut_wfc=5.0e1,
            ibrav=0,
            noncolin=True,
            lspinorb=True,
            occupations="fixed",
            nbnd=570,
            conv_thr=3.76e-8,
            mixing_beta=5e-1,
            masses={
                "C": 12.0107,
                "H": 1.00794,
                "I": 126.90447,
                "N": 14.0067,
                "Pb": 207.2
            },
            pseudos={
                "C" : "C.pbesol-n-kjpaw_psl.1.0.0.UPF",
                "H" : "H.pbesol-rrkjus_psl.1.0.0.UPF",
                "I" : "I.pbesol-n-kjpaw_psl.0.2.UPF",
                "N" : "N.pbesol-theos.UPF",
                "Pb": "Pb.pbesol-FR.upf"
            },
            forces=[[0, 0, 1]] * 8 + [[1, 1, 1]] * 4 + [[0, 0, 1]] * 4 + [[1, 1, 1]] * 172,
            k_points_mode="automatic",
            k_points="6 6 2 0 0 0"
        )
        qe_writer = PWscfInputWriter("perovskite_utils/tests/qe_written.relax.in", cif_reader.structure, calculation)
        qe_writer.write_file()

if __name__ == '__main__':
    unittest.main()