"""Utility classes for generation of file I/O for Quantum Espresso and CIF files.
"""
from abc import ABC, abstractmethod
import re
import numpy as np
import copy
from perovskite_utils.structure import CoordinateModes, Structure, StructureAtom

BOHR_ANGSTROM = 0.5291772109


class NumberFormatter:
    """Formats number data into string representations, or vice-versa."""

    @staticmethod
    def get_constant_length(num_data, width, precision, alignment="right"):
        """Returns a string representation of number data with constant character
        length and constitent for positive/negative numbers.

        Args:
            num_data (iterable): number data to be converted into strings
            width (int): desired character length of the strings
            precision (int): number of decimal places in string representation of numbers
            alignment (str, optional): Alignment of number within string.
                Can be "left" or "right". Defaults to "right".

        Returns:
            iterable(str) : string data
        """
        assert alignment == "right" or "left", f"Invalid alignment of '{alignment}'"
        format_string = (
            f"""{{0: {">" if alignment == "right" else "<"}{width}.{precision}f}}"""
        )
        try:
            return list(map(format_string.format, num_data))
        except TypeError:
            return format_string.format(num_data)

    @staticmethod
    def string_to_float(str_data):
        """Converts string data into floats

        Args:
            str_data (iterable(str)): string data

        Returns:
            iterable(float): float data
        """
        try:
            return list(map(float, str_data))
        except TypeError:
            return float(str_data)


class FileReaderFactory:
    @staticmethod
    def create_file_reader(read_states):
        pass


class FileReaderState(ABC):
    """Abstract class for a state of the FileReader finite state machine."""

    def __init__(self, file_reader):
        if not isinstance(file_reader, FileReader):
            raise ValueError("file_reader argument is not of type FileReader")
        self._file_reader = file_reader

    @abstractmethod
    def enter(self):
        """Abstract method for behavior when state is entered."""

    @abstractmethod
    def execute(self):
        """Abstract method for behavior when state is active"""

    @abstractmethod
    def exit(self):
        """Abstract method for behavior when state is exitted."""


class PWscfInputReaderState(FileReaderState):
    _REGEX_NAMELIST = re.compile("^&")
    """Abstract class for a Quantum Espresso (QE) PWscf input reader state of a FileReader finite state machine.
    """

    def _switch_namelist_card(self, token):
        namelist_regex = self._REGEX_NAMELIST.match(token)
        lowered_token = token.lower()  # all QE reader states are lower case
        if namelist_regex is not None:
            self._file_reader.switch_state(lowered_token[1:])
        elif token.upper() == token and self._file_reader.has_state(lowered_token):
            self._file_reader.switch_state(lowered_token)
        else:
            return False

        return True


class PWscfInputIdleReaderState(PWscfInputReaderState):
    def enter(self):
        print("Entering idle state.")

    def execute(self):
        tokens = self._file_reader.current_str.split(" ")
        self._switch_namelist_card(tokens[0].strip())

    def exit(self):
        print("Exiting idle state.")


class ControlReaderState(PWscfInputReaderState):
    """Reader state of Quantum Espresso's (QE) &control namelist"""

    def enter(self):
        print("Entering control namelist state.")

    def execute(self):
        """Reads system namelist"""
        tokens = self._file_reader.current_str.split(" ")
        if self._switch_namelist_card(tokens[0].strip()):
            return
        elif tokens[0].strip() == "/":
            return

        field = tokens[2]
        field_value = tokens[-1].rstrip()
        if field == "prefix":
            # 1:-1 splice to remove the single quotes from the prefix
            self._file_reader.structure.name = field_value[1:-1]

    def exit(self):
        """Abstract method for behavior when state is exitted."""
        print("Exiting control namelist state.")


class SystemReaderState(PWscfInputReaderState):
    """Reader state of Quantum Espresso's (QE) &system namelist"""

    def enter(self):
        print("Entering system namelist state.")

    def execute(self):
        """Reads system namelist"""
        tokens = self._file_reader.current_str.split(" ")
        if self._switch_namelist_card(tokens[0].strip()):
            return
        elif tokens[0].strip() == "/":
            return

    def exit(self):
        """Abstract method for behavior when state is exitted."""
        print("Exiting system namelist state.")


class AtomicPositionsReaderState(PWscfInputReaderState):
    """Reader state of Quantum Espresso's (QE) ATOMIC_POSITIONS card"""

    def enter(self):
        print("Entering ATOMIC_POSITIONS card state.")
        tokens = self._file_reader.current_str.split(" ")
        if tokens[1].rstrip() == "crystal":
            self._file_reader.structure.coordinate_mode = CoordinateModes.FRACTIONAL

    def execute(self):
        tokens = self._file_reader.current_str.split(" ")
        if self._switch_namelist_card(tokens[0].strip()):
            return
        non_empty_tokens = list(filter(None, self._file_reader.current_str.split(" ")))
        label = non_empty_tokens[0]
        x, y, z = NumberFormatter.string_to_float(non_empty_tokens[1:4])
        self._file_reader.structure.atoms.append(StructureAtom([x, y, z], label))

    def exit(self):
        print("Exiting ATOMIC_POSITIONS card state.")


class CellParametersReaderState(PWscfInputReaderState):
    """Reader state of Quantum Espresso's (QE) CELL_PARAMETERS card"""

    def enter(self):
        print("Entering CELL_PARAMETERS card state.")

    def execute(self):
        """Reads system namelist"""
        tokens = self._file_reader.current_str.split(" ")
        if self._switch_namelist_card(tokens[0].strip()):
            return
        non_empty_tokens = list(filter(None, self._file_reader.current_str.split(" ")))
        lattice_vec = NumberFormatter.string_to_float(non_empty_tokens)
        self._file_reader.structure.lattice_vec.append(lattice_vec)

    def exit(self):
        """Abstract method for behavior when state is exitted."""
        print("Exiting CELL_PARAMETERS card state.")


class PWscfOutputReaderState(FileReaderState):
    """Abstract class for a Quantum Espresso (QE) PWscf calculation='relax' output reader state of a FileReader finite state machine."""


class PWscfCellParametersReaderState(PWscfOutputReaderState):
    def __init__(self, file_reader):
        super().__init__(file_reader)
        self._lattice_parameter = None
        self._units = None

    def enter(self):
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        self._units = non_empty_tokens[-1]
        self._lattice_parameter = float(non_empty_tokens[-2])
        if self._units == "a.u.":
            self._lattice_parameter *= BOHR_ANGSTROM

    def execute(self):
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        if len(non_empty_tokens) > 0:
            if non_empty_tokens[0] == "PseudoPot.":
                self._file_reader.switch_state("idle")
                return

            if " ".join(non_empty_tokens[0:3]) == "number of electrons":
                self._file_reader.num_electrons = float(non_empty_tokens[-1])
            elif non_empty_tokens[0] in ["a(1)", "a(2)", "a(3)"]:
                lattice_vec = NumberFormatter.string_to_float(non_empty_tokens[3:6])
                lattice_vec = list(
                    map(
                        lambda component: component * self._lattice_parameter,
                        lattice_vec,
                    )
                )

                self._file_reader.structure.lattice_vec.append(lattice_vec)

    def exit(self):
        pass


class RelaxIdleReaderState(PWscfOutputReaderState):
    def enter(self):
        pass

    def execute(self):
        if "Writing config-only to output data dir" in self._file_reader.current_str:
            self._file_reader.switch_state("name")

        elif "lattice parameter (alat)" in self._file_reader.current_str:
            self._file_reader.switch_state("cell_parameters")

        elif "End of BFGS Geometry Optimization" in self._file_reader.current_str:
            self._file_reader.switch_state("positions")

    def exit(self):
        pass


class RelaxNameReaderState(PWscfOutputReaderState):
    def enter(self):
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        name = ".".join(non_empty_tokens[-1].split("/")[-2].split(".")[:-1])
        self._file_reader.structure.name = name
        self._file_reader.switch_state("idle")

    def execute(self):
        pass

    def exit(self):
        pass


class RelaxPositionsReaderState(PWscfOutputReaderState):
    def __init__(self, file_reader):
        super().__init__(file_reader)
        self._started_reading = False

    def enter(self):
        pass

    def execute(self):
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        if len(non_empty_tokens) == 0:
            if self._started_reading:
                self._file_reader.switch_state("idle")
                return
            else:
                self._started_reading = True
                return

        if not self._started_reading:
            return

        if non_empty_tokens[0] == "ATOMIC_POSITIONS":
            if non_empty_tokens[1] == "(crystal)":
                self._file_reader.structure.coordinate_mode = CoordinateModes.FRACTIONAL

            # TODO: implement for cartesian coordinate case

        else:
            label = non_empty_tokens[0]
            x, y, z = NumberFormatter.string_to_float(non_empty_tokens[1:4])
            self._file_reader.structure.atoms.append(StructureAtom([x, y, z], label))

    def exit(self):
        pass


class SCFIdleReaderState(PWscfOutputReaderState):
    def enter(self):
        pass

    def execute(self):
        if "Writing all to output data dir" in self._file_reader.current_str:
            self._file_reader.switch_state("name")
        elif "lattice parameter (alat)" in self._file_reader.current_str:
            self._file_reader.switch_state("cell_parameters")

    def exit(self):
        pass


class SCFNameReaderState(PWscfOutputReaderState):
    def enter(self):
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        name = ".".join(non_empty_tokens[-1].split("/")[-2].split(".")[:-1])
        self._file_reader.structure.name = name
        self._file_reader.switch_state("idle")

    def execute(self):
        pass

    def exit(self):
        pass


class CIFReaderState(FileReaderState):
    """Abstract class for a CIF file reader state of a FileReader finite state machine."""


class CIFIdleReaderState(CIFReaderState):
    def enter(self):
        pass

    def execute(self):
        if "data_" in self._file_reader.current_str:
            self._file_reader.switch_state("name")

        elif (
            len(self._file_reader.current_str.split("_")) > 1
            and "cell" == self._file_reader.current_str.split("_")[1]
        ):
            self._file_reader.switch_state("cell_parameters")

        elif "loop_" in self._file_reader.current_str:
            self._file_reader.switch_state("loop")

    def exit(self):
        pass


class CIFNameReaderState(CIFReaderState):
    def enter(self):
        name = "_".join(self._file_reader.current_str.strip().split("_")[1:])
        self._file_reader.structure.name = name
        self._file_reader.switch_state("idle")

    def execute(self):
        pass

    def exit(self):
        pass


class CIFLoopReaderState(CIFReaderState):
    def __init__(self, file_reader):
        super().__init__(file_reader)
        self._field_order = []

    def enter(self):
        self._field_order = []

    def execute(self):
        field_tokens = list(self._file_reader.current_str.strip().split("_"))
        if len(field_tokens) < 2:
            if all(
                map(
                    lambda field: field in self._field_order,
                    [
                        "_atom_site_type_symbol",
                        "_atom_site_fract_x",
                        "_atom_site_fract_y",
                        "_atom_site_fract_z",
                    ],
                )
            ):
                self._file_reader.pos_field_order = copy.deepcopy(self._field_order)
                self._file_reader.switch_state("positions")
            else:
                self._file_reader.switch_state("idle")
        self._field_order.append(self._file_reader.current_str.strip())

    def exit(self):
        pass


class CIFCellParametersReaderState(CIFReaderState):
    def __init__(self, file_reader):
        super().__init__(file_reader)
        self._cell_lengths = dict()
        self._cell_angles = dict()

    def enter(self):
        self.execute()

    def execute(self):
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        if len(non_empty_tokens) == 0:
            self._file_reader.switch_state("idle")
            return
        cell_param_type, param_var = non_empty_tokens[0].split("_")[-2:]
        if cell_param_type == "length":
            self._cell_lengths[param_var] = float(non_empty_tokens[-1].split("(")[0])
        elif cell_param_type == "angle":
            self._cell_angles[param_var] = np.deg2rad(
                float(non_empty_tokens[-1].split("(")[0])
            )

    def exit(self):
        if not all(map(lambda var: var in ["a", "b", "c"], self._cell_lengths.keys())):
            raise RuntimeError(
                f"Not enough cell lengths read, only {self._cell_lengths.keys()}. \
            Need a, b, c."
            )
        elif not all(
            map(lambda var: var in ["alpha", "beta", "gamma"], self._cell_angles.keys())
        ):
            raise RuntimeError(
                f"Not enough cell angles read, only {self._cell_angles.keys()}. \
            Need alpha, beta, gamma."
            )
        self._file_reader.structure.set_lattice_vec_from_cell_params(
            self._cell_lengths["a"],
            self._cell_lengths["b"],
            self._cell_lengths["c"],
            self._cell_angles["alpha"],
            self._cell_angles["beta"],
            self._cell_angles["gamma"],
        )


class CIFPositionsReaderState(CIFReaderState):
    def enter(self):
        self.execute()

    def execute(self):
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        if len(non_empty_tokens) == 0:
            self._file_reader.switch_state("idle")
            return

        # TODO: add option for Cartesian coordinates
        self._file_reader.structure.coordinate_mode = CoordinateModes.FRACTIONAL
        fract_pos = dict()
        label = None
        for i, field in enumerate(self._file_reader.pos_field_order):
            field_tail = "_".join(field.split("_")[3:])
            if field_tail == "fract_x":
                fract_pos["x"] = float(non_empty_tokens[i].split("(")[0])
            elif field_tail == "fract_y":
                fract_pos["y"] = float(non_empty_tokens[i].split("(")[0])
            elif field_tail == "fract_z":
                fract_pos["z"] = float(non_empty_tokens[i].split("(")[0])
            elif field_tail == "type_symbol":
                label = non_empty_tokens[i]

        self._file_reader.structure.atoms.append(
            StructureAtom([fract_pos["x"], fract_pos["y"], fract_pos["z"]], label)
        )

    def exit(self):
        pass


class FileReader(ABC):
    """An input file reader that implements a finite state machine"""

    def __init__(self, reader_states, start_state_name, read_path):
        self._reader_states = reader_states
        if not all(
            map(lambda s: isinstance(s, FileReaderState), self._reader_states.values())
        ):
            raise ValueError(
                f"read_states argument '{reader_states}' contains classes not of \
                             type FileReaderState."
            )
        self._current_str = (
            ""  # current string being proceessed. Usually this occurs line by line.
        )
        self._read_path = read_path
        self.current_state = self._reader_states[start_state_name]

    @abstractmethod
    def read_file(self):
        """Abstract method for reading file from read_path"""

    @abstractmethod
    def feed(self, str_data):
        """Abstract method for feeding a new line that results in an update to the FSM

        Args:
            line_str (str): str of line from the file
        """

    @property
    def current_str(self):
        return self._current_str

    @property
    def read_path(self):
        return self._read_path

    def switch_state(self, new_state_name):
        self.current_state.exit()
        self.current_state = self._reader_states[new_state_name]
        self.current_state.enter()

    def execute_state(self):
        self.current_state.execute()

    def has_state(self, state):
        return state in self._reader_states.keys()


class StructureFileReader(FileReader):
    """An abstract finite state machine (FSM) for reading structure input files"""

    def __init__(self, reader_states, start_state_name, read_path):
        super().__init__(reader_states, start_state_name, read_path)
        self.structure = Structure.create_empty_structure()

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, new_structure):
        if not isinstance(new_structure, Structure):
            raise ValueError("new_structure argument is not of type Structure.")

        self._structure = new_structure


class PWscfInputReader(StructureFileReader):
    """A Quantum Espresso (QE) PWscf input file structure reader"""

    def __init__(self, read_path):
        super().__init__(
            reader_states={
                "idle": PWscfInputIdleReaderState(self),
                "control": ControlReaderState(self),
                "system": SystemReaderState(self),
                "electrons": PWscfInputIdleReaderState(self),
                "ions": PWscfInputIdleReaderState(self),
                "atomic_species": PWscfInputIdleReaderState(self),
                "atomic_positions": AtomicPositionsReaderState(self),
                "k_points": PWscfInputIdleReaderState(self),
                "cell_parameters": CellParametersReaderState(self),
            },
            start_state_name="idle",
            read_path=read_path,
        )
        self.structure.struct_type = "input"

    def read_file(self):
        with open(self.read_path, "r", encoding="ascii") as qe_file:
            for line in qe_file:
                self.feed(line)

    def feed(self, str_data):
        self._current_str = str_data
        self.execute_state()


class PWscfOutputReader(StructureFileReader):
    """Abstract file reader class for Quantum Espresso (QE) PWscf outputs"""

    def __init__(self, reader_states, start_state_name, read_path):
        super().__init__(reader_states, start_state_name, read_path)
        self.num_electrons = None


class PWscfRelaxOutputReader(PWscfOutputReader):
    """A Quantum Espresso (QE) PWscf 'relax' calculation output file structure reader"""

    def __init__(self, read_path):
        super().__init__(
            reader_states={
                "idle": RelaxIdleReaderState(self),
                "name": RelaxNameReaderState(self),
                "positions": RelaxPositionsReaderState(self),
                "cell_parameters": PWscfCellParametersReaderState(self),
            },
            start_state_name="idle",
            read_path=read_path,
        )
        self.structure.struct_type = "relaxed"

    def read_file(self):
        with open(self.read_path, "r", encoding="ascii") as qe_file:
            for line in qe_file:
                self.feed(line)

    def feed(self, str_data):
        self._current_str = str_data
        self.execute_state()


class PWscfSCFOutputReader(PWscfOutputReader):
    """A Quantum Espresso (QE) PWscf 'scf' calculation output file structure reader"""

    def __init__(self, read_path):
        super().__init__(
            reader_states={
                "idle": SCFIdleReaderState(self),
                "name": SCFNameReaderState(self),
                "cell_parameters": PWscfCellParametersReaderState(self),
            },
            start_state_name="idle",
            read_path=read_path,
        )
        self.structure.struct_type = "scf_input"

    def read_file(self):
        with open(self.read_path, "r", encoding="ascii") as qe_file:
            for line in qe_file:
                self.feed(line)

    def feed(self, str_data):
        self._current_str = str_data
        self.execute_state()


class CIFReader(StructureFileReader):
    """A CIF file structure reader"""

    def __init__(self, read_path):
        super().__init__(
            reader_states={
                "idle": CIFIdleReaderState(self),
                "name": CIFNameReaderState(self),
                "loop": CIFLoopReaderState(self),
                "cell_parameters": CIFCellParametersReaderState(self),
                "positions": CIFPositionsReaderState(self),
            },
            start_state_name="idle",
            read_path=read_path,
        )
        self.structure.struct_type = "input"
        self.pos_field_order = []

    def read_file(self):
        with open(self.read_path, "r", encoding="utf-8") as cif_file:
            for line in cif_file:
                self.feed(line)

    def feed(self, str_data):
        self._current_str = str_data
        self.execute_state()


class FileWriter(ABC):
    """An abstract FileWriter class"""

    def __init__(self, write_path, encoding):
        self._write_path = write_path
        self._encoding = encoding
        self._file_str = ""

    def write_file(self):
        with open(self._write_path, mode="w", encoding=self._encoding) as write_f:
            write_f.write(self._file_str)


class StructureFileWriter(FileWriter):
    """An output file writer that writes structure data"""

    def __init__(self, write_path, encoding, structure):
        super().__init__(write_path, encoding)
        self._structure = structure

    @property
    def structure(self):
        return self._structure


class CIFFileWriter(StructureFileWriter):
    """A CIF structure file writer"""

    def __init__(self, write_path, structure):
        if structure.coordinate_mode == CoordinateModes.FRACTIONAL:
            super().__init__(write_path, "utf-8", structure)
        elif structure.coordinate_mode == CoordinateModes.CARTESIAN:
            super().__init__(write_path, "utf-8", structure.to_fract())

        self._cell_angles = []
        self._cell_volume = None
        self._cell_lengths = []
        self._numbered_labels = []
        self._file_str = self.__serialize_structure()

    def __set_crystal_params(self):
        # convert into crystallographical parameters
        cell_params = self.structure.get_cell_params_from_lattice_vec()
        self._cell_lengths = cell_params[:3]
        self._cell_angles = cell_params[3:]
        self._cell_volume = self.structure.get_cell_volume()

    def __set_numbered_labels(self):
        atom_labels = [atom.label for atom in self.structure.atoms]
        atom_counts = dict()
        for label in atom_labels:
            if label not in atom_counts.keys():
                atom_counts[label] = 1
            else:
                atom_counts[label] += 1
            self._numbered_labels.append(f"{label + str(atom_counts[label]): <4}")

    def __serialize_structure(self):
        if (
            len(self._cell_angles) == 0
            or self._cell_volume is None
            or len(self._cell_lengths) == 0
        ):
            self.__set_crystal_params()

        if len(self._numbered_labels) == 0:
            self.__set_numbered_labels()

        # CIF data line
        data_str = f"data_{self._structure.name}\n"

        # cell parameeters
        formatted_cell_lengths = NumberFormatter.get_constant_length(
            self._cell_lengths, 15, 6
        )
        formatted_cell_angles = NumberFormatter.get_constant_length(
            self._cell_angles, 15, 6
        )
        formatted_cell_volume = NumberFormatter.get_constant_length(
            self._cell_volume, 15, 6
        )
        cell_str = f"""_cell_length_a    {formatted_cell_lengths[0]}
_cell_length_b    {formatted_cell_lengths[1]}
_cell_length_c    {formatted_cell_lengths[2]}
_cell_angle_alpha {formatted_cell_angles[0]}
_cell_angle_beta  {formatted_cell_angles[1]}
_cell_angle_gamma {formatted_cell_angles[2]}
_cell_volume      {formatted_cell_volume}\n"""

        # atomic position loop
        formatted_pos = [
            NumberFormatter.get_constant_length(atom.pos, 10, 6)
            for atom in self.structure.atoms
        ]
        formatted_atoms = ""
        for i, pos in enumerate(formatted_pos):
            label = self.structure.atoms[i].label
            formatted_atoms += (
                f"    {self._numbered_labels[i]}{pos[0]}{pos[1]}{pos[2]} {label}\n"
            )

        pos_str = (
            """loop_
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_type_symbol\n"""
            + formatted_atoms
        )

        return "\n".join([data_str, cell_str, pos_str])


class PWscfCalculation:
    def __init__(
        self,
        structure,
        calculation,
        pseudo_dir,
        ecut_wfc,
        ecut_rho,
        occupations,
        pseudos,
        forces,
        k_points_mode,
        k_points,
        out_prefix="out_",
        verbosity="low",
        etot_conv_thr=None,
        forc_conv_thr=None,
        restart_mode=None,
        max_seconds=None,
        ibrav=0,
        nosym=None,
        noncolin=None,
        lspinorb=None,
        nbnd=None,
        conv_thr=None,
        mixing_beta=None,
        masses=None,
    ):
        self.structure = structure
        self.unique_labels = []
        for atom in self.structure.atoms:
            if atom.label not in self.unique_labels:
                self.unique_labels.append(atom.label)

        self.prefix = self.structure.name
        self.calculation = calculation
        self.pseudo_dir = pseudo_dir
        self.pseudos = pseudos
        self.forces = forces
        if not len(self.forces) == len(self.structure.atoms):
            raise ValueError(
                f"len of forces ({len(self.forces)} not equal to len of structure.atoms {len(self.structure.atoms)})"
            )
        if not sorted(self.unique_labels) == sorted(self.pseudos.keys()):
            raise ValueError(
                f"keys of pseudos, {self.pseudos.keys()} must match the unique labels of structure, {self.unique_labels}."
            )
        self.ecut_wfc = ecut_wfc
        self.ecut_rho = ecut_rho
        self.occupations = occupations
        self.k_points_mode = k_points_mode
        self.k_points = k_points
        self.out_prefix = out_prefix
        self.verbosity = verbosity
        self.etot_conv_thr = etot_conv_thr
        self.forc_conv_thr = forc_conv_thr
        self.restart_mode = restart_mode
        self.max_seconds = max_seconds
        self.ibrav = ibrav
        self.nat = len(self.structure.atoms)
        self.ntyp = len(self.unique_labels)
        self.nosym = nosym
        self.noncolin = noncolin
        self.lspinorb = lspinorb
        self.nbnd = nbnd
        self.conv_thr = conv_thr
        self.mixing_beta = mixing_beta
        self.masses = masses
        if self.masses is None:
            for label in self.unique_labels:
                self.masses[label] = 1.0
        else:
            if not sorted(self.unique_labels) == sorted(self.masses.keys()):
                raise ValueError(
                    f"keys of masses, {self.masses.keys()} must match the unique labels of structure, {self.unique_labels}."
                )

        if self.ibrav != 0 and self.structure.lattice_vec is not None:
            raise ValueError(
                f"Input ibrav is {self.ibrav} but lattice vectors are specified in {self.structure.lattice_vec}. \
                             Must use either ibrav != 0 or specify lattice vectors, but not both."
            )
        elif self.ibrav != 0:
            # TODO: implement ibrav != 0
            raise NotImplementedError("ibrav != 0 is not implemented")


class PWscfInputWriter(StructureFileWriter):
    """A PWscf input file writer"""

    def __init__(self, write_path, structure, pw_scf_calc):
        super().__init__(write_path, "ascii", structure)
        self.pw_scf_calc = pw_scf_calc
        self._namelists = None
        self._cards = None
        self.__format_namelists()
        self.__format_cards()
        self._file_str = self._namelists + self._cards

    def __format_sci_not(self, num):
        sci_not = f"{num:.10E}"
        if not isinstance(num, float) or "E" not in sci_not:
            raise ValueError("num must be a float in scientific notation")

        return sci_not.replace("E", "d")

    def __format_bool(self, bool_val):
        if not isinstance(bool_val, bool):
            raise ValueError("bool_val must be a boolean.")

        return ".true." if bool_val else ".false."

    def __format_namelists(self):
        if self._namelists is None:
            control_str = f"""&CONTROL
  calculation = '{self.pw_scf_calc.calculation}'
  outdir = './out_{self.pw_scf_calc.prefix}/'
  prefix = '{self.pw_scf_calc.prefix}'
  pseudo_dir = './pseudo/'
  verbosity = '{self.pw_scf_calc.verbosity}'
"""
            if self.pw_scf_calc.etot_conv_thr is not None:
                control_str += f"  etot_conv_thr = {self.__format_sci_not(self.pw_scf_calc.etot_conv_thr)}\n"
            if self.pw_scf_calc.forc_conv_thr is not None:
                control_str += f"  forc_conv_thr = {self.__format_sci_not(self.pw_scf_calc.forc_conv_thr)}\n"
            if self.pw_scf_calc.restart_mode is not None:
                control_str += f"  restart_mode = {self.pw_scf_calc.restart_mode}\n"
            if self.pw_scf_calc.max_seconds is not None:
                control_str += f"  max_seconds = {self.pw_scf_calc.max_seconds}\n"
            control_str += "/\n"

            system_str = f"""&SYSTEM
  ecutrho = {self.__format_sci_not(self.pw_scf_calc.ecut_rho)}
  ecutwfc = {self.__format_sci_not(self.pw_scf_calc.ecut_wfc)}
  ibrav = {self.pw_scf_calc.ibrav:.0f}
  nat = {self.pw_scf_calc.nat}
  ntyp = {self.pw_scf_calc.ntyp}
  occupations = '{self.pw_scf_calc.occupations}'
"""
            if self.pw_scf_calc.nosym is not None:
                system_str += (
                    f"  nosym = {self.__format_bool(self.pw_scf_calc.nosym)}\n"
                )
            if self.pw_scf_calc.noncolin is not None:
                system_str += (
                    f"  noncolin = {self.__format_bool(self.pw_scf_calc.noncolin)}\n"
                )
            if self.pw_scf_calc.lspinorb is not None:
                system_str += (
                    f"  lspinorb = {self.__format_bool(self.pw_scf_calc.lspinorb)}\n"
                )
            if self.pw_scf_calc.nbnd is not None:
                system_str += f"  nbnd = {self.pw_scf_calc.nbnd:.0f}\n"
            system_str += "/\n"

            electrons_str = "&ELECTRONS\n"
            if self.pw_scf_calc.conv_thr is not None:
                electrons_str += (
                    f"  conv_thr = {self.__format_sci_not(self.pw_scf_calc.conv_thr)}\n"
                )
            if self.pw_scf_calc.mixing_beta is not None:
                electrons_str += f"  noncolin = {self.__format_sci_not(self.pw_scf_calc.mixing_beta)}\n"
            electrons_str += "/\n"

            ions_str = (
                "&IONS\n/\n"
                if self.pw_scf_calc.calculation in ["relax", "vc-relax"]
                else ""
            )

            self._namelists = control_str + system_str + electrons_str + ions_str

    def __format_cards(self):
        atomic_species_str = "ATOMIC_SPECIES\n"
        for label in self.pw_scf_calc.unique_labels:
            atomic_species_str += f"{label: <3}{NumberFormatter.get_constant_length(self.pw_scf_calc.masses[label], 10, 6)} {self.pw_scf_calc.pseudos[label]}\n"

        atomic_positions_str = "ATOMIC_POSITIONS "
        if self.structure.coordinate_mode == CoordinateModes.FRACTIONAL:
            atomic_positions_str += "crystal\n"
        elif self.structure.coordinate_mode == CoordinateModes.CARTESIAN:
            atomic_positions_str += "angstrom\n"
        else:
            raise ValueError(
                f"structure.coordinate_mode has invalid value of '{self.structure.coordinate_mode}'"
            )

        for i, atom in enumerate(self.structure.atoms):
            atomic_positions_str += f"{atom.label: <3}\
{''.join(NumberFormatter.get_constant_length(atom.pos, 15, 10))} \
{''.join(NumberFormatter.get_constant_length(self.pw_scf_calc.forces[i], 2, 0))}\n"

        k_points_str = (
            f"K_POINTS {self.pw_scf_calc.k_points_mode}\n"
            + self.pw_scf_calc.k_points
            + "\n"
        )

        cell_parameters_str = f"""CELL_PARAMETERS angstrom
    {''.join(NumberFormatter.get_constant_length(self.structure.lattice_vec[0], 14, 10))}
    {''.join(NumberFormatter.get_constant_length(self.structure.lattice_vec[1], 14, 10))}
    {''.join(NumberFormatter.get_constant_length(self.structure.lattice_vec[2], 14, 10))}
"""

        self._cards = (
            atomic_species_str
            + atomic_positions_str
            + k_points_str
            + cell_parameters_str
        )
