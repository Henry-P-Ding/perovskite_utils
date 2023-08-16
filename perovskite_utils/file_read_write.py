"""Utility classes for generation of file I/O for first-principles calculations.
"""
import copy
import re
from abc import ABC, abstractmethod
from typing import Iterable, Dict

import numpy as np
import scipy.constants

from perovskite_utils.structure import (
    CoordinateModes,
    Structure,
    StructureAtom,
)

BOHR_ANGSTROM = scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom


class NumberFormatter:
    """Formats number data into string representations, or vice-versa."""

    @staticmethod
    def get_constant_length(
            num_data: float | Iterable[float],
            width: int,
            precision: int,
            alignment: str = "right",
    ) -> str | Iterable[str]:
        """Returns a formatted string representation of number data

        Number data is formatted with constant character length, consistent for positive/negative numbers. num_data can either be a single float or a collection of floats.

        :param num_data: number data to be converted
        :type num_data: float | Iterable[float]
        :param width: character length of each formatted number
        :type width: int
        :param precision: number of decimal places for the formatted numbers
        :type precision: int
        :param alignment: formatted position of number relative to the left and right ends of the string, defaults to "right"
        :type alignment: str, optional
        :return: formatted string representation of number data
        :rtype: str | Iterable[str]
        """

        if alignment not in ["right", "left"]:
            raise ValueError(f"Invalid alignment of '{alignment}'")
        format_string = (
            f"""{{0: {">" if alignment == "right" else "<"}{width}.{precision}f}}"""
        )
        try:
            return list(map(format_string.format, num_data))
        except TypeError:
            return format_string.format(num_data)

    @staticmethod
    def string_to_float(str_data: str | Iterable[str]) -> float | Iterable[float]:
        """Converts string data into float data

        :param str_data: string data to be converted
        :type str_data: str | Iterable[str]
        :return: float data converted from string data
        :rtype: float | Iterable[float]
        """

        try:
            return list(map(float, str_data))
        except TypeError:
            return float(str_data)


class FileReaderState(ABC):
    """
    Abstract base class for a state of a :class:`FileReader` finite state machine.

    :ivar _file_reader: stores the :class:`~FileReader` finite state machine this state is associated with
    :vartype _file_reader: FileReader
    """

    def __init__(self, file_reader: "FileReader") -> None:
        """:class:`~FileReaderState` constructor

        :param file_reader: the :class:`~FileReader` finite state machine this state is associated with.
        :type file_reader: :class:`~FileReader`
        :raises ValueError: when the file_reader argument is not of type :class:`~FileReader`
        """
        if not isinstance(file_reader, FileReader):
            raise ValueError("file_reader argument is not of type FileReader")
        self._file_reader = file_reader

    @abstractmethod
    def enter(self) -> None:
        """Abstract method for behavior when state is entered."""

    @abstractmethod
    def execute(self) -> None:
        """Abstract method for behavior when state is active"""

    @abstractmethod
    def exit(self) -> None:
        """Abstract method for behavior when state is exited."""


class PWscfInputReaderState(FileReaderState, ABC):
    # TODO: make this docstring summary more concise like the PWscfInputIdleReaderState class
    """Abstract base class for a Quantum Espresso (QE) PWscf input reader state of a :class:`FileReader` finite state
    machine."""
    _REGEX_NAMELIST = re.compile("^&")  # regex for namelist titles

    def _switch_namelist_card(self, token: str) -> bool:
        """Switches to a namelist or card state based on an input string token

        :param token: input string token
        :type token: str
        :return: True if state is successfully switched, False otherwise
        :rtype: bool
        """
        namelist_regex = self._REGEX_NAMELIST.match(token)
        lowered_token = token.lower()  # all QE reader states are lower case
        if namelist_regex is not None:
            self._file_reader.switch_state(
                lowered_token[1:]
            )  # selects [1:] substring to remove the leading '&' character
        elif token.upper() == token and self._file_reader.has_state(lowered_token):
            self._file_reader.switch_state(lowered_token)
        else:
            return False

        return True


class PWscfInputIdleReaderState(PWscfInputReaderState):
    """PWscf input reader idle helper state for transitioning between namelist or card states"""

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        """Swithces to a corresponding namelist or card state"""
        tokens = self._file_reader.current_str.split(" ")
        self._switch_namelist_card(tokens[0].strip())

    def exit(self) -> None:
        pass


class ControlReaderState(PWscfInputReaderState):
    """PWscf input reader state for the &control namelist"""

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        """Reads data from &control namelist"""
        tokens = self._file_reader.current_str.split(" ")

        # check if at the beginning of a new namelist or at the end of a namelist, exiting if necessary
        if self._switch_namelist_card(tokens[0].strip()):
            return
        elif (
                tokens[0].strip() == "/"
        ):  # end of a namelist is indicated by a '/' character
            return

        field = tokens[2]  # PWscf input field
        field_value = tokens[-1].rstrip()
        if field == "prefix":
            # 1:-1 splice to remove the single quotes from the prefix
            self._file_reader.structure.name = field_value[1:-1]

    def exit(self) -> None:
        pass


class SystemReaderState(PWscfInputReaderState):
    """PWscf input reader state for the &system namelist"""

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        """Reads data from &system namelist"""
        tokens = self._file_reader.current_str.split(" ")

        # check if at the beginning of a new namelist or at the end of a namelist, exiting if necessary
        if self._switch_namelist_card(tokens[0].strip()):
            return
        elif tokens[0].strip() == "/":
            return

    def exit(self) -> None:
        pass


class AtomicPositionsReaderState(PWscfInputReaderState):
    """PWscf input reader state for the ATOMIC_POSITIONS card"""

    def enter(self) -> None:
        tokens = self._file_reader.current_str.split(" ")
        # check if input structure is specified with fractional or cartesian coordinates
        if "crystal" in tokens[1].rstrip():
            self._file_reader.structure.coordinate_mode = CoordinateModes.FRACTIONAL

    def execute(self) -> None:
        """Reads data from the ATOMIC_POSITIONS card"""
        tokens = self._file_reader.current_str.split(" ")
        if self._switch_namelist_card(tokens[0].strip()):
            return
        non_empty_tokens = list(filter(None, self._file_reader.current_str.split(" ")))
        # atom label, corresponding to a pseudopotential
        label = non_empty_tokens[0]
        # atomic position coordinates, either fractional or cartesian
        x, y, z = NumberFormatter.string_to_float(non_empty_tokens[1:4])
        self._file_reader.structure.atoms.append(StructureAtom([x, y, z], label))

    def exit(self) -> None:
        pass


class CellParametersReaderState(PWscfInputReaderState):
    """PWscf input reader state for the CELL_PARAMETERS card"""

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        """Reads data from the CELL_PARAMETERS card"""
        tokens = self._file_reader.current_str.split(" ")
        if self._switch_namelist_card(tokens[0].strip()):
            return
        non_empty_tokens = list(filter(None, self._file_reader.current_str.split(" ")))
        lattice_vec = NumberFormatter.string_to_float(non_empty_tokens)
        self._file_reader.structure.lattice_vec.append(lattice_vec)

    def exit(self) -> None:
        pass


class PWscfOutputReaderState(FileReaderState):
    """Abstract base class for a Quantum Espresso (QE) PWscf calculation='relax' output reader
    state of a :class:`FileReader` finite state machine."""


class PWscfCellParametersReaderState(PWscfOutputReaderState):
    """
    PWscf output reader state for extracting structure data

    Extracts lattice parameter, unit cell lattice vectors, reciprocal lattice vectors, and
    number of electrons.

    :ivar _lattice_parameter: the "a" unit cell lattice parameter
    :vartype _lattice_parameter: float
    :ivar _units: the length unit used by lattice vectors
    :vartype _units: str
    """

    def __init__(self, file_reader: "FileReader") -> None:
        """:class:`PWscfCellParametersReaderState` constructor

        :param file_reader: the :class:`~FileReader` finite state machine this state is associated with.
        :type file_reader: FileReader
        """
        super().__init__(file_reader)
        self._lattice_parameter = None  # "a" unit cell lattice parameter
        self._units = None  # unit of lattice vectors

    def enter(self) -> None:
        """
        Extracts the lattice parameter in angstroms

        The lattice parameter and all cell parameters are always expressed in angstroms. Since the idle helper state
        switches to :class:`~PWscfCellParametersReaderState` state on the line containing the lattice parameter,
        the prefix must be extracted on the :meth:`~PWscfCellParametersReaderState.enter` method.
        :meth:`~PWscfCellParametersReaderState.execute` is first called on the next line of input.
        """
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        self._units = non_empty_tokens[-1]
        self._lattice_parameter = float(non_empty_tokens[-2])
        if self._units == "a.u.":
            # lattice parameter is always converted to angstroms
            self._lattice_parameter *= scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom

    def execute(self) -> None:
        """
        Extracts unit cell lattice vectors, reciprocal lattice vectors, and
        number of electrons.
        """
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        # skip all blank lines
        if len(non_empty_tokens) > 0:
            # Pseudopotential section ends the structure data section of the output, exit state
            if non_empty_tokens[0] == "PseudoPot.":
                self._file_reader.switch_state("idle")
                return

            # extracts number of electrons in the system
            if " ".join(non_empty_tokens[0:3]) == "number of electrons":
                self._file_reader.num_electrons = float(non_empty_tokens[-1])
            # extracts real space lattice parameters
            elif non_empty_tokens[0] in ["a(1)", "a(2)", "a(3)"]:
                lattice_vec = NumberFormatter.string_to_float(non_empty_tokens[3:6])
                lattice_vec = list(
                    # real space lattice components are expressed in units of the lattice parameter in the
                    # output, so they are re-expressed in inverse angstroms here
                    map(
                        lambda component: component * self._lattice_parameter,
                        lattice_vec,
                    )
                )
                self._file_reader.structure.lattice_vec.append(lattice_vec)
            # extracts reciprocal lattice parameters
            elif non_empty_tokens[0] in ["b(1)", "b(2)", "b(3)"]:
                recip_vec = NumberFormatter.string_to_float(non_empty_tokens[3:6])
                recip_vec = list(
                    # reciprocal lattice components are expressed in units of 2 * pi / lattice parameter in the
                    # output, so they are re-expressed in inverse angstroms here
                    map(
                        lambda component: component
                                          * 2
                                          * np.pi
                                          / self._lattice_parameter,
                        recip_vec,
                    )
                )
                self._file_reader.structure.recip_vec.append(recip_vec)

    def exit(self) -> None:
        pass


class RelaxIdleReaderState(PWscfOutputReaderState):
    """
    PWscf calculation='relax' output idle helper state for transitioning between states

    Reads file input to determine transitions into the "name", "cell_parameters", and "positions" states,
    corresponding to :class:`~RelaxNameReaderState`, :class:`~PWscfCellParametersReaderState`,
    and :class:`~RelaxPositionsReaderState`. Note that the "positions" state will only be entered if the geometry
    optimization cycle has successfully ended, indicated by the line "End of BFGS Geometry Optimization" line in
    the output.
    """

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        # "Writing config-only to output data dir" output contains the PWscf prefix on the same line
        if "Writing config-only to output data dir" in self._file_reader.current_str:
            self._file_reader.switch_state("name")

        # "lattice parameter (alat)" indicates the start of the section in the PWscf output dedicated towards structural
        # information
        elif "lattice parameter (alat)" in self._file_reader.current_str:
            self._file_reader.switch_state("cell_parameters")

        # "End of BFGS Geometry Optimization" indicates a successfully completed geometry optimization cycle. This
        # line will not be present in the output for incomplete optimization runs that have terminated without
        # satisfying the convergence thresholds. Atomic positions for those outputs will not be read.
        elif "End of BFGS Geometry Optimization" in self._file_reader.current_str:
            self._file_reader.switch_state("positions")

    def exit(self) -> None:
        pass


class RelaxNameReaderState(PWscfOutputReaderState):
    """
    PWscf calculation='relax' output reader state for extracting the PWscf calculation prefix

    The PWscf calculation prefix is set by the "prefix" field in the PWscf input file.
    """

    def enter(self) -> None:
        """
        Extracts the PWscf calculation prefix

        Since :class:`~RelaxIdleReaderState` state switches to :class:`~RelaxNameReaderState` state on the line
        containing the prefix, the prefix must be extracted on the :meth:`~RelaxNameReaderState.enter` method. :meth:`~RelaxNameReaderState.execute` is
        first called on the next line of input. Once the prefix is read, the :meth:`~RelaxNameReaderState.enter` method
        calls :meth:`~FileReader.switch_state` to switch back to the :class:`~RelaxIdleReaderState` state.
        """
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        # The output contains the prefix in the form of "{some words} {outdir}/{prefix}.save". This processes the
        # output to extract only the prefix
        name = ".".join(non_empty_tokens[-1].split("/")[-2].split(".")[:-1])
        self._file_reader.structure.name = name
        self._file_reader.switch_state("idle")

    def execute(self) -> None:
        pass

    def exit(self) -> None:
        pass


class RelaxPositionsReaderState(PWscfOutputReaderState):
    """
    PWscf calculation='relax' output reader state for extracting final atomic positions

    In the calculation='relax' output, the final atomic positions are prefixed by the line "End of BFGS Geometry
    Optimization" before the atomic positions are outputted in the PWscf ATOMIC_POSITIONS card input format.

    :ivar _started_reading: flag for whether the file reader has started reading atomic positions
    :vartype _started_reading: boolean
    """

    def __init__(self, file_reader: 'FileReader') -> None:
        """:class:`RelaxPositionsReaderState` constructor

        :param file_reader: the :class:`~FileReader` finite state machine this state is associated with.
        :type file_reader: FileReader
        """
        super().__init__(file_reader)
        # flag for whether atomic positions are being read
        self._started_reading = False

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        # blanks lines preface or tail the atomic positions in the output
        if (
                len(non_empty_tokens) == 0
                or self._file_reader.current_str.strip() == "End final coordinates"
        ):
            # if atomic positions have already been read, then the blank lines must indicate the end of atomic positions
            if self._started_reading:
                self._file_reader.switch_state("idle")
                return
        elif non_empty_tokens[0] == "ATOMIC_POSITIONS":
            self._started_reading = True
            # check if relaxed structure is specified with fractional or cartesian coordinates
            if non_empty_tokens[1] == "(crystal)":
                self._file_reader.structure.coordinate_mode = CoordinateModes.FRACTIONAL
            return

        if not self._started_reading:
            return

        # atom label, corresponding to a pseudopotential
        label = non_empty_tokens[0]
        # atomic position coordinates, either fractional or cartesian
        x, y, z = NumberFormatter.string_to_float(non_empty_tokens[1:4])
        self._file_reader.structure.atoms.append(StructureAtom([x, y, z], label))

    def exit(self) -> None:
        pass


class SCFIdleReaderState(PWscfOutputReaderState):
    """
    PWscf calculation='scf' output idle helper state for transitioning between states

    Reads file input to determine transitions into the "name" and "cell_parameters" states corresponding to
    :class:`~SCFNameReaderState` and :class:`~PWscfCellParametersReaderState`.
    """

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        # "Writing config-only to output data dir" output contains the PWscf prefix on the same line
        if "Writing all to output data dir" in self._file_reader.current_str:
            self._file_reader.switch_state("name")
        # "lattice parameter (alat)" indicates the start of the section in the PWscf output dedicated towards structural
        # information
        elif "lattice parameter (alat)" in self._file_reader.current_str:
            self._file_reader.switch_state("cell_parameters")

    def exit(self) -> None:
        pass


# TODO: merge with RelaxNameReaderState
class SCFNameReaderState(PWscfOutputReaderState):
    """
    PWscf calculation='scf' output reader state for extracting the PWscf calculation prefix

    The PWscf calculation prefix is set by the "prefix" field in the PWscf input file.
    """

    def enter(self) -> None:
        """
        Extracts the PWscf calculation prefix

        Since :class:`~SCFIdleReaderState` state switches to :class:`~SCFNameReaderState` state on the line
        containing the prefix, the prefix must be extracted on the :meth:`~SCFNameReaderState.enter` method. :meth:`~SCFNameReaderState.execute` is
        first called on the next line of input. Once the prefix is read, the :meth:`~SCFNameReaderState.enter` method calls
        :meth:`~FileReader.switch_state` to switch back to the :class:`~SCFNameReaderState` state.
        """
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        name = ".".join(non_empty_tokens[-1].split("/")[-2].split(".")[:-1])
        self._file_reader.structure.name = name
        self._file_reader.switch_state("idle")

    def execute(self) -> None:
        pass

    def exit(self) -> None:
        pass


class CIFReaderState(FileReaderState):
    """Abstract base class for a reader state of a :class:`CIFReader` finite state machine."""

    def __init__(self, file_reader: 'CIFReader'):
        """:class:`CIFReaderState` constructor

        :param file_reader: the :class:`~CIFReader` finite state machine this state is associated with
        :type file_reader: CIFReader
        """
        super().__init__(file_reader)


class CIFIdleReaderState(CIFReaderState):
    """
    CIF file idle reader state for transitioning between states

    Reads file input to determine transitions into the 'name', 'cell_parameters', and 'loop' states corresponding to
    :class:`~CIFNameReaderState`, :class:`~CIFCellParametersReaderState`, and :class:`~CIFLoopReaderState`.
    """

    def enter(self) -> None:
        pass

    def execute(self) -> None:
        # "data_" is followed by the system name in the CIF format
        if "data_" in self._file_reader.current_str:
            self._file_reader.switch_state("name")

        elif (
                # cell parameter fields start with _cell in the CIF format
                len(self._file_reader.current_str.split("_")) > 1
                and "cell" == self._file_reader.current_str.split("_")[1]
        ):
            self._file_reader.switch_state("cell_parameters")

        # loop_ indicates the start of looped information such as atomic positions in the CIF format
        elif "loop_" in self._file_reader.current_str:
            self._file_reader.switch_state("loop")

    def exit(self) -> None:
        pass


class CIFNameReaderState(CIFReaderState):
    """
    CIF file reader state for extracting the name of the system
    """

    def enter(self) -> None:
        """
        Extracts the system name

        Since :class:`~CIFIdleReaderState` state switches to :class:`~CIFNameReaderState` state on the line
        containing the system name, the name must be extracted on the :meth:`~CIFNameReaderState.enter` method. :meth:`~CIFNameReaderState.execute` is
        first called on the next line of input. Once the prefix is read, the :meth:`~CIFNameReaderState.enter` method calls
        :meth:`~FileReader.switch_state` to switch back to the :class:`~CIFIdleReaderState` state.
        """
        name = "_".join(self._file_reader.current_str.strip().split("_")[1:])
        self._file_reader.structure.name = name
        self._file_reader.switch_state("idle")

    def execute(self) -> None:
        pass

    def exit(self) -> None:
        pass


class CIFLoopReaderState(CIFReaderState):
    """
    CIF file reader state for extracting loop information of the system

    The fields under the loop_ flag in a CIF file indicate the order of information specified in the loop. This state
    extracts the field order and transitions to the "positions" state, or :class:`CIFPositionsReaderState`, to read in
    atomic positions.

    :ivar _field_order: order of fields specified in the loop_
    :vartype _field_order: Iterable[str]
    """

    def __init__(self, file_reader: 'CIFReader') -> None:
        """:class:`CIFLoopReaderState` constructor

        :param file_reader: the :class:`~CIFReader` finite state machine this state is associated with.
        :type file_reader: :class:`~CIFReader`
        """
        super().__init__(file_reader)
        # order of fields, such as atomic position and label, containing loop information specified in the CIF file
        self._field_order = []

    def enter(self) -> None:
        """
        Resets :attr:`~_field_order` to a blank list to read in a new field order
        """
        self._field_order = []

    def execute(self) -> None:
        field_tokens = list(self._file_reader.current_str.strip().split("_"))
        # fields have a minimum of two field tokens; if less, then fields are no longer being read
        if len(field_tokens) < 2:
            # check if fields indicate a loop containing atomic position information
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
                # copy field order to the :attr:`~._file_reader.pos_field_order` attribute so that the
                # :class:`CIFPositionsReaderState` has access to the field order
                self._file_reader.pos_field_order = copy.deepcopy(self._field_order)
                self._file_reader.switch_state("positions")
            else:
                self._file_reader.switch_state("idle")

        self._field_order.append(self._file_reader.current_str.strip())

    def exit(self) -> None:
        pass


class CIFCellParametersReaderState(CIFReaderState):
    """
    CIF file reader state for extracting cell parameter information of the system

    The fields under the "loop_" flag in a CIF file indicate the order of information specified in the loop. This state
    extracts the field order and transitions to the "positions" state, or :class:`CIFPositionsReaderState`, to read in
    atomic positions.

    :ivar _cell_lengths: maps "a", "b", "c" to the corresponding crystallographic cell lengths
    :vartype _cell_lengths: Dict[str, float]
    :ivar _cell_angles: maps "alpha", "beta", "gamma" to the corresponding crystallographic cell angles in degrees
    :vartype _cell_angles: Dict[str, float]
    """

    def __init__(self, file_reader: 'CIFReader') -> None:
        """:class:`CIFCellParametersReaderState` constructor

        The :attr:`~._cell_lengths` and :attr:`~._cell_angles` instance dictionaries are used to store cell parameter
        information within :class:`CIFCellParametersReaderState` so that lattice vectors and cell volume may be
        calculated once all cell parameters are read and the state exits.

        :param file_reader: the :class:`~CIFReader` finite state machine this state is associated with.
        :type file_reader: :class:`~CIFReader`
        """
        super().__init__(file_reader)
        # maps the "a", "b", "c" cell lengths in a dictionary
        self._cell_lengths = dict()
        # maps the "alpha", "beta", "gamma" cell angles in degrees (CIF specifies in degrees) in a dictionary
        self._cell_angles = dict()

    def enter(self) -> None:
        """
        Calls :meth:`~CIFCellParametersReaderState.execute` on the first line of cell parameter information

        Since :class:`~CIFIdleReaderState` state switches to :class:`~CIFCellParametersReaderState` state on a line
        containing the cell parameter information, the information must be extracted by calling :meth:`~CIFCellParametersReaderState.execute` on
        the :meth:`~CIFCellParametersReaderState.enter` method. :meth:`~execute` is first called by :attr:`~FileReaderState._file_reader` on the next line of
        input.
        """
        self.execute()

    def execute(self) -> None:
        """
        Extracts atomic position information in the CIF file
        """
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        # a blank line indicates the end of atomic position information in the CIF file
        if len(non_empty_tokens) == 0:
            self._file_reader.switch_state("idle")
            return
        cell_param_type, param_var = non_empty_tokens[0].split("_")[-2:]
        if cell_param_type == "length":
            self._cell_lengths[param_var] = float(non_empty_tokens[-1].split("(")[0])
        elif cell_param_type == "angle":
            # all angles are stored in radians
            self._cell_angles[param_var] = np.deg2rad(
                float(non_empty_tokens[-1].split("(")[0])
            )

    def exit(self):
        """
        Calculates lattice vectors from extracted cell parameter information

        Uses the cell parameters stored in the :attr:`~._cell_lengths` and :attr:`~._cell_angles` dictionaries to
        calculate the lattice vectors. Lattice vectors are defined such that the "a" cell_length is aligned with the
        x-axis in Cartesian coordinates.
        """
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
    """
    CIF file reader state for extracting atomic position information of the system

    This state should only be entered following the :class:`CIFLoopReaderState` state, since atomic position
    information in a CIF file is under a "loop_" flag. The :attr:`CIFReader.pos_field_order` attribute must be
    already set by :class:`CIFLoopReaderState` so the order of fields in the atomic positions input can be read
    correctly.
    """

    def enter(self) -> None:
        """
        Calls :meth:`~CIFPositionsReaderState.execute` on the first line of atomic position information

        Since :class:`~CIFLoopReaderState` state switches to :class:`~CIFPositionsReaderState` state on a line
        containing atomic position information, the information must be extracted by calling :meth:`~CIFPositionsReaderState.execute` on
        the :meth:`~CIFPositionsReaderState.enter` method. :meth:`~CIFPositionsReaderState.execute` is first called by :attr:`FileReaderState._file_reader` on the next line of
        input.
        """
        self.execute()

    def execute(self) -> None:
        """
        Extracts atomic position information
        """
        non_empty_tokens = list(
            filter(None, self._file_reader.current_str.strip().split(" "))
        )
        # a blank line indicates the end of atomic positions information
        if len(non_empty_tokens) == 0:
            self._file_reader.switch_state("idle")
            return

        # TODO: add option for Cartesian coordinates
        self._file_reader.structure.coordinate_mode = CoordinateModes.FRACTIONAL
        fract_pos = dict()
        label = None
        for i, field in enumerate(self._file_reader.pos_field_order):
            # field_tail is the end of the loop_ field indicating the type of field, such as "fract_x" or "type_symbol"
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

    def exit(self) -> None:
        pass


class FileReader(ABC):
    """
    Abstract base class for a file reader that implements a finite state machine

    :ivar _reader_states: registry of states that dictate the behavior of states in the finite state machine
    :vartype _reader_states: Dict[str, FileReaderState]
    :ivar _current_str: current string data from the input file being processed
    :vartype _current_str: str
    :ivar _read_path: path to the file to be read
    :vartype _read_path: str
    :ivar current_state: current state of the finite state machine
    :vartype current_state: FileReaderState
    """

    def __init__(self, reader_states: Dict[str, FileReaderState], start_state_name: str, read_path: str) -> None:
        """:class:`FileReader` constructor

        :param reader_states: registry of states that dictate the behavior of states in the finite state machine
        :type reader_states: Dict[str, FileReaderState]
        :param start_state_name: state the finite state machine begins in
        :type start_state_name: str
        :param read_path: path to the file to be read
        :type read_path: str
        """
        self._reader_states = reader_states
        # check that all values in self._reader_states are FileReaderState instances
        if not all(
                map(lambda s: isinstance(s, FileReaderState), self._reader_states.values())
        ):
            raise ValueError(
                f"read_states argument '{reader_states}' contains classes not of \
                             type FileReaderState."
            )
        self._current_str = (
            ""  # current string being proceessed. The input file may be processed line by line.
        )
        self._read_path = read_path
        self.current_state = self._reader_states[start_state_name]

    @abstractmethod
    def read_file(self) -> None:
        """Abstract method for reading file from read_path"""

    @abstractmethod
    def feed(self, str_data) -> None:
        """Abstract method for feeding a new line that results in an update to the FSM

        :param str_data: input string data from the file, might be a single line from the input file
        :type str_data: str
        """

    @property
    def current_str(self) -> str:
        """
        Gets the current string data being processed
        """
        return self._current_str

    @property
    def read_path(self) -> str:
        """
        Gets the path to the file being read
        """
        return self._read_path

    def switch_state(self, new_state_name: str) -> None:
        """
        Switches the finite state machine to a new state.

        Classes that inherit :class:`FileReader` and are pure finite state machines should only call
        :meth:`~FileReader.switch_state` through the :attr:`FileReaderState._file_reader` attribute within an instance of
        :class:`~FileReaderState`.

        :param new_state_name: name of the new state as defined in the :attr:`~._reader_states` dict
        :type new_state_name: str
        """
        self.current_state.exit()
        self.current_state = self._reader_states[new_state_name]
        self.current_state.enter()

    def execute_state(self) -> None:
        """
        Executes the current state
        """
        self.current_state.execute()

    def has_state(self, state_name: str) -> bool:
        """
        Checks if a specified state exists in the finite state machine

        :param state_name: the name of state
        :type state_name: str
        :return: True if the state exists in the finite state machine, False otherwise
        :rtype: bool
        """
        return state_name in self._reader_states.keys()


class StructureFileReader(FileReader, ABC):
    """
    Abstract base class for a file reader that reads files containing structural information

    Nearly identical to :class:`~FileReader`, but contain and additional :attr:`~structure` attribute which stores an
    instance of :class:`Structure` which holds structural information.

    :ivar structure: structure to be extracted from the file by the reader
    :vartype structure: Structure
    """

    def __init__(self, reader_states: Dict[str, FileReaderState], start_state_name: str, read_path: str) -> None:
        """:class:`StructureFileReader` constructor

        :param reader_states: registry of states that dictate the behavior of states in the finite state machine
        :type reader_states: Dict[str, FileReaderState]
        :param start_state_name: state the finite state machine begins in
        :type start_state_name: str
        :param read_path: path to the file to be read
        :type read_path: str
        """
        super().__init__(reader_states, start_state_name, read_path)
        self.structure = Structure.create_empty_structure()

    @property
    def structure(self) -> Structure:
        """
        Gets or sets the structural information.

        :raises ValueError: if attempts to set :attr:`~structure` to an object that is not an instance of :class:`Structure`
        """
        return self._structure

    @structure.setter
    def structure(self, new_structure: Structure) -> None:
        if not isinstance(new_structure, Structure):
            raise ValueError("new_structure argument is not of type Structure.")

        self._structure = new_structure


class PWscfInputReader(StructureFileReader):
    """A Quantum Espresso PWscf input file structure reader"""

    def __init__(self, read_path: str) -> None:
        """:class:`PWscfInputReader` constructor

        :param read_path: path to the input file
        :type read_path: str
        """
        super().__init__(
            reader_states={
                # idle helper state
                "idle": PWscfInputIdleReaderState(self),
                # namelist states
                "control": ControlReaderState(self),
                "system": SystemReaderState(self),
                "electrons": PWscfInputIdleReaderState(self),
                "ions": PWscfInputIdleReaderState(self),
                # card states
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
        """
        Reads the input file line by line
        """
        # PWscf input files require ASCII encoding
        with open(self.read_path, "r", encoding="ascii") as qe_file:
            for line in qe_file:
                self.feed(line)

    def feed(self, str_data: str) -> None:
        """
        Feeds new string data from the input file into the finite state machine
        :param str_data: string data to be fed to the finite stsaet machine
        :type str_data: str
        """
        self._current_str = str_data
        self.execute_state()


class PWscfOutputReader(StructureFileReader, ABC):
    """
    Abstract base class for PWscf output file readers

    :ivar num_electrons: number of electrons in the PWscf calculation
    :vartype num_electrons: int
    """

    def __init__(self, reader_states: Dict[str, FileReaderState], start_state_name: str, read_path: str):
        """:class:`PWscfOutputReader` constructor

        :param reader_states: registry of states that dictate the behavior of states in the finite state machine
        :type reader_states: Dict[str, FileReaderState]
        :param start_state_name: state the finite state machine begins in
        :type start_state_name: str
        :param read_path: path to the file to be read
        :type read_path: str
        """
        super().__init__(reader_states, start_state_name, read_path)
        self.num_electrons = None


class PWscfRelaxOutputReader(PWscfOutputReader):
    """A Quantum Espresso PWscf calculation='relax' output file structure reader"""

    def __init__(self, read_path: str) -> None:
        """:class:`~PWscfRelaxOutputReader` constructor

        :param read_path: path to input file
        :type read_path: str
        """
        super().__init__(
            reader_states={
                # idle helper state
                "idle": RelaxIdleReaderState(self),
                # reads calculation prefix
                "name": RelaxNameReaderState(self),
                # reads atomic positions
                "positions": RelaxPositionsReaderState(self),
                # reads structural parameters
                "cell_parameters": PWscfCellParametersReaderState(self),
            },
            start_state_name="idle",
            read_path=read_path,
        )
        self.structure.struct_type = "relaxed"

    def read_file(self):
        """
        Reads the input file line by line
        """
        # PWscf output files contain ASCII encoding
        with open(self.read_path, "r", encoding="ascii") as qe_file:
            for line in qe_file:
                self.feed(line)

    def feed(self, str_data):
        self._current_str = str_data
        self.execute_state()


class PWscfSCFOutputReader(PWscfOutputReader):
    """A Quantum Espresso (QE) PWscf 'scf' calculation output file structure reader"""

    def __init__(self, read_path) -> None:
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

    def read_file(self) -> None:
        with open(self.read_path, "r", encoding="ascii") as qe_file:
            for line in qe_file:
                self.feed(line)

    def feed(self, str_data: str) -> None:
        """
        Feeds new string data from the input file into the finite state machine
        :param str_data: string data to be fed to the finite state machine
        :type str_data: str
        """
        # TODO: implement this in the base class
        self._current_str = str_data
        self.execute_state()


class CIFReader(StructureFileReader):
    """
    A CIF structure file reader

    :ivar pos_field_order: order of fields in a loop_ section of a CIF file, set by :class:`~CIFLoopReaderState`.
    :vartype pos_field_order: Iterable[str]
    """

    def __init__(self, read_path: str) -> None:
        super().__init__(
            reader_states={
                # idle helper state
                "idle": CIFIdleReaderState(self),
                # reads the system's name
                "name": CIFNameReaderState(self),
                # reads the field order in a CIF loop_
                "loop": CIFLoopReaderState(self),
                # reads system structural information
                "cell_parameters": CIFCellParametersReaderState(self),
                # reads atomic positions
                "positions": CIFPositionsReaderState(self),
            },
            start_state_name="idle",
            read_path=read_path,
        )
        self.structure.struct_type = "input"
        # order of fields in t
        self.pos_field_order = []

    def read_file(self, encoding: str = "utf-8") -> None:
        """
        Reads the input file line by line

        :param encoding: encoding of the CIF file, defaults to 'utf-8'
        :type encoding: str
        """
        with open(self.read_path, "r", encoding=encoding) as cif_file:
            for line in cif_file:
                self.feed(line)

    def feed(self, str_data: str) -> None:
        """
        Feeds new string data from the input file into the finite state machine

        :param str_data: string data to be fed to the finite state machine
        :type str_data: str
        """
        self._current_str = str_data
        self.execute_state()


class FileWriter(ABC):
    """
    An abstract base class for a file writer

    :ivar _write_path: path to the file to be written
    :vartype _write_path: str
    :ivar _encoding: encoding of the file to be written
    :vartype _encoding: str
    :ivar _file_str: serialized data of the file to be written
    :vartype _file_str: str
    """

    def __init__(self, write_path: str, encoding: str) -> None:
        """:class:`~FileWriter` constructor

        :param write_path: path to the file to be written
        :type write_path: str
        :param encoding: encoding of the file to be written
        :type encoding: str
        """
        self._write_path = write_path
        self._encoding = encoding
        self._file_str = ""

    def write_file(self):
        """
        Writes file
        """
        with open(self._write_path, mode="w", encoding=self._encoding) as write_f:
            write_f.write(self._file_str)


class StructureFileWriter(FileWriter):
    """
    An output file writer that writes structural information

    :class:`~StructureFileWriter` is nearly identical to :class:`~FileWriter`, but stores additional structural
    information in :attr:`~StructureFileWriter.structure`.
    """

    def __init__(self, write_path, encoding, structure):
        super().__init__(write_path, encoding)
        self._structure = structure

    @property
    def structure(self) -> Structure:
        """
        Gets the structure information

        :return: structural information
        :rtype: Structure
        """
        return self._structure


class CIFFileWriter(StructureFileWriter):
    """A CIF structure file writer

    Since cell parameters are specified as the standard cell length and cell angle lattice parameters in a CIF file,
    they are calculated from the :obj:`~.Structure.lattice_vec` attributes in :attr:`StructureFileWriter.structure` and stored in the :attr:`~CIFFileWriter._cell_angles`,
    :obj:`~.CIFFileWriter._cell_lengths`, and :attr:`~CIFFileWriter._cell_volume` attributes.

    :ivar _cell_angles: standard crystallographic cell angles in degrees
    :vartype _cell_angles: tuple[float, float, float]
    :ivar _cell_volume: cell volume in cubic angstroms
    :vartype _cell_volume: float
    :ivar _cell_lengths: standard crystallographic cell lengths in angstroms
    :vartype _cell_lengths: tuple[float, float, float]
    :ivar _numbered_labels: collection of numbered labels for each atom, counting by the number of each element present
    :vartype _numbered_labels: Iterable[str]
    """

    def __init__(self, write_path: str, structure: Structure) -> None:
        """:class:`~CIFFileWriter` constructor

        :param write_path: path to the file to be written
        :type write_path: str
        :param structure: structural information to be written into the CIF file
        :type structure: Structure
        """
        if structure.coordinate_mode == CoordinateModes.FRACTIONAL:
            super().__init__(write_path, "utf-8", structure)
        elif structure.coordinate_mode == CoordinateModes.CARTESIAN:
            super().__init__(write_path, "utf-8", structure.to_fract())

        self._cell_angles = []
        self._cell_volume = None
        self._cell_lengths = []
        # each atom has a unique numbered label
        self._numbered_labels = []
        self._file_str = self.__serialize_structure()

    def __set_crystal_params(self) -> None:
        """
        Calculates crystallographic cell parameters from the structure lattice vectors
        """
        cell_params = self.structure.get_cell_params_from_lattice_vec()
        self._cell_lengths = cell_params[:3]
        self._cell_angles = cell_params[3:]
        self._cell_volume = self.structure.get_cell_volume()

    def __set_numbered_labels(self) -> None:
        """
        Creates numbered labels for each atom counting by each atom, with separate counts for each element
        """
        atom_labels = [atom.label for atom in self.structure.atoms]
        atom_counts = dict()
        for label in atom_labels:
            if label not in atom_counts.keys():
                atom_counts[label] = 1
            else:
                atom_counts[label] += 1
            self._numbered_labels.append(f"{label + str(atom_counts[label]): <4}")

    def __serialize_structure(self) -> str:
        """
        Serializes the structure information in :attr:`~structure` into a string formatted as a CIF file

        :return: serialized structure information
        :rtype: str
        """
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

        # cell parameters
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
    """
    Stores settings for a Quantum Espresso PWscf calculation

    Settings for a PWscf calculation are specified in the Quantum Espresso `documentation
    <https://www.quantum-espresso.org/Doc/INPUT_PW.html>`_ Input options for settings are checked to make sure they
    are valid settings. # TODO: implement this for all possible settings
    """

    def __init__(
            self,
            structure: Structure,
            calculation: str,
            pseudo_dir: str,
            ecutwfc: float,
            ecutrho: float,
            occupations: str,
            pseudos: Dict[str, str],
            forces: Iterable[tuple[int, int, int]],
            k_points_mode: str,
            k_points: str,
            disk_io: str,
            out_prefix: str = "out_",
            verbosity: str = "low",
            etot_conv_thr: float = None,
            forc_conv_thr: float = None,
            restart_mode: str = None,
            max_seconds: int = None,
            ibrav: int = 0,
            nosym: str = None,
            noncolin: bool = None,
            lspinorb: bool = None,
            nbnd: int = None,
            conv_thr: float = None,
            mixing_beta: float = None,
            masses: Dict[str, float] = None,
            vdw_corr: str = None,
    ):
        """:class:`~PWscfCalculation`

        :param structure: structure the calculation is performed on
        :type structure: Structure
        :param calculation: corresponds to 'calculation' setting in the &control namelist
        :type calculation: str
        :param pseudo_dir: corresponds to the 'pseudo_dir' setting in the &control namelist
        :type pseudo_dir: str
        :param ecutwfc: corresponds to the 'ecutwfc' setting in the &system namelist
        :type ecutwfc: float
        :param ecutrho: corresponds to the 'ecutrho' setting in the &system namelist
        :type ecutrho: float
        :param occupations: corresponds to the 'occupations' setting in the &system namelist # TODO: add check for occupations str
        :type occupations: str
        :param pseudos: dict mapping atom labels 'X' to pseudopotential paths 'PseudoPot_X', in the ATOMIC_SPECIES card
        :type pseudos: Dict[str, str]
        :param forces: 'if_pos(1)', 'if_pos(2)', 'if_pos(3)' toggles for each atom for the ATOMIC_POSITIONS card  # TODO: change to boolean components
        :type forces: Iterable[tuple[int, int, int]]
        :param k_points_mode: corresponds to the K_POINTS card option
        :type k_points_mode: str
        :param k_points: k-points used by calculation specified under the K_POINTS card
        :type k_points: str
        :param disk_io: corresponds to the 'disk_io' setting in the &control namelist # TODO: add check
        :type disk_io: str
        :param out_prefix: prefix for the 'outdir' setting in the &control namelist. The prefix is followed by the structure name. Defaults to 'out_'
        :type out_prefix: str
        :param verbosity: corresponds to the 'verbosity' setting in the &control namelist, defaults to 'low' # TODO: add check
        :type verbosity: str
        :param etot_conv_thr: corresponds to the 'etot_conv_thr' setting in the &control namelist, defaults to None (unspecified in input)
        :type etot_conv_thr: float
        :param forc_conv_thr: corresponds to the 'forc_conv_thr' setting in the &control namelist, defaults to None (unspecified in input)
        :type forc_conv_thr: float
        :param restart_mode: corresponds to the 'restart_mode' setting in the &control namelist, defaults to None (unspecified in input)
        :type restart_mode: str
        :param max_seconds: corresponds to the 'max_seconds' setting, defaults to None (unspecified in input)
        :type max_seconds: int
        :param ibrav: corresponds to the 'ibrav' setting in the &system namelist, defaults to 0
        :type ibrav: int
        :param nosym: corresponds to the 'nosym' setting in the &system namelist, defaults to None (unspecified)
        :type nosym: bool
        :param noncolin: corresponds to the 'noncolin' setting in the &system namelist, defaults to None (unspecified)
        :type noncolin: bool
        :param lspinorb: corresponds to the 'lspinorb' setting in the &system namelist, defaults to None (unspecified)
        :type lspinorb: bool
        :param nbnd: corresponds to the 'nbnd' setting in the &system namelist, defaults to None (unspecified)
        :type nbnd: int
        :param conv_thr: corresponds to the 'conv_thr' setting in the &system namelist, defaults to None (unspecified)
        :type conv_thr: float
        :param mixing_beta: corresponds to the 'mixing_beta' setting in the &system namelist, defaults to None (unspecified)
        :type mixing_beta: float
        :param masses: dict mapping atom labels 'X' their masses 'Mass_X', in the ATOMIC_SPECIES card, defaults to None (unspecified)
        :type masses: Dict[str, float]
        :param vdw_corr: corresponds to the 'vdw_corr' setting in the &system namelist, defaults to None (unspecified)
        :type vdw_corr: str
        """
        self.structure = structure
        self.unique_labels = []
        # extract unique atomic labels in the structure, used to verify the correct pseudos and masses dicts later
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
        if not all(map(lambda label: label in self.pseudos.keys(), self.unique_labels)):
            raise ValueError(
                f"keys of pseudos, {self.pseudos.keys()} must contain all unique labels of structure, {self.unique_labels}."
            )
        self.ecut_wfc = ecutwfc
        self.ecut_rho = ecutrho
        self.occupations = occupations
        self.k_points_mode = k_points_mode
        self.k_points = k_points
        self.disk_io = disk_io
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
        self.vdw_corr = vdw_corr
        if self.masses is None:
            self.masses = dict()
            # set default mass to 1.0 if the atomic masses are unspecified
            for label in self.unique_labels:
                self.masses[label] = 1.0
        else:
            if not all(
                    map(lambda label: label in self.masses.keys(), self.unique_labels)
            ):
                raise ValueError(
                    f"keys of masses, {self.masses.keys()} must contain all unique labels of structure, {self.unique_labels}."
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
    """A Quantum Espresso (QE) PWscf input file writer"""

    def __init__(self, write_path: str, structure: Structure, pw_scf_calc: PWscfCalculation) -> None:
        """:class:`PWscfInputWriter` constructor

        :param write_path: path to PWscf input file to be written
        :type write_path: str
        :param structure: structure specified in the PWscf input file
        :type structure: Structure
        :param pw_scf_calc: PWscf calculation settings
        :type pw_scf_calc: PWscfCalculation
        """
        super().__init__(write_path, "ascii", structure)
        self.pw_scf_calc = pw_scf_calc
        self._namelists = None
        self._cards = None
        self.__format_namelists()
        self.__format_cards()
        # combine the namelists and cards into the file string
        self._file_str = self._namelists + self._cards

    def __format_sci_not(self, num: float) -> str:
        """
        Formats float numbers into the scientific notation format accepted by PWscf, to 10 decimal places

        :param num: float number to be formatted
        :type num: float
        :return: float number formatted in scientific notation
        :rtype: str
        """
        sci_not = f"{num:.10E}"
        if not isinstance(num, float) or "E" not in sci_not:
            raise ValueError("num must be a float in scientific notation")

        return sci_not.replace("E", "d")  # PWscf specifies the start of the exponent with a "d"

    def __format_bool(self, bool_val: bool) -> str:
        """
        Formats a bool into the FORTRAN-style boolean format accepted by PWscf
        :param bool_val: bool to be formatted
        :type bool_val: bool
        :return: bool formatted to a FORTRAN-style boolean format
        :rtype: str
        """
        if not isinstance(bool_val, bool):
            raise ValueError("bool_val must be a boolean.")

        return ".true." if bool_val else ".false."

    def __format_namelists(self):
        """
        Writes strings containing namelists for the PWscf input file to the :attr:`~_namelists` attribute
        """
        if self._namelists is None:
            control_str = f"""&CONTROL
  calculation = '{self.pw_scf_calc.calculation}'
  outdir = './{self.pw_scf_calc.out_prefix}{self.pw_scf_calc.prefix}/'
  prefix = '{self.pw_scf_calc.prefix}'
  pseudo_dir = './pseudo/'
  verbosity = '{self.pw_scf_calc.verbosity}'
  disk_io = '{self.pw_scf_calc.disk_io}'
"""
            if self.pw_scf_calc.etot_conv_thr is not None:
                control_str += f"  etot_conv_thr = {self.__format_sci_not(self.pw_scf_calc.etot_conv_thr)}\n"
            if self.pw_scf_calc.forc_conv_thr is not None:
                control_str += f"  forc_conv_thr = {self.__format_sci_not(self.pw_scf_calc.forc_conv_thr)}\n"
            if self.pw_scf_calc.restart_mode is not None:
                control_str += f"  restart_mode = '{self.pw_scf_calc.restart_mode}'\n"
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
            if self.pw_scf_calc.vdw_corr is not None:
                system_str += f"  vdw_corr = '{self.pw_scf_calc.vdw_corr}'\n"
            system_str += "/\n"

            electrons_str = "&ELECTRONS\n"
            if self.pw_scf_calc.conv_thr is not None:
                electrons_str += (
                    f"  conv_thr = {self.__format_sci_not(self.pw_scf_calc.conv_thr)}\n"
                )
            if self.pw_scf_calc.mixing_beta is not None:
                electrons_str += f"  mixing_beta = {self.__format_sci_not(self.pw_scf_calc.mixing_beta)}\n"
            electrons_str += "/\n"

            ions_str = (
                "&IONS\n/\n"
                if self.pw_scf_calc.calculation in ["relax", "vc-relax"]
                else ""
            )

            self._namelists = control_str + system_str + electrons_str + ions_str

    def __format_cards(self):
        """
        Writes strings containing cards for the PWscf input file to the :attr:`~_cards` attribute
        """
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


class BandsXCalculation:
    """
    Stores settings for a Quantum Espresso band.x utility calculation

    Settings for a band.x calculation are specified in the Quantum Espresso `documentation
    <https://www.quantum-espresso.org/Doc/INPUT_BANDS.html>`_ Input options for settings are checked to make sure they
    are valid settings. # TODO: implement this for all possible settings
    """

    def __init__(
            self,
            structure: Structure,
            lsym: str,
            out_prefix: str = "out_",
            filband_suffix: str = ".band_structure.dat",
            lsigma3: bool = None,
    ):
        """:class:`~BandsXCalculation` constructor

        :param structure: structure utility is performed on
        :type structure: Structure
        :param lsym: corresponds to the 'lsym' setting in the &bands namelist
        :type lsym: str
        :param out_prefix: specifies the prefix to the out setting in the &bands namelist. The prefix precedes the system name. Defaults to out_
        :type out_prefix: str
        :param filband_suffix: specifies the suffix to the filband setting in the &bands namelist. The suffix follows the system name. Defaults to .band_structure.dat
        :type filband_suffix: str
        :param lsigma3: corresponds to the 'lsigma(3)' setting in the &bands namelist, defaults to None (unspecified)
        :type lsigma3: bool
        """
        self.structure = structure
        self.lsym = lsym
        self.lsigma3 = lsigma3
        self.prefix = structure.name
        self.out_prefix = out_prefix
        self.filband_suffix = filband_suffix


class BandsXInputWriter(StructureFileWriter):
    """A Quantum Espresso (QE) band.x utility input file writer"""

    def __init__(self, write_path: str, bands_x_calc: BandsXCalculation) -> None:
        """:class:`~BandsXInputWriter` constructor

        :param write_path: path to the input file to be written
        :type write_path: str
        :param bands_x_calc: BandsX calculation settings
        :type bands_x_calc: BandsXCalculation
        """
        super().__init__(write_path, encoding="ascii", structure=bands_x_calc.structure)
        self._bands_x_calc = bands_x_calc
        self._file_str = f"""&bands
    prefix = '{self._bands_x_calc.prefix}',
    outdir = './{self._bands_x_calc.out_prefix}{self._bands_x_calc.prefix}/',
    filband = '{self._bands_x_calc.prefix}{self._bands_x_calc.filband_suffix}',
    lsym = {self.__format_bool(self._bands_x_calc.lsym)}"""

        if self._bands_x_calc.lsigma3 is not None:
            self._file_str += f""",
    lsigma(3) = {self.__format_bool(self._bands_x_calc.lsigma3)}"""

        self._file_str += "\n/\n"

    # TODO: merge this with the PWscf formatter into a Quantum Espresso writer
    def __format_bool(self, bool_val):
        """
        Formats a bool into the FORTRAN-style boolean format accepted by PWscf
        :param bool_val: bool to be formatted
        :type bool_val: bool
        :return: bool formatted to a FORTRAN-style boolean format
        :rtype: str
        """
        if not isinstance(bool_val, bool):
            raise ValueError("bool_val must be a boolean.")

        return ".true." if bool_val else ".false."
