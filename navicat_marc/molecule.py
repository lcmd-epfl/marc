#!/usr/bin/env python

import re
from os.path import dirname

import networkx as nx
import numpy as np
import scipy.spatial

from navicat_marc.exceptions import InputError

ha_to_kcalmol = 627.509
kcalmol_to_ha = 0.00159360

symbol_to_number = {
    "Em": 0,  # empty site
    "Vc": 0,  # empty site
    "Va": 0,  # empty site
    "D": 1,
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Me": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Uun": 110,
    "Uuu": 111,
    "Uub": 112,
}

number_to_symbol = {v: k for k, v in symbol_to_number.items()}

missing = 0.2
covalent_radii = np.array(
    [
        missing,  # X
        0.31,  # H
        0.28,  # He
        1.28,  # Li
        0.96,  # Be
        0.84,  # B
        0.76,  # C
        0.71,  # N
        0.66,  # O
        0.57,  # F
        0.58,  # Ne
        1.66,  # Na
        1.41,  # Mg
        1.21,  # Al
        1.11,  # Si
        1.07,  # P
        1.05,  # S
        1.02,  # Cl
        1.06,  # Ar
        2.03,  # K
        1.76,  # Ca
        1.70,  # Sc
        1.60,  # Ti
        1.53,  # V
        1.39,  # Cr
        1.39,  # Mn
        1.32,  # Fe
        1.26,  # Co
        1.24,  # Ni
        1.32,  # Cu
        1.22,  # Zn
        1.22,  # Ga
        1.20,  # Ge
        1.19,  # As
        1.20,  # Se
        1.20,  # Br
        1.16,  # Kr
        2.20,  # Rb
        1.95,  # Sr
        1.90,  # Y
        1.75,  # Zr
        1.64,  # Nb
        1.54,  # Mo
        1.47,  # Tc
        1.46,  # Ru
        1.42,  # Rh
        1.39,  # Pd
        1.45,  # Ag
        1.44,  # Cd
        1.42,  # In
        1.39,  # Sn
        1.39,  # Sb
        1.38,  # Te
        1.39,  # I
        1.40,  # Xe
        2.44,  # Cs
        2.15,  # Ba
        2.07,  # La
        2.04,  # Ce
        2.03,  # Pr
        2.01,  # Nd
        1.99,  # Pm
        1.98,  # Sm
        1.98,  # Eu
        1.96,  # Gd
        1.94,  # Tb
        1.92,  # Dy
        1.92,  # Ho
        1.89,  # Er
        1.90,  # Tm
        1.87,  # Yb
        1.87,  # Lu
        1.75,  # Hf
        1.70,  # Ta
        1.62,  # W
        1.51,  # Re
        1.44,  # Os
        1.41,  # Ir
        1.36,  # Pt
        1.36,  # Au
        1.32,  # Hg
        1.45,  # Tl
        1.46,  # Pb
        1.48,  # Bi
        1.40,  # Po
        1.50,  # At
        1.50,  # Rn
        2.60,  # Fr
        2.21,  # Ra
        2.15,  # Ac
        2.06,  # Th
        2.00,  # Pa
        1.96,  # U
        1.90,  # Np
        1.87,  # Pu
        1.80,  # Am
        1.69,  # Cm
        missing,  # Bk
        missing,  # Cf
        missing,  # Es
        missing,  # Fm
        missing,  # Md
        missing,  # No
        missing,  # Lr
        missing,  # Rf
        missing,  # Db
        missing,  # Sg
        missing,  # Bh
        missing,  # Hs
        missing,  # Mt
        missing,  # Ds
        missing,  # Rg
        missing,  # Cn
        missing,  # Nh
        missing,  # Fl
        missing,  # Mc
        missing,  # Lv
        missing,  # Ts
        missing,  # Og
    ]
)


class Molecule:
    def __init__(
        self,
        atoms=None,
        coordinates=None,
        energy=None,
        filename=None,
        lines=None,
        radii=None,
        scale_factor=1.2,
        noh=True,
    ):
        self.scale_factor = scale_factor
        self.radii = radii
        if filename is not None:
            self.from_file(filename, noh)
        elif lines is not None:
            self.from_lines(lines, noh)
        else:
            self.atoms = atoms
            self.coordinates = coordinates
            if self.radii is None and self.atoms is not None:
                self.set_radii()
            else:
                self.radii = radii
            self.set_am()
            self.set_graph()

    def from_file(self, filename, noh=True):
        f = open(filename, "r")
        V = list()
        atoms = list()
        n_atoms = 0

        # Read the first line to obtain the number of atoms to read
        try:
            n_atoms = int(f.readline().strip())
        except ValueError:
            raise InputError(
                f"Could not obtain the number of atoms in the .xyz file {filename} from first line. Please check format."
            )

        # The title line may contain an energy
        title = f.readline().strip()
        energy = None
        if "energy:" in title and energy is None:
            try:
                etitle = title.split(":")[1].split(" ")[1].rstrip()
                energy = float(etitle) * ha_to_kcalmol
            except ValueError:
                energy = None
            except AttributeError:
                energy = None
        if "Eopt" in title and energy is None:
            try:
                etitle = title.split("Eopt")[-1].rstrip()
                energy = float(etitle) * ha_to_kcalmol
            except ValueError:
                energy = None
            except AttributeError:
                energy = None
        if energy is None:
            try:
                energy = float(title) * ha_to_kcalmol
            except ValueError:
                energy = None
            except AttributeError:
                energy = None

        # Use the number of atoms to not read beyond the end of a file
        for lines_read, line in enumerate(f):

            if lines_read == n_atoms:
                break

            atom = re.findall(r"[a-zA-Z]+", line)[0]
            atom = symbol_to_number[atom]

            numbers = re.findall(r"[-]?\d+\.\d*(?:[Ee][-\+]\d+)?", line)
            numbers = [float(number) for number in numbers]

            # The numbers are not valid unless we obtain exacly three
            if len(numbers) == 3:
                V.append(np.array(numbers))
                atoms.append(atom)
            else:
                raise InputError(
                    "Reading the .xyz file failed in line {0}. Please check the format.".format(
                        lines_read + 2
                    )
                )

        f.close()
        atoms = np.array(atoms, dtype=int)
        V = np.array(V)
        self.title = title
        self.coordinates_with_h = V
        self.atoms_with_h = atoms
        if noh:
            self.atoms = atoms[np.where(atoms > 1)]
            self.coordinates = V[np.where(atoms > 1)]
        else:
            self.atoms = atoms
            self.coordinates = V
        self.energy = energy
        if self.radii is None and self.atoms is not None:
            self.set_radii()
        self.set_am()
        self.set_graph()

    def from_lines(self, lines, noh=True):
        V = list()
        atoms = list()
        n_atoms = 0
        lines_iter = iter(lines)

        # Read the first line to obtain the number of atoms to read
        try:
            n_atoms = int(next(lines_iter).strip())
        except ValueError:
            raise InputError(
                f"Could not obtain the number of atoms in the .xyz file line {lines[0]}. Please check format."
            )

        # The title line may contain an energy
        title = next(lines_iter).strip()
        energy = None
        if "energy:" in title and energy is None:
            try:
                etitle = title.split(":")[1].split(" ")[1].rstrip()
                energy = float(etitle) * ha_to_kcalmol
            except ValueError:
                energy = None
            except AttributeError:
                energy = None
        if "Eopt" in title and energy is None:
            try:
                etitle = title.split("Eopt")[-1].rstrip()
                energy = float(etitle) * ha_to_kcalmol
            except ValueError:
                energy = None
            except AttributeError:
                energy = None
        if energy is None:
            try:
                energy = float(title) * ha_to_kcalmol
            except ValueError:
                energy = None
            except AttributeError:
                energy = None

        # Use the number of atoms to not read beyond the end of a file
        for lines_read, line in enumerate(lines_iter):

            if lines_read == n_atoms:
                break

            atom = re.findall(r"[a-zA-Z]+", line)[0]
            atom = symbol_to_number[atom]

            numbers = re.findall(r"[-]?\d+\.\d*(?:[Ee][-\+]\d+)?", line)
            numbers = [float(number) for number in numbers]

            # The numbers are not valid unless we obtain exacly three
            if len(numbers) == 3:
                V.append(np.array(numbers))
                atoms.append(atom)
            else:
                raise InputError(
                    f"Understanding the chunk of lines failed at line:\n {line}\nPlease check the format."
                )

        atoms = np.array(atoms, dtype=int)
        V = np.array(V)
        self.title = title
        self.coordinates_with_h = V
        self.atoms_with_h = atoms
        if noh:
            self.atoms = atoms[np.where(atoms > 1)]
            self.coordinates = V[np.where(atoms > 1)]
        else:
            self.atoms = atoms
            self.coordinates = V
        self.energy = energy
        if self.radii is None and self.atoms is not None:
            self.set_radii()
        self.set_am()
        self.set_graph()

    def set_radii(self):
        radii = np.array([covalent_radii[i] for i in self.atoms], dtype=float)
        self.radii = radii

    def set_am(self):
        n = len(self.atoms)
        am = np.zeros((n, n), dtype=float)
        row, col = np.triu_indices(n, 1)
        dm = scipy.spatial.distance.pdist(self.coordinates)
        rm = self.scale_factor * scipy.spatial.distance.pdist(
            self.radii.reshape(-1, 1), metric=lambda x, y: x + y
        )
        am[row, col] = am[col, row] = dm - rm
        self.am = am < 0

    def set_graph(self):
        G = nx.from_numpy_array(self.am, create_using=nx.Graph)
        an_dict = {i: self.atoms[i] for i in range(len(self.atoms))}
        coord_dict = {i: self.coordinates[i] for i in range(len(self.atoms))}
        nx.set_node_attributes(G, an_dict, "atomic_number")
        nx.set_node_attributes(G, coord_dict, "coordinates")
        ds = np.zeros((len(G.edges())))
        cs = np.zeros_like(ds)
        for i, edge in enumerate(G.edges()):
            ds[i] = np.linalg.norm(
                self.coordinates[edge[0]] - self.coordinates[edge[1]]
            )
            cs[i] = self.atoms[edge[0]] * self.atoms[edge[1]] / ds[i] ** 2
        b_dict = nx.edge_betweenness_centrality(G, normalized=False)
        d_dict = {edge: d for edge, d in zip(b_dict.keys(), ds)}
        c_dict = {edge: c for edge, c in zip(b_dict.keys(), cs)}
        nx.set_edge_attributes(G, b_dict, "betweenness")
        nx.set_edge_attributes(G, d_dict, "distance")
        nx.set_edge_attributes(G, c_dict, "coulomb_term")
        self.graph = G

    def write(self, rootname="output"):
        filename = f"{rootname}.xyz"
        f = open(filename, "w+")
        print(f"{len(self.atoms_with_h)}", file=f)
        if self.energy is not None:
            printable_energy = np.round(self.energy * kcalmol_to_ha, decimals=6)
            print(f"{printable_energy}", file=f)
        else:
            print(f"{self.title}", file=f)
        for i, atom in enumerate(self.atoms_with_h):
            print(
                f"{number_to_symbol[atom]:2}    {np.round(self.coordinates_with_h[i][0],decimals=6): }    {np.round(self.coordinates_with_h[i][1],decimals=6): }    {np.round(self.coordinates_with_h[i][2],decimals=6): }",
                file=f,
            )
        f.close()

    def __iter__(self):
        for value in [
            self.energy,
            self.coordinates,
            self.atoms,
            self.radii,
            self.am,
            self.graph,
        ]:
            yield value


def test_compare_origin(path=f"{dirname(__file__)}/test_files/"):
    chunk_a = [
        "19",
        "(2S)-2-Amino-3-methylbutanoic acid",
        "  C      0.2036     -0.4958      0.3403",
        "  N      1.4832     -1.2440      0.2997",
        "  C      0.3147      0.9660      0.8346",
        "  C     -1.0593      1.6179      0.8658",
        "  C      0.9346      1.0303      2.2224",
        "  C     -0.3596     -0.5230     -1.0775",
        "  O      0.1045     -0.0437     -2.0961",
        "  O     -1.5354     -1.1775     -1.2134",
        "  H     -0.4768     -1.0587      1.0299",
        "  H      1.8309     -1.3539      1.2292",
        "  H      2.1548     -0.7502     -0.2505",
        "  H      0.9641      1.5372      0.1249",
        "  H     -1.5332      1.6117     -0.1249",
        "  H     -0.9924      2.6651      1.1892",
        "  H     -1.7373      1.1021      1.5594",
        "  H      0.9116      2.0570      2.6127",
        "  H      1.9862      0.7132      2.2244",
        "  H      0.3950      0.3965      2.9394",
        "  H     -1.8067     -1.1757     -2.1262",
    ]
    molecule_a = Molecule(lines=chunk_a)
    molecule_b = Molecule(filename=f"{path}L-Valine.xyz")
    for value_a, value_b in zip(molecule_a, molecule_b):
        if isinstance(value_a, np.ndarray):
            assert np.allclose(value_a, value_b)
        elif isinstance(value_a, list):
            assert all(value_a == value_b)
        elif isinstance(value_a, nx.Graph):
            assert nx.is_isomorphic(value_a, value_b)
        elif value_a is None:
            assert value_b is None


def test_molecule_from_lines():
    chunks = []
    molecule_1 = [
        "19",
        "(2S)-2-Amino-3-methylbutanoic acid",
        "  C      0.2036     -0.4958      0.3403",
        "  N      1.4832     -1.2440      0.2997",
        "  C      0.3147      0.9660      0.8346",
        "  C     -1.0593      1.6179      0.8658",
        "  C      0.9346      1.0303      2.2224",
        "  C     -0.3596     -0.5230     -1.0775",
        "  O      0.1045     -0.0437     -2.0961",
        "  O     -1.5354     -1.1775     -1.2134",
        "  H     -0.4768     -1.0587      1.0299",
        "  H      1.8309     -1.3539      1.2292",
        "  H      2.1548     -0.7502     -0.2505",
        "  H      0.9641      1.5372      0.1249",
        "  H     -1.5332      1.6117     -0.1249",
        "  H     -0.9924      2.6651      1.1892",
        "  H     -1.7373      1.1021      1.5594",
        "  H      0.9116      2.0570      2.6127",
        "  H      1.9862      0.7132      2.2244",
        "  H      0.3950      0.3965      2.9394",
        "  H     -1.8067     -1.1757     -2.1262",
    ]
    molecule_2 = [
        "46",
        "Testosterone",
        "  C     -4.0599     -2.1760     -0.8224",
        "  O     -4.9516     -2.8840     -1.2414",
        "  C     -4.2163     -0.6676     -0.7586",
        "  C     -2.8826      0.0343     -0.9993",
        "  C     -2.7857     -2.7158     -0.3131",
        "  C     -1.7443     -1.9501      0.0575",
        "  C     -0.5249     -2.5861      0.6659",
        "  C      0.7827     -1.9356      0.2082",
        "  C      0.7295     -0.4182      0.4294",
        "  C      2.0267      0.2733     -0.0072",
        "  C      3.3706     -0.1900      0.5799",
        "  C      4.3192      1.0027      0.3273",
        "  C      3.4317      2.2202     -0.0268",
        "  O      3.9384      3.3043      0.7679",
        "  C      1.9714      1.7980      0.3340",
        "  C      1.7106      2.0852      1.8182",
        "  C      0.8686      2.4201     -0.5385",
        "  C     -0.4798      1.7315     -0.2601",
        "  C     -0.4282      0.1930     -0.4091",
        "  C     -1.7900     -0.4363     -0.0132",
        "  H     -2.7575     -3.8052     -0.2565",
        "  H     -0.6135     -2.5201      1.7733",
        "  H     -0.4925     -3.6738      0.4451",
        "  H      1.6336     -2.3732      0.7646",
        "  H      0.9690     -2.1675     -0.8583",
        "  H      0.5479     -0.2168      1.5138",
        "  H      2.1058      0.1678     -1.1210",
        "  H      3.2853     -0.4094      1.6573",
        "  H      3.7364     -1.1101      0.0994",
        "  H      4.9267      1.2382      1.2217",
        "  H      5.0342      0.7890     -0.4822",
        "  H      3.5293      2.5237     -1.0880",
        "  H      3.2906      4.0296      0.8195",
        "  H      0.8832      1.4797      2.2078",
        "  H      2.5972      1.8604      2.4268",
        "  H      1.4613      3.1361      1.9910",
        "  H      0.7818      3.5044     -0.3452",
        "  H      1.1247      2.3262     -1.6107",
        "  H     -0.8205      1.9959      0.7596",
        "  H     -1.2411      2.1442     -0.9502",
        "  H     -0.2361     -0.0466     -1.4843",
        "  H     -4.6390     -0.3984      0.2304",
        "  H     -4.9734     -0.3373     -1.4982",
        "  H     -2.5493     -0.1544     -2.0405",
        "  H     -3.0184      1.1303     -0.9178",
        "  H     -2.0604     -0.0823      1.0166",
    ]
    for chunk in chunks:
        a = Molecule(filename=f"{filename}")
        assert a.energy is None
        assert nx.is_connected(a.graph)


def test_molecule_from_file(path=f"{dirname(__file__)}/test_files/"):
    filenames = [
        "L-Valine.xyz",
        "Benzaldehyde.xyz",
        "Ferrocene.xyz",
        "Testosterone.xyz",
    ]
    for filename in filenames:
        a = Molecule(filename=f"{path}{filename}")
        assert a.energy is None
        assert nx.is_connected(a.graph)


def test_molecule_to_file(path=f"{dirname(__file__)}/test_files/"):
    filenames = [
        "L-Valine.xyz",
        "Benzaldehyde.xyz",
        "Ferrocene.xyz",
        "Testosterone.xyz",
    ]
    for filename in filenames:
        a = Molecule(filename=f"{path}{filename}")
        a.write("test")
