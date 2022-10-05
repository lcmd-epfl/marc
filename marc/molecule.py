#!/usr/bin/env python

import re

import networkx as nx
import numpy as np
import scipy.spatial

from marc.exceptions import InputError

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
        if title.lstrip("-").isdigit():
            energy = float(title)
        else:
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
        if title.lstrip("-").isdigit():
            energy = float(title)
        else:
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
        am = np.zeros((n, n), dtype=int)
        row, col = np.triu_indices(n, 1)
        dm = scipy.spatial.distance.pdist(self.coordinates)
        rm = scipy.spatial.distance.pdist(self.radii.reshape(-1,1), metric=lambda x, y: x + y)
        am[row, col] = am[col, row] = dm - self.scale_factor * rm
        self.am = (am < 0).astype(int)

    def set_graph(self):
        G = nx.from_numpy_matrix(self.am, create_using=nx.Graph)
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
            cs[i] = z[edge[0]] * z[edge[1]]
        b_dict = nx.edge_betweenness_centrality(G, normalized=False)
        d_dict = {edge: d for edge, d in zip(b_dict.keys(), ds)}
        c_dict = {edge: c for edge, c in zip(b_dict.keys(), cs)}
        nx.set_edge_attributes(G, b_dict, "betweenness")
        nx.set_edge_attributes(G, d_dict, "distance")
        nx.set_edge_attributes(G, c_dict, "coulomb_term")
        self.graph = G


def test_molecule():
    lines = [
        "3",
        "-36.12",
        "C  0.0  1.0  2.3",
        "C  1.8  0.8  3.5",
        "O 19.2  29.1  2.0",
    ]
    a = Molecule(lines=lines)
