#!/usr/bin/env python

import numpy as np
import networkx as nx
from marc.exceptions import InputError


class molecule:
    def __init__(self, filename, noh=True):
        """
        Simple molecule class.
        """

        f = open(filename, "r")
        V = list()
        atoms = list()
        n_atoms = 0

        # Read the first line to obtain the number of atoms to read
        try:
            n_atoms = int(f.readline())
        except ValueError:
            raise InputError("Could not obtain the number of atoms in the .xyz file.")

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
            atom = atom.upper()

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
        atoms = np.array(atoms)
        V = np.array(V)
        self.title = title
        if noh:
            self.atoms = atoms[np.where(atoms > 1)]
            self.coordinates = Z[np.where(atoms > 1)]
        else:
            self.atoms = atoms
            self.coordinates = V
        self.energy = energy
        set_am(self)
        set_graph(self)

    def set_am(self):
        n = len(self.atoms)
        am = np.zeros((n, n), dtype=int)
        row, col = np.triu_indices(am)
        am[row, col] = am[col, row] = scipy.spatial.distance.pdist(
            self.coordinates
        ) - scale_factor * scipy.spatial.distance.pdist(
            radii.reshape(-1, 1), metric=lambda x, y: x + y
        )
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
            ds[i] = np.linalg.norm(coords[edge[0]] - coords[edge[1]])
            cs[i] = z[edge[0]] * z[edge[1]]
        d_dict = {edge: d for edge, d in zip(b_dict.keys(), ds)}
        c_dict = {edge: c for edge, c in zip(b_dict.keys(), cs)}
        nx.set_edge_attributes(G, d_dict, "distance")
        nx.set_edge_attributes(G, c_dict, "coulomb_term")
        self.graph = G
