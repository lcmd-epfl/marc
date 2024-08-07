#!/usr/bin/env python

import argparse
import glob
import re
import sys
from itertools import cycle

import networkx as nx
import numpy as np

from navicat_marc.exceptions import InputError
from navicat_marc.molecule import Molecule, at_eq, b_eq
from navicat_marc.rmsd import kabsch_rmsd, reorder_distance, reorder_hungarian


def long_substr(data):
    """Longest common substring for basename elucidation."""
    substr = ""
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and all(data[0][i : i + j] in x for x in data):
                    substr = data[0][i : i + j]
    while (substr[-1] == "-") or (substr[-1] == "_"):
        substr = substr[:-1]
    while (substr[0] == "-") or (substr[0] == "_"):
        substr = substr[1:]
    if substr == "":
        substr = "Molecule"
    return substr


valid_c = ["kmeans", "agglomerative", "affprop"]
valid_m = ["rmsd", "erel", "da", "ewrmsd", "ewda", "mix", "avg"]


def yesno(question):
    """Simple Yes/No Function."""
    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)
    if ans == "y":
        return True
    return False


def bround(x, base: float = 10, type=None) -> float:
    if type == "max":
        return base * np.ceil(x / base)
    elif type == "min":
        return base * np.floor(x / base)
    else:
        tick = base * np.round(x / base)
        return tick


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def group_data_points(bc, ec, names):
    try:
        groups = np.array([str(i)[bc:ec].upper() for i in names], dtype=object)
    except Exception as m:
        raise InputError(
            f"Grouping by name characters did not work. Error message was:\n {m}"
        )
    type_tags = np.unique(groups)
    cycol = cycle("bgrcmky")
    cymar = cycle("^ospXDvH")
    cdict = dict(zip(type_tags, cycol))
    mdict = dict(zip(type_tags, cymar))
    cb = np.array([cdict[i] for i in groups])
    ms = np.array([mdict[i] for i in groups])
    return cb, ms


def molecules_from_file(filename, scale_factor=1.10, noh=True):
    molecules = []
    f = open(filename, "r")
    n_atoms = 0

    lines = list(f.readlines())
    f.close()
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise InputError(
            f"Could not obtain the number of atoms in the .xyz file {filename} from first line. Check format."
        )

    if len(lines) % (n_atoms + 2) != 0:
        raise InputError(
            f"Could not parse trajectory xyz file {filename} properly. Check format."
        )
    for chunk in chunker(lines, n_atoms + 2):
        molecule = Molecule(lines=chunk, scale_factor=scale_factor, noh=noh)
        molecules.append(molecule)
    return molecules


def processargs(arguments):
    input_list = sys.argv
    input_str = " ".join(input_list)
    version_str = "0.2.2"
    mbuilder = argparse.ArgumentParser(
        prog="navicat_marc",
        description="Analyse conformer ensembles to find the most representative structures.",
        epilog="Remember to cite the marc paper or repository - \n if they have a DOI by now\n - and enjoy!",
    )
    mbuilder.add_argument(
        "-version", "--version", action="version", version=f"%(prog)s {version_str}"
    )
    mbuilder.add_argument(
        "-i",
        "--i",
        "-input",
        dest="input",
        nargs="+",
        action="append",
        type=str,
        required=True,
        help="Filename(s) containing the conformational ensemble as an xyz trajectory, or separately as xyz files.",
    )
    mbuilder.add_argument(
        "-c",
        "--c",
        "-cluster",
        "--cluster",
        dest="c",
        type=str,
        default="kmeans",
        help=f"Clustering algorithms to use. (default: kmeans)\nPossible values are: {valid_c}",
    )
    mbuilder.add_argument(
        "-m",
        "--m",
        "-metric",
        "--metric",
        dest="m",
        type=str,
        default="avg",
        help=f"Metric to use to define distance. (default: avg)\nPossible values are: {valid_m}",
    )
    mbuilder.add_argument(
        "-n",
        "--n",
        "--n_clusters",
        dest="n",
        default=None,
        help="Number of representative conformers to select. (default: select using gap method)",
    )
    mbuilder.add_argument(
        "-ewin",
        "--ewin",
        dest="ewin",
        default=None,
        help="If set to a float, energy window for conformers to be accepted in kcal/mol. (default: None)",
    )
    mbuilder.add_argument(
        "-sf",
        "--sf",
        dest="sf",
        type=float,
        default=1.10,
        help="If set to a float, scale factor used to determine connectivity from 3D coordinates using covalent radii. (default: 1.10)",
    )
    mbuilder.add_argument(
        "-mine",
        "--mine",
        dest="mine",
        action="store_true",
        default=False,
        help="If set, the minimum energy conformer per sample will be taken instead of the centermost one. (default: False, True if energies available)",
    )
    mbuilder.add_argument(
        "-yesh",
        "--yesh",
        dest="noh",
        action="store_false",
        default=True,
        help="If set, hydrogen atoms will be considered for all RMSD and dihedral angle computations. (default: ignore hydrogens)",
    )
    mbuilder.add_argument(
        "-s",
        "--s",
        "-sort",
        "--sort",
        dest="sort",
        action="store_true",
        default=False,
        help="If set, will attempt to sort molecular geometries within isomorphisms w.r.t. the first structure. This can be time consuming. (default: False)",
    )
    mbuilder.add_argument(
        "-nosymm",
        "--nosymm",
        dest="nosymm",
        action="store_true",
        default=False,
        help="If set, will avoid sorting molecular geometries within isomorphisms unless its requested explicitly with the allsort/sort flags. (default: False)",
    )
    mbuilder.add_argument(
        "-as",
        "--as",
        "-allsort",
        "--allsort",
        "-truesort",
        "--truesort",
        dest="allsort",
        action="store_true",
        default=False,
        help="If set, will attempt to sort molecular geometries within isomorphisms in every pairwise comparison. This is extremely time consuming in large molecules. (default: False, True if molecule is small)",
    )
    mbuilder.add_argument(
        "-o",
        "-output",
        "--o",
        "--output",
        dest="output_filename",
        default=None,
        help="If set to a filename, filename of the output file that will log marc's standard output. (default: None)",
    )
    mbuilder.add_argument(
        "-efile",
        "--efile",
        dest="efile",
        default=None,
        help="If set to a filename, file containing the energies of each conformer in crest format. (default: None)",
    )
    mbuilder.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=1,
        help="Verbosity level of the code. Higher is more verbose and viceversa. (default: 1)",
    )
    mbuilder.add_argument(
        "-pm",
        "--pm",
        "-plotmode",
        "--plotmode",
        dest="plotmode",
        type=int,
        default=0,
        help="Plotting mode. Set to 1 to generate agglomerative dendrograms and to 2 to also generate TSNE plots. (default: 0)",
    )
    args = mbuilder.parse_args(arguments)

    if args.output_filename is not None:
        if not isinstance(args.output_filename, str) or re.search(
            r"[^A-Za-z0-9_\-\\]", args.output_filename
        ):
            raise InputError(
                f"The provided output filename {args.output_filename} is invalid!"
            )
        orig_stdout = sys.stdout
        ofile = open(args.output_filename, "w")
        sys.stdout = ofile
    print(f"Executing marc version {version_str} with input arguments:\n{input_str}")
    input_list = [item for sublist in args.input for item in sublist]
    if args.verb > 2:
        print(f"Input files are {input_list}.")
    if len(input_list) > 1:
        terminations = list(set([i[-3:] for i in input_list]))
        basename = long_substr([filename.split("/")[-1] for filename in input_list])
        if len(terminations) > 1 or terminations[0] != "xyz":
            raise InputError(
                f"Files with {terminations} instead of all xyz termination fed as input. Exiting."
            )
        molecules = [
            Molecule(filename=i, scale_factor=args.sf, noh=args.noh) for i in input_list
        ]

    # This is left as a hook, but should basically never trigger due to argparse
    elif len(input_list) == 0:
        input_list = glob.glob("./*.xyz")
        terminations = list(set([i[-3:] for i in input_list]))
        basename = long_substr([filename.split("/")[-1] for filename in input_list])
        if len(terminations) > 1 or terminations[0] != "xyz":
            raise InputError(
                f"Files with {terminations} instead of all xyz termination fed as input. Exiting."
            )
        molecules = [
            Molecule(filename=i, scale_factor=args.sf, noh=args.noh) for i in input_list
        ]

    else:
        basename = input_list[0].split("/")[-1].split(".")[0]
        termination = input_list[0][-3:]
        if termination == "xyz":
            molecules = molecules_from_file(
                input_list[0], scale_factor=args.sf, noh=args.noh
            )
        else:
            raise InputError(
                f"File with {termination} instead of xyz termination fed as input. Exiting."
            )
    # Set energy window if requested
    if args.ewin is not None:
        try:
            ewin = float(args.ewin)
        except TypeError:
            raise InputError(
                f"ewin was set to {args.ewin} which is not a float nor None. Exiting."
            )
    else:
        ewin = None

    # Set energy file if provided
    if args.efile is not None:
        try:
            filename = str(args.efile)
        except TypeError:
            raise InputError(
                f"efile was set to {args.efile} which is not a string nor None. Exiting."
            )
        try:
            energies = []
            g = open(filename, "r")
            _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
            for line in g.readlines():
                trline = _RE_COMBINE_WHITESPACE.sub(" ", line).strip()
                e = float(trline.split(" ")[1])
                energies.append(e)
            g.close()
        except OSError:
            raise InputError(
                f"efile was set to {filename} which could not be found or opened to read energies. Exiting."
            )
        except TypeError:
            raise InputError(
                f"Line \n{line}\n did not satisfy the expected crest-like format. Exiting."
            )
        # Setting up energies in molecule objects
        if args.verb > 0:
            print(
                "Setting up energies from file. This will overwrite the energy values in the xyz files. Double check the ordering!"
            )
        if len(molecules) == len(energies):
            for i, (molecule, energy) in enumerate(zip(molecules, energies)):
                molecule.energy = energy
                if args.verb > 1:
                    print(f"Molecule {i} set with energy {energy}")
        else:
            raise InputError(
                f"The number of molecules ({len(molecules)}) and energies ({len(energies)}) in file {filename} does not match. Exiting."
            )

    # Check for number of molecules
    if len(molecules) < 3:
        raise InputError("Less than three molecules provided. Exiting.")

    # Check for atom ordering and number
    sort = False
    for molecule_a, molecule_b in zip(molecules, molecules[1:]):
        atoms_a = molecule_a.atoms_with_h
        atoms_b = molecule_b.atoms_with_h
        if len(atoms_a) == len(atoms_b):
            if not all(atoms_a == atoms_b):
                if args.verb > 4:
                    print(
                        "Attempting reordering and updating molecular geometries because atom lists were not identical."
                    )
                dview = reorder_distance(
                    atoms_a,
                    atoms_b,
                    molecule_a.coordinates_with_h,
                    molecule_b.coordinates_with_h,
                )
                dres, _ = kabsch_rmsd(
                    molecule_a.coordinates_with_h, molecule_b.coordinates_with_h[dview]
                )
                if args.verb > 4:
                    print(f"After distance reordering, RMSD is {dres}.")
                hview = reorder_hungarian(
                    atoms_a,
                    atoms_b[dview],
                    molecule_a.coordinates_with_h,
                    molecule_b.coordinates_with_h[dview],
                )
                hres, _ = kabsch_rmsd(
                    molecule_a.coordinates_with_h,
                    molecule_b.coordinates_with_h[dview][hview],
                )
                if args.verb > 4:
                    print(f"After hungarian reordering on top of it, RMSD is {hres}.")
                if hres < dres:
                    molecule_b.update_with_h(
                        atoms_b[dview][hview],
                        molecule_b.coordinates_with_h[dview][hview],
                    )
                else:
                    molecule_b.update_with_h(
                        atoms_b[dview], molecule_b.coordinates_with_h[dview]
                    )
        else:
            raise InputError("Molecules do not have the same number of atoms. Exiting.")
        natoms = len(atoms_b)
        if natoms == 1:
            raise InputError("Molecules are monoatomic. Exiting.")
        elif natoms == 2:
            dof = 1
        else:
            dof = 3 * len(atoms_b) - 6
        if not dof > 0:
            raise InputError("Molecules have less than 1 degree of freedom. Exiting.")
    if args.verb > 0 and sort:
        print(
            "Warning! The given molecule geometries were not sorted. Molecules have been reordered to try to fix this.\n This does not guarantee that the RMSDs will be properly calculated! If your input structures are completely shuffled, all calculated RMSD-based metrics might be bad."
        )

    # Check for isomorphism
    for molecule_a, molecule_b in zip(molecules, molecules[1:]):
        g_a = molecule_a.graph
        g_b = molecule_b.graph

        GM = nx.algorithms.isomorphism.GraphMatcher(g_a, g_b, at_eq, b_eq)
        if GM.is_isomorphic():
            im = True
        else:
            if args.verb > 0:
                print("Warning! Molecule topologies are not isomorphic.")
            im = False
            break

    # Double check if energies are properly set
    energies = [molecule.energy for molecule in molecules]
    if None in energies:
        if args.verb > 2:
            print(f"Energies are: {energies}")

    # Check input args typing/values
    if args.c not in valid_c:
        raise InputError(
            f"Unknown clustering algorithm selected. Valid algorithms are:\n {valid_c}\n Exiting."
        )

    if args.m not in valid_m:
        raise InputError(
            f"Unknown metric for clustering selected. Valid metrics are:\n {valid_m}\n Exiting."
        )

    if args.n is not None:
        try:
            n = int(args.n)
        except ValueError:
            raise InputError(
                f"n must be an integer or None, but {n} was provided. Exiting."
            )
    else:
        n = None

    if args.sort != sort:
        sort = args.sort
    if args.allsort:
        if args.verb > 0:
            print(
                "Warning! All pairs isomorphism-based sorting is remarkably expensive. This option is available for cases in which the input structures are isomorphically shuffled.\n This may take a ridiculous amount of time, as all isomorphisms must be checked, so you may want to explore other options or use a metric that does not require RMSDs."
            )
        truesort = True
        sort = True
    elif natoms < 50 and not args.nosymm:
        if args.verb > 0:
            print(
                "Activating symmetry-aware isomorphic sorting because the system is small. This can be avoided using the nosymm flag."
            )
        truesort = True
        sort = True
    else:
        truesort = False
    if args.m in ["erel", "da", "ewda"]:
        sort = False
        truesort = False
        if args.verb > 0:
            print("Warning! Will not sort molecules because no RMSDs will be computed.")
    if ((sort and truesort) or sort) and not im:
        raise InputError(
            f"Input molecules cannot be reliably sorted.\n The automatic sorting routine will fail because the reconstructed molecular graphs were found not to be isomorphic. It might be possible to fix this by adapting the covalent radius scale factor of the graph constructor with the -sf option, but sometimes it cannot be fixed (e.g. different conformers have significantly different bond lengths). For those cases, you can use a metric that does not require RMSDs.\n Exiting."
        )

    return (
        basename,
        np.array(molecules, dtype=object),
        dof,
        args.c,
        args.m,
        n,
        ewin,
        args.mine,
        sort,
        truesort,
        args.plotmode,
        args.verb,
    )


def test_molecules_from_file(path="navicat_marc/test_files/"):
    filenames = [
        "3_h2o_conformers.xyz",
    ]
    for filename in filenames:
        molecules = molecules_from_file(f"{path}{filename}", noh=False)
        for molecule in molecules:
            print(molecule.energy, molecule.coordinates, molecule.atoms)
            assert molecule.energy is not None
