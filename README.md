navicat-marc: modular Analysis of Representative Conformers
==============================================
<!-- zenodo badge will go here -->

![marc logo](./images/marc_logo.png)
[![PyPI version](https://badge.fury.io/py/navicat-marc.svg)](https://badge.fury.io/py/navicat_marc)

## Contents
* [About](#about-)
* [Install](#install-)
* [Concept](#concept-)
* [Examples](#examples-)
* [Citation](#citation-)

## About [↑](#about)

The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `networkx`


## Install [↑](#install)

You can install marc using pip:

```python
pip install navicat_marc
```

Afterwards, you can call marc as:

```python 
python -m navicat_marc [-h] [-version] -i INPUT [INPUT ...] [-c C] [-m M] [-n N] [-ewin EWIN] [-sf SF] [-mine] [-yesh] [-s] [-ts] [-efile EFILE] [-v VERB] [-pm PLOTMODE]
```
or simply

```python 
navicat_marc [-h] [-version] -i INPUT [INPUT ...] [-c C] [-m M] [-n N] [-ewin EWIN] [-sf SF] [-mine] [-yesh] [-s] [-ts] [-efile EFILE] [-v VERB] [-pm PLOTMODE]
```

Alternatively, you can download the package and execute:

```python 
python setup.py install
```

Afterwards, you can call marc as:

```python 
python -m navicat_marc [-h] [-version] -i INPUT [INPUT ...] [-c C] [-m M] [-n N] [-ewin EWIN] [-sf SF] [-mine] [-yesh] [-s] [-ts] [-efile EFILE] [-v VERB] [-pm PLOTMODE]
```
or

```python 
navicat_marc [-h] [-version] -i INPUT [INPUT ...] [-c C] [-m M] [-n N] [-ewin EWIN] [-sf SF] [-mine] [-yesh] [-s] [-ts] [-efile EFILE] [-v VERB] [-pm PLOTMODE]
```

Options can be consulted using the `-h` flag in either case. The help menu is quite detailed. 

Note that the main functions are all exposed and called directly in sequential order from `marc.py`, in case you want to incorporate them in your own code.

## Concept [↑](#concept)

Several strategies are available for the generation of conformational ensembles. Typically, one then needs to sort the ensemble and proceed with the study of the most energetically favored conformers, which will be the most accesible thermodynamically following a Boltzmann distribution.

However, sorting conformers accurately requires high quality energy computations. Accurately determining the energy of every structure may be too computationally demanding. Hence, marc provides a convenient way of accomplishing three goals:

- Select a handful of conformers that are representative of the diversity of the conformational ensemble using combined metrics.
- Apply energy cutoffs based on the available energies to remove entire clusters from the space using the `-ewin` flag and inputting a treshold in kcal/mol.
- Proceed iteratively, helping the user select non-redundant conformers than can then be refined with a higher level and fed back to marc.

The default clustering metric used in marc is the `"mix"` distance, which measures pairwise similarity based on heavy-atom rmsd times the energy difference times the kernel of the heavy-atom dihedral angles of the system. 

The logic behind this choice is that rmsd ought to be good except in cases where trivial single bond rotations increase the rmsd without affecting the energy, while the dihedral metric smooths systems that only differ by a few torsions. The possible metrics (to be fed to the `-m` flag) are `"rmsd"`, `"erel"` (based on the available energies), `"da"` (based on the most relevant dihedral angle of the molecule), `"ewrmsd"` (combining geometry and energy) and `"mix"` (combining geometry, dihedrals and energy).  

## Examples [↑](#examples)

The examples subdirectory contains some examples obtained by running [CREST](https://xtb-docs.readthedocs.io/en/latest/crest.html). Any of the xyz files can be run as:

```python
navicar_marc -i [FILENAME]
```

Options can be consulted with the `-h` flag.

The input of marc is either a series or xyz files or a single trajectory-like xyz file with many conformers. All structures are expected to be analogous in terms of sorting and molecular topology. Energies per conformer, at any level of theory of your liking but in atomic units, can be provided in atomic units in the title line of each xyz block or file. Alternatively, energies can be provided in a plaintext file whose filename can be passed to the `ewin` command line argument. Such file must contain the same number of lines as conformers and two numbers per line (separated by blank spaces): an index, and an energy in atomic units. The energy window specified in the `ewin` command line argument should be in kcal/mol.

Note that, by default, marc will select the most representative conformer out of every cluster. If you can provide energy values that you trust strongly, the `mine` flag will ensure that the lowest energy conformer of every cluster is selected.

The output of marc are `n` selected xyz files which will be called `INPUT_selected_n.xyz` in the runtime directory. Conformers discarded by the `ewin` threshold will be printed with the `rejected` appendix instead. The discarding checks two criteria: if a cluster has an average energy that is `mine` kcal/mol higher than the lowest conformer (plus half a standard deviation), and its lowest energy member is also higher than the threshold, the entire cluster will be discarded.

High verbosity levels (`-v 1`, `-v 2`, etc.) will print significantly more information while marc runs. To be as automated as possible, reasonable default values are set for most choices, but extreme verbosity can be obtained by raising the value.

As a final note, marc does not consider hydrogen atoms for geometry analysis. You can force marc to include them by using the `yesh` flag.

## Citation [↑](#citation)

Please cite our work with the repository DOI.

---


