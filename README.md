marc: Modular Analysis of Representative Conformers
==============================================
<!-- zenodo badge will go here -->

![marc logo](./images/marc_logo.png)

## Contents
* [About](#about-)
* [Install](#install-)
* [Examples](#examples-)
* [Citation](#citation-)

## About [↑](#about)

The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`


## Install [↑](#install)

Download and add marc.py to your path. No strings attached. Run as:

```python
python marc [-h] [-version] -i [INPUT] [-c C] [-m M] [-n N] [-ewin EWIN] [-efile EFILE] [-v VERB] [-pm PLOTMODE]
```

You can also execute:

```python 
python setup.py install
```

to install marc as a python module. Afterwards, you can call volcanic as:

```python 
python -m marc [-h] [-version] -i [INPUT] [-c C] [-m M] [-n N] [-ewin EWIN] [-efile EFILE] [-v VERB] [-pm PLOTMODE]
```

Options can be consulted using the `-h` flag in either case. The help menu is quite detailed. 

Note that the main functions are all exposed and called directly in sequential order from `marc.py`, in case you want to incorporate them in your own code.

## Examples [↑](#examples)

The examples subdirectory contains some examples in [CREST](https://xtb-docs.readthedocs.io/en/latest/crest.html) format. Any of the xyz files can be run as:

```python
python marc.py -i [FILENAME]
```

Options can be consulted with the `-h` flag.

The input of marc is either a series or xyz files or a single trajectory-like xyz file with many conformers. All structures are expected to be analogous in terms of sorting and molecular topology. Energies per conformer, at any level of theory of your liking, can be provided in the title line of each xyz block or file. Alternatively, energies can be provided in a plaintext file whose filename can be passed to the `ewin` command line argument. Such file must contain the same number of lines as conformers and two numbers per line (separated by blank spaces): an index, and an energy in any units. The energy window specified in the `ewin` command line argument should be in the same units (typically, kcal/mol).

The output of marc are `n` selected xyz files which will be called `INPUT_selected_n.xyz` in the runtime directory.

High verbosity levels (`-v 1`, `-v 2`, etc.) will print significantly more information while marc runs. To be as automated as possible, reasonable default values are set for most choices.


## Citation [↑](#citation)

Please cite our work with the repository DOI.

---


