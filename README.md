# [GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability](https://drive.google.com/file/d/1mGO-rLOZ-_TXu_-8KIfSUiFEqymxs2x5/view)

## Disclaimer

This is code associated with the paper ["GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability,"](https://drive.google.com/file/d/1mGO-rLOZ-_TXu_-8KIfSUiFEqymxs2x5/view) published in the [NeurIPS](https://nips.cc/) Workshop on [Differential Geometry meets Deep Learning 2020](https://sites.google.com/view/diffgeo4dl/).

**THIS REPOSITORY WILL BE UPDATED WITH FURTHER DOCUMENTATION SOON!!**

If you have any questions, please feel free to reach out to us or make an issue.

## Preliminaries

Our package is designed to run in Python 3.6 and pip version 20.2.4....

```
pip install -r requirements.txt
```

## Usage

To use our package...

```
<details on library import and script execution go here>
```

How saving is done:

Results are expected to saved in specific locations. If this code is not used to create equivalences classes, but the plotting functions want to be used, we advise to follow the structure laied out in get_grid.py and simply use the methods in interpolation.py which are agnostic to the saved locations.

### Run experiment.py to produce elements in equivalence classes

* To check if the elements converged to elements in the equivalence class, run stats_plotting.
* Run the griding code to produce a set of elements in a subspace spanned by elements that were found.
* Subset the set by elements wiht loss less than some epsilon and choose an appropriate plotting mechanism.

### Getting directories and run IDs

After creating an experiment this will be dumped to **GENNI_HOME/experiment** where **GENNI_HOME** is set in the **genni.yml** file. An easy way to get the experiment directory and the run ids is to use the _tree_ command line argument as follows:

```sh
tree $GENNI_HOME/experiments -d -L 3
```

An example output looks like

```sh
experiments
└── Nov09_19-52-12_isak-arch
    ├── models
    │   └── 1604947934.637504
    └── runs
        └── 1604947934.637504
```

where `Nov09_19-52-12_isak-arch` is the identifier of the experiment and
`1604947934.637504` is an ID of a hyperparameter setting of this experiment.

## Citing

If you use GENNI anywhere in your work, please cite use using

```
@article{2020,
    title={GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability},
    author={Lengyel, Daniel and Petangoda, Janith and Falk, Isak and Highnam, Kate and Lazarou, Michalis and Kolbeinsson, Arinbjörn and Deisenroth, Marc Peter and Jennings, Nicholas R.},
    booktitle={NeurIPS Workshop on Differential Geometry meets Deep Learning},
    year={2020}
}
```
