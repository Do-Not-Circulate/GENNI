# [GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability](https://drive.google.com/file/d/1mGO-rLOZ-_TXu_-8KIfSUiFEqymxs2x5/view)

## Disclaimer

This is code associated with the paper ["GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability,"](https://drive.google.com/file/d/1mGO-rLOZ-_TXu_-8KIfSUiFEqymxs2x5/view) published in the [NeurIPS](https://nips.cc/) Workshop on [Differential Geometry meets Deep Learning 2020](https://sites.google.com/view/diffgeo4dl/).

If you have any questions, please feel free to reach out to us or make an issue.

## Installing

Genni is available from PyPI [here](https://pypi.org/project/genni/). In order
to install simply use `pip`

```sh
pip install genni
```

## Usage

In order to use the package, please set `genni.yml` in the top directory of your
project and add / set the variable `genni_home` pointing to where genni should keep
all of the generated files.

### Generating a run

In order to calculate the approximate equivalence classes of parameters of your
network architecture that leads to the same function you first need to create an
experiment. An example file of how to do this can be found in
`scripts/experiment.py` which has some architectures predefined, but you can add
your own if you want to by looking at how the file is designed.

Generating an experiment can be done by calling

```
python scripts/experiment.py
```

### Getting directories and run IDs

After generating an experiment this will populate `${GENNI_HOME}/experiment`
with a directory having as a name the timestamp of when it was run. An easy way
to look at the generated experiments is use the `tree` command. Below is an
example output when running this after generating a couple of experiments

```sh
tree $GENNI_HOME/experiments -d -L 3
```

with the output

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

## Plotting

We have prepared a notebook called `notebooks/SubspaceAnalysis.ipynb` showing
how to

- Load your experiment together with necessary paths and experiment ids
- Compute grids and values for plotting
- Different ways of visualising the approximate equivalence classes in the form
  of a
  - Contour plot
  - 3d iso-surface plot
  - UMAP projected 2d plot of 3d iso-surface

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
