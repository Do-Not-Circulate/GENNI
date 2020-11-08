# GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability

## Disclaimer

This is code associated with the paper "GENNI: Visualising the Geometry of
Equivalences for Neural Network Identifiability," published in the
[NeurIPS](https://nips.cc/) Workshop on [Differential Geometry meets Deep
Learning 2020](https://sites.google.com/view/diffgeo4dl/).

## Preliminaries

Our package is designed to run in Python 3.6 and pip version 20.2.4....

```
pip install -r requirements.txt
```

## Usage

To use our package...

```
>>> from GENNI import genni_vis
>>> plot = genni_vis(mesh, V, X, Y, eigenpairs)
>>> plot.optimize()
```

```
python demo_dragon.py --help
usage: demo_dragon.py [-h] [--num-eigenpairs NUM_EIGENPAIRS] [--seed SEED]
                      [--output-dir OUTPUT_DIR]
                      [--eigenpairs-file EIGENPAIRS_FILE] [--mayavi]
                      [--num-samples NUM_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  --num-eigenpairs NUM_EIGENPAIRS
                        Number of eigenpairs to use. Default is 500
  --seed SEED           Random seed
  --output-dir OUTPUT_DIR
                        Output directory to save .pvd files to. Default is
                        ./output
  --eigenpairs-file EIGENPAIRS_FILE
                        .npy file with precomputed eigenpairs
  --mayavi              Render results to .png with Mayavi
  --num-samples NUM_SAMPLES
                        Number of random samples to generate
```

How saving is done:

Results are expected to saved in specific locations. If this code is not used to create equivalences classes, but the plotting functions want to be used, we advise to follow the structure laied out in get_grid.py and simply use the methods in interpolation.py which are agnostic to the saved locations.

### Run experiment.py to produce elements in equivalence classes

### To check if the elements converged to elements in the equivalence class, run stats_plotting.

### Run the griding code to produce a set of elements in a subspace spanned by elements that were found.

### Subset the set by elements wiht loss less than some epsilon and choose appropriate plotting mechanism.

## Reproducing the paper

- [ ] How to reproduce figures
- [ ] How to reproduce values

## Citing

If you use GENNI anywhere in your work, please cite use using

```
@article{2020,
    title={GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability},
    author={},
    booktitle={},
    year={2020}
}
```

## TODO LIST

- [x] Licence (MIT)
- [ ] Documentation
- [ ] Github actions
  - Contributing
  - Pull request / Issues templates
- [ ] Put on PyPI
- [ ] Make environment?
- [ ] CI
- [ ] Make package conform to PEP and packaging standards
