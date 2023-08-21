# jet-physics-and-machine-learning


<p align="center">
<!-- <a href="https://github.com/sparmar24/jet-physics-and-machine-learning/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a> -->
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10.2-brightgreen"></a>
<a href="https://matplotlib.org/stable/"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.6-green"></a>
<a href="https://github.com/impy-project/chromo/"><img alt="CHROMO" src="https://img.shields.io/badge/chromo-brightgreen"></a>
<a href="https://github.com/scikit-hep/fastjet/"><img alt="Fastjet" src="https://img.shields.io/badge/fastjet-brightgreen"></a>
<a href="https://learn.microsoft.com/en-us/azure/machine-learning/"><img alt="Machine Learning" src="https://img.shields.io/badge/Machine%20Learning-Yes-blue"></a>
<a href="https://github.com/psf/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://scikit-learn.org/stable/"><img alt="scikit-learn" src="https://img.shields.io/badge/Scikit--learn-1.2.2-orange"></a>
<a href="https://github.com/pdbpp/pdbpp"><img alt="pdbpp" src="https://img.shields.io/badge/pdbpp-yes-pink"></a>
<a href="https://pypi.org/project/pylint/"><img alt="pylint" src="https://img.shields.io/badge/pylint-2.17.1-blue"></a>
</p>


## A. Dev Environment
Set up your dev environment by following instructions [here](dev-environment-readme.md).

## B. Physics Environment
For your physics environment, you need
1. [PYTHIA](https://www.pythia.org/): You can interact with this and other models by using [CHROMO: A Hadronic Interaction Model interaface in Python](https://github.com/impy-project/chromo).
2. [FastJet](https://github.com/scikit-hep/fastjet): You need this for its algorithms to cluster jets.
3. [Uproot](https://github.com/scikit-hep/uproot5): You need this for reading and writing ROOT files in pure Python and NumPy.

All the above packages are available at PyPI and are pip-installed via pyproject.toml at [dev environment](#a-dev-environment).

## C. Physics Workflow in brief
1. [Generate pp collision events from Pythia.](src/generate_pp_events_with_pythia.py)
2. [Reconstruct jets using anti-kt algorithm from fastjet and tag them with necessary heavy flavours.](src/reconstruct_and_tag_jets.py)
3. [Apply different ML algorithms to classify the reconstructed jets.](src/jet_classification_ML.py)
4. Find the best ML model with accuracy score, confusion matrix, f1 score etc.
5. Optimize the model by varying different parameters.
