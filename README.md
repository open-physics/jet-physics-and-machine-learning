# jet-physics-and-machine-learning

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
