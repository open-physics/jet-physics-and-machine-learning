# jet-physics-and-machine-learning

## A. Dev Environment
Set up your dev environment by following instructions [here](https://github.com/open-physics/jet-physics-and-machine-learning/blob/master/dev-environment-readme.md).

## B. Physics Environment
For your physics environment, you need
1. PYTHIA: Download pythia from https://www.pythia.org/.
2. Uproot and related others: These are pip-installed via pyproject.toml at [dev environment](#markdown-header-a-dev-environment).

## C. Physics Workflow in brief
1. Generate pp collision events from Pythia.
2. Reconstruct jets using jet algorithms.
3. Apply different ML algorithms to classify the reconstructed jets.
4. Find the best ML model with accuracy score, confusion matrix, f1 score etc.
5. Optimize the model by varying different parameters.
