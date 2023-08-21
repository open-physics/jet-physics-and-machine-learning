import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import uproot
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def flatten_list(list_to_flatten):
    flattened_list = [element for sublist in list_to_flatten for element in sublist]
    return flattened_list


def load_root_file():
    """define project directory path"""
    workdir = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    # define path to root file
    file_root = uproot.open(f"{workdir}/data/uproot_jet_tagging.root")
    return file_root


def root_to_csv():
    workdir = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )

    file_root = load_root_file()
    events = file_root.keys()
    event_dataframes = []
    start = 0
    end = 0
    for event in events:
        dfs = []
        tree = file_root[event]
        for key in tree.keys():
            value = tree[key].array()
            end = len(value)
            data_frame = pd.DataFrame(
                {key: value}, index=list(np.arange(start, start + end))
            )
            dfs.append(data_frame)
        start += end
        event_dataframes.append(pd.concat(dfs, axis=1))
    # Concat all event-by-event data frames
    all_df = pd.concat(event_dataframes)
    # Save dataframe to csv file format
    all_df.to_csv(f"{workdir}/data/tagged_D0meson_jets.csv")


def load_dataset(data_file):
    data_folder = "data"
    workdir = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    inputfile = os.path.join(workdir, data_folder, data_file)
    try:
        dataset = pd.read_csv(inputfile)
    except OSError as err:
        raise UserWarning(
            f"Your data file {data_file} does not exist "
            f"in {workdir}/{data_folder}.\n"
            f"Kindly, place your data file correctly."
        ) from err
    return dataset


def pred_score(ps_estimator, ps_train_X, ps_test_X, ps_train_y, ps_test_y):
    # predict accuracy score for each classification model
    ps_estimator.fit(ps_train_X, ps_train_y)
    prediction = ps_estimator.predict(ps_test_X)
    confusion = confusion_matrix(ps_test_y, prediction)
    accuracy = accuracy_score(ps_test_y, prediction)
    return prediction, confusion, accuracy


class DeepNeuralNetwork:
    def __init__(self, input_shape, output_shape, batch_size, epochs, verbose):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu", input_shape=input_shape),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(output_shape, activation="sigmoid"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, x_train, y_train):
        self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
        )

    def predict(self, x_test):
        predict = self.model.predict(x_test)
        threshold = 0.5
        prediction = np.where(predict >= threshold, 1, 0)
        return np.array(flatten_list(list(prediction)))

    def summary(self):
        return self.model.summary()


def main():
    dataset = load_dataset("tagged_D0meson_jets.csv")
    # Select jets with positive mass
    dataset = dataset[dataset["jets_mass_sq"] >= 0]
    # 1. Select charm jets
    dataset_charm = dataset[dataset["jets_charm"] == 1]
    # 2. Select an equal number of non-charm jets, in a randomized fashion
    dataset_non = dataset[dataset["jets_charm"] == 0].sample(len(dataset_charm))
    # Combine the two dataframes.
    combined_dataset = pd.concat([dataset_charm, dataset_non], ignore_index=True)
    # Randomly mix the rows in the combined dataframe.
    combined_dataset = combined_dataset.sample(frac=1, random_state=42).reset_index(
        drop=True
    )
    shuffled_dataset = pd.DataFrame(combined_dataset)

    X = shuffled_dataset.iloc[:, 1:-1].values
    y = shuffled_dataset.iloc[:, -1].values

    # Split the training and test data
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=40
    )

    # data preprocessing
    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    test_X = sc.transform(test_X)

    models_and_predictions = {}
    estimators = {
        "logistic": LogisticRegression(random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=10),
        "svc_linear": SVC(kernel="linear", random_state=0),
        "svc_rbf": SVC(kernel="rbf", random_state=0),
        "gaussian_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(criterion="entropy", random_state=0),
        "random_forest": RandomForestClassifier(
            criterion="entropy", n_estimators=10, random_state=0
        ),
        "dnn": DeepNeuralNetwork(
            input_shape=(8,), output_shape=1, batch_size=256, epochs=15, verbose=0
        ),
    }

    for estimator_name, estimator in estimators.items():
        prediction, confusion, accuracy = pred_score(
            estimator, train_X, test_X, train_y, test_y
        )

        models_and_predictions[estimator_name] = {
            "prediction": prediction,
            "confusion": confusion,
            "accuracy": accuracy,
        }
        if estimator_name == "dnn":
            logging.info(f"this:{estimator.summary()}")
            # logging.debug(f"this debug:{estimator.summary()}")
            # logging.warning(f"this warning:{estimator.summary()}")


if __name__ == "__main__":
    load_root_file()
    root_to_csv()
    main()
