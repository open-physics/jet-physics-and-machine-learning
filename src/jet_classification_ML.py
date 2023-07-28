import os

import numpy as np
import pandas as pd
import uproot
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from xgboost import XGBClassifier


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
        exit()
    return dataset


def pred_score(ps_estimator, ps_train_X, ps_test_X, ps_train_y, ps_test_y):
    # predict accuracy score for each classification model
    ps_estimator.fit(ps_train_X, ps_train_y)
    prediction = ps_estimator.predict(ps_test_X)
    confusion = confusion_matrix(ps_test_y, prediction)
    accuracy = accuracy_score(ps_test_y, prediction)
    return prediction, confusion, accuracy


def best_estimator_parameter_accuracy(b_estimator, b_train_X, b_train_y):
    parameters = [
        {"C": [0.25, 0.5, 0.75, 1], "kernel": ["linear"]},
        {
            "C": [0.25, 0.5, 0.75, 1],
            "kernel": ["rbf"],
            "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    ]
    grid_search = GridSearchCV(
        estimator=b_estimator,
        param_grid=parameters,
        scoring="accuracy",
        cv=10,
        n_jobs=-1,
    )
    grid_search.fit(b_train_X, b_train_y)
    b_accuracy = grid_search.best_score_
    b_estimator = grid_search.best_estimator_
    b_parameters = grid_search.best_params_
    return b_estimator, b_accuracy, b_parameters


def main():
    dataset = load_dataset("tagged_D0meson_jets.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

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
        # "xgboost": XGBClassifier(),
    }
    # breakpoint()

    for estimator_name, estimator in estimators.items():
        prediction, confusion, accuracy = pred_score(
            estimator, train_X, test_X, train_y, test_y
        )

        models_and_predictions[f"{estimator_name}"] = {
            "prediction": prediction,
            "confusion": confusion,
            "accuracy": accuracy,
        }

    """ Will check later """
    # # svc_estimator = SVC(kernel="rbf", random_state=0)
    # # b_estimator, b_accuracy, b_parameters = best_estimator_parameter_accuracy(svc_estimator, train_X, train_y)
    # breakpoint()

    # svc = SVC(kernel="rbf", random_state=0)
    # svc.fit(train_X, train_y)
    # y_pred = svc.predict(test_X)
    # print(y_pred)
    #
    # from sklearn.metrics import confusion_matrix, accuracy_score
    # cm = confusion_matrix(test_y, y_pred)
    # accuracy = accuracy_score(test_y, y_pred)
    # print(cm)
    # print(accuracy)
    #
    # # # applying k-Fold cross validation
    # # from sklearn.model_selection import cross_val_score
    # # accuracies = cross_val_score(estimator=svc, X=train_X, y = train_y, cv = 10)
    # # print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    # # #print("Standard Deviation: {.2f} %".format(accuracies.std()*100))
    # # print("k-fold=10 gives accuracy score:", accuracies)
    # # print("best :", sum(accuracies)/len(accuracies))
    # # # print(accuracies.std()*100)
    #
    # # applying grid search technique to find best model and best parameters
    # from sklearn.model_selection import GridSearchCV
    # parameters = [{"C" : [0.25, 0.5, 0.75, 1], "kernel" : ["linear"]},
    #               {"C" : [0.25, 0.5, 0.75, 1], "kernel" : ["rbf"], "gamma" : [0.1, 0.2, 0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    # grid_search = GridSearchCV(estimator = svc,
    #                            param_grid = parameters,
    #                            scoring = "accuracy",
    #                            cv = 10,
    #                            n_jobs = -1)
    # grid_search.fit(train_X, train_y)
    # best_accuracy = grid_search.best_score_
    # best_estimator = grid_search.best_estimator_
    # best_parameters = grid_search.best_params_
    # print("best_accuracy:", best_accuracy)
    # print("best_estimator:", best_estimator)
    # print("best_parameters: ", best_parameters)


if __name__ == "__main__":
    load_root_file()
    root_to_csv()
    main()
