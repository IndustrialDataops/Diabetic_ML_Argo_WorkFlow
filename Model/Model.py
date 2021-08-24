import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pickle
import argparse
import logging

param_grid = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 5, 6, 7, 8],
    "criterion": ["gini", "entropy"],
}

parser = argparse.ArgumentParser(description="Model")
parser.add_argument(
    "--input", "-i", required=False, help="Input file location"
)
parser.add_argument("--target", "-t", required=False, help="Target Location")
args = parser.parse_args()


def trainModel(input:str,target:str):
    data = pd.read_parquet(input)
    df = data[["systolic", "diastolic", "hdl", "ldl", "bmi", "age", "diabetic"]]
    df.sample(frac=1)
    X = df.drop("diabetic", axis=1)
    y = df["diabetic"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfc = RandomForestClassifier(random_state=42)
    gs_clf = GridSearchCV(rfc, param_grid=param_grid, cv=5, n_jobs=3, verbose=True)
    gs_clf.fit(X_train, y_train)
    logging.info("Model Score :{0}".format(gs_clf.score(X_test, y_test)))
    logging.info("Saving the model")
    pickle.dump(gs_clf, open(target, "wb"))
    return None


if __name__ == "__main__":
    logging.info("Starting the job")
    input = args.input
    target = args.target
    trainModel(input,target)
    logging.info("Completed")
