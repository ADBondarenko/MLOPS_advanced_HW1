import logging
from io import StringIO
from typing import Any, Dict, List, Union

import dvc.api
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


def read_data_test() -> Union[pd.DataFrame, pd.Series]:
    """
    Takes a predetermined TRAIN dataset from DVC remote storage (GDrive),
    separates dataset into X_train, y_train.
    Also creates an MLFlow experiment.

    Keyword arguments:
             None: None

    returns:
             X_train : pd.DataFrame,
             y_train : pd.Series
    """

    # Data download from GDrive
    # data_url = dvc.api.get_url(path="train.csv", remote="gdrive_mlops")
    data = dvc.api.read(
        "test.csv",
        repo="https://github.com/ADBondarenko/MLOPS_advanced_HW1",
        remote="mlops_gdrive",
        mode="r",
    )

    log.info("Data download complete")
    # Data preprocessing in accordance with specs
    df = pd.read_csv(StringIO(data))
    y_test = df.target
    X_test = df.drop(columns=["target"]).set_index("time")

    log.info("Data is split into X_train, y_train")

    return X_test, y_test


def predict_test(model: Any, X_test: pd.DataFrame):
    """
    Takes a pretrained model, data and outputs an
    array of y_pred. Wrapper of model.predict()
    Keyword arguments:
            model : Any -- pretrained regression model of
                           any sklearn class implemented
            X_test : pd.DataFrame,

    returns:
            y_pred : np.ndarray | List | pd.Series
    """
    log.info("Prediction started...")

    y_pred = model.predict(X_test)
    log.info("Prediction succesfull...")
    return y_pred


def evaluate_model_test(
    model: Any, X_test: pd.DataFrame, y_test: Union[pd.Series, List, np.ndarray]
) -> Dict:
    """
    A redundant func for outputting model results.
    Takes a pretrained model, data and outputs a
    Dict of refression quality metrics.

    Keyword arguments:
            model : Any -- pretrained regression model of
                           any sklearn class implemented
            X_train : pd.DataFrame,
            y_train : Union[pd.Series, List, np.ndarray]
    returns:
            metrics: Dict = {"mean_squared_error" : float
                             "r2_score" : float
                             "median_absolute_error": : float}
    """
    log.info("Starting model evalutation...")
    y_true = y_test
    y_pred = model.predict(X_test)

    results_dict = {
        "mean_squared_error_test": mean_squared_error(y_true=y_true, y_pred=y_pred),
        "r2_score_test": r2_score(y_true=y_true, y_pred=y_pred),
        "median_absolute_error_test": median_absolute_error(
            y_true=y_true, y_pred=y_pred
        ),
    }
    log.info("Model evaluation succesfull!")
    return results_dict


if __name__ == "__main__":
    # For linter to ignore unsued names
    log = logging.getLogger(__name__)
    log.addHandler(logging.StreamHandler())
    raise ValueError("Not to be run separately from train.py")
