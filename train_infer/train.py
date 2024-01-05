import logging
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Union

import dvc.api
import hydra
import mlflow
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoLars, LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


@dataclass
class LinearRegressionConfig:
    fit_intercept: bool = True
    normalize: bool = False


@dataclass
class LassoRegressionConfig:
    alpha: float = 1.0
    fit_intercept: bool = True
    max_iter: int = 1000


@dataclass
class HistGradientBoostingRegressorConfig:
    loss: str = "squared_error"
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.0


def init() -> str:
    """
    Initializes MLFlow Tracking on the local server.
    Experiment name is in format of "%Y%m%d %H:%M:%S"+task_name.

    Keyword arguments:
            None

    returns:
            None
    """

    # Создаём эксперимент
    timestamp_ = str(datetime.now().strftime("%Y%m%d %H:%M:%S"))
    exp_name = f"{timestamp_}_mlops_adv_hw"
    uri_ = "https://128.0.1.1:8080"
    mlflow.create_experiment(exp_name, artifact_location=uri_)
    mlflow.set_experiment(exp_name)
    mlflow.set_registry_uri(uri_)
    return exp_name


def read_data() -> Union[pd.DataFrame, pd.Series]:
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
    data = dvc.api.read("train.csv", remote="mlops_gdrive", mode="r")

    log.info("Data download complete")
    # Data preprocessing in accordance with specs
    df = pd.read_csv(StringIO(data))
    y_train = df.target
    X_train = df.drop(columns=["target"])

    log.info("Data is split into X_train, y_train")

    return X_train, y_train


def train_model(
    cfg: DictConfig, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray]
):
    """
    Trains a linear model for regression problem
    Supports sklearn LassoLars,
    LinearRegression, HistGradientBoostingRegressor
    regression implementaion with aliases:
    ["hist_gb_reg", "lin_reg", "lasso_reg"].
    Input kwarg 'model_name' will be wrapped as Enum in the future.
    Returns a trained model.

    Keyword arguments:
    cfg: Union[LinearRegressionConfig,
               LassoRegressionConfig,
               HistGradientBoostingRegressorConfig] -- Hydra config for models
    X_train : pd.DataFrame -- train features data
    y_train : Union[pd.Series, List, np.ndarray],
              (pd.Series | List | np.ndarray) -- train target data

    returns:  model : sklearn.base.BaseEstimator
    """
    # Initialize model:

    model_name = cfg.model
    if model_name == "hist_gb_reg":
        params = {k: v for k, v in cfg.models.boosting_reg}
        model = HistGradientBoostingRegressor(**params)

    elif model_name == "lin_reg":
        params = {k: v for k, v in cfg.models.lin_reg}
        model = LinearRegression(**params)

    elif model_name == "lasso_reg":
        params = {k: v for k, v in cfg.models.lasso_reg}
        model = LassoLars(**params)

    else:
        log.info(f"Support for the model {model_name} is not added")
        raise ValueError("Model not implemented")

    # Train model:
    model.fit(X_train, y_train)

    log.info("Model succesfully fit")

    return model


def evaluate_model(
    model: Any, X_train: pd.DataFrame, y_train: Union[pd.Series, List, np.ndarray]
) -> Dict:
    """
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

    y_true = y_train
    y_pred = model.predict(X_train)

    results_dict = {
        "mean_squared_error_train": mean_squared_error(y_true=y_true, y_pred=y_pred),
        "r2_score_train": r2_score(y_true=y_true, y_pred=y_pred),
        "median_absolute_error_train": median_absolute_error(
            y_true=y_true, y_pred=y_pred
        ),
    }

    return results_dict


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main routine for train.py. To be used with infer.py in the
    train_infer.py script hence the func returns all of the
    callbacks - to avoid using external
    S3/GDrive storage and experiment/parameter tracking.

    Logs MLFlow Metrics upon calling.
    This is the fallback of the procedural programming, unfortunately.
    Keyword arguments:\
            cfg: DictConfig - inputs for a Hydra decorator
    returns:
            exp_id : str -- exp_id for MLFlow
            model_ : Union[sklearn.ensemble._hist_gradient_boosting.\
                           gradient_boosting.HistGradientBoostingRegressor,
                           sklearn.linear_model._base.LinearRegression,
                           sklearn.linear_model._least_angle.LassoLars]
                           -- trained model
            metrics : dict -- a dictionary of metrics
    """
    exp_id = init()
    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg.models, resolve=True))
        X_train, y_train = read_data()
        model_ = train_model(cfg, X_train, y_train)

        mlflow.save_model(cfg.save, model_)
        mlflow.sklearn.log_model(model_, f"{exp_id}_model")

        metrics = evaluate_model(model_)
        for metric in list(metrics.keys()):
            mlflow.log_metric(f"{metric}", metrics[metric])

    return exp_id, model_, metrics


if __name__ == "__main__":
    # Initializing a logger
    log = logging.getLogger(__name__)
    log.addHandler(logging.StreamHandler())

    # Registering the configuration.
    cs = ConfigStore.instance()
    cs.store(group="models", name="default", node=LinearRegressionConfig)
    cs.store(group="models", name="default", node=LassoRegressionConfig)

    cs.store(group="models", name="default", node=HistGradientBoostingRegressorConfig)
    log.info("Hydra configs saved to ConfigStore")
    exp_id, model_, metrics = main()
    log.info("Training finished")
