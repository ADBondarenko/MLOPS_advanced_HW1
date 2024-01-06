import logging

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from train_infer.infer import evaluate_model_test, read_data_test

# multiple imports invoke coflicts in black and isort -
# they want to format them differenrently - see
# https://ice-panda.medium.com/pre-commit-hooks-claim-files-are-modified-but-no-changes-are-detected-by-git-c43217f51d8a
# this routine is disabled for that purpose
# fmt: off
from train_infer.train import (HistGradientBoostingRegressorConfig,  # isort:skip
                               LassoRegressionConfig, LinearRegressionConfig,  # isort:skip
                               evaluate_model, init, read_data, train_model)  # isort:skip

# Initializing a logger globally, code will break
# otherwise when called outside of '__main__'

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
# fmt: on


@hydra.main(config_path="configs", config_name="config")
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
        X_test, y_test = read_data_test()
        model_ = train_model(cfg, X_train, y_train)
        log.info("Started logging model...")
        mlflow.sklearn.log_model(model_, f"{exp_id}_model")
        log.info("Logging model succesfull")
        metrics_train = evaluate_model(model_, X_train, y_train)
        metrics_test = evaluate_model_test(model_, X_test, y_test)
        log.info("Started logging train metrics...")
        for metric in list(metrics_train.keys()):
            mlflow.log_metric(f"{metric}", metrics_train[metric])
        log.info("Logging train metrics succesfull")
        log.info("Started logging test metrics...")
        for metric in list(metrics_test.keys()):
            mlflow.log_metric(f"{metric}", metrics_test[metric])
        log.info("Logging test metrics succesfull")
    trained_model_dict = {
        "exp_id": exp_id,
        "model": model_,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
    }
    return trained_model_dict


if __name__ == "__main__":
    # Registering the configuration.
    cs = ConfigStore.instance()
    cs.store(group="models", name="default", node=LinearRegressionConfig)
    cs.store(group="models", name="default", node=LassoRegressionConfig)

    cs.store(group="models", name="default", node=HistGradientBoostingRegressorConfig)
    log.info("Hydra configs saved to ConfigStore")
    trained_model_dict = main()
    log.info("Training finished")
