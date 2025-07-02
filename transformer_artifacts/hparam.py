from fit import train_character_transformer

import mlflow
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback


def tune_with_callback(mlflow_tracking_uri):
    tuner = tune.Tuner(
        train_character_transformer,
        run_config=train.RunConfig(
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name="basic_gpt2-style_character_transformer",
                    save_artifact=True,
                )
            ],
        ),
        param_space={
            "dropout": tune.grid_search([0.05, 0.1, 0.2, 0.25]),
            "n_embed": tune.grid_search([32, 64, 128]),
        },
    )
    results = tuner.fit()


mlflow_tracking_uri = "http://127.0.0.1:8080"
tune_with_callback(mlflow_tracking_uri)


experiment = mlflow.get_experiment_by_name("basic_gpt2-style_character_transformer")
if experiment:
    experiment_id = experiment.experiment_id
    df = mlflow.search_runs([experiment_id])
    print(df)
else:
    print("Experiment not found.")
