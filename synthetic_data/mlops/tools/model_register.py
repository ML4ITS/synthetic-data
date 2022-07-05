from mlflow import get_experiment_by_name, register_model
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


class MlflowModelRegister:
    def __init__(self, experiment_name: str) -> None:
        self.experiment = self._set_experiment(experiment_name)
        self.client = MlflowClient()

    def _set_experiment(self, experiment_name: str) -> None:
        experiment = get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_name: str} does not exist")
        return experiment

    def register(self, model_name: str) -> None:
        # Find all trials in the experiment 'folder'
        experiment_id = self.experiment.experiment_id
        trial_runs = self.client.search_runs(experiment_ids=[experiment_id])

        # Decide what to filter by
        best_validation_loss = float("inf")

        top_trial = None
        for trial in trial_runs:

            if not trial.data.metrics:
                # no more data left
                break

            if "done" in trial.data.metrics:
                # check if the trials are completed/empty
                if bool(trial.data.metrics["done"]):
                    break

            if "best_run" not in trial.data.tags:
                if best_validation_loss > trial.data.metrics["validation_loss"]:
                    top_trial = trial
                    best_validation_loss = trial.data.metrics["validation_loss"]

        # finally
        self._save_top_trial(top_trial, model_name)

    def _save_top_trial(self, top_trial: Run, model_name: str) -> None:
        if top_trial is None:
            raise ValueError("Could not find best experiment run")
        run_id = top_trial.info.run_id
        register_model(f"runs:/{run_id}/model", model_name)
