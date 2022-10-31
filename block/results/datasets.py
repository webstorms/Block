import glob

import torch
import pandas as pd
from brainbox import trainer

import block.trainer


class BaseDatasetResultsBuilder:

    def __init__(self, models_root, dataset, batch_size=256, build_activity=True, hyperparams_mapper=None, model_loader=None):
        self._hyperparams_mapper = hyperparams_mapper
        self._model_loader = model_loader
        self._results_df = self._build_results_df(models_root, dataset, batch_size, build_activity)

    @property
    def results_df(self):
        return self._results_df

    def _build_results_df(self, models_root, dataset, batch_size, build_activity):
        model_ids = self._get_model_ids(models_root)

        accuracy_df = self._get_accuracy(models_root, model_ids, dataset, batch_size)
        average_duration_per_bath_df = self._get_average_duration_per_batch(models_root, model_ids)
        hyperparameters_df = trainer.build_models_df(models_root, model_ids, self._hyperparams_mapper)

        if build_activity:
            spike_counts_df = self._get_average_spikes_per_sample(models_root, model_ids, dataset, batch_size)
            results_df = accuracy_df.join(spike_counts_df).join(average_duration_per_bath_df).join(hyperparameters_df)
        else:
            results_df = accuracy_df.join(average_duration_per_bath_df).join(hyperparameters_df)

        results_df = results_df.reset_index()

        return results_df

    def _get_accuracy(self, models_root, model_ids, dataset, batch_size):
        return self._compute_metric_per_sample("accuracy", models_root, model_ids, dataset, ResultsBuilderMetric.accuracy_metric, batch_size)

    def _get_average_spikes_per_sample(self, models_root, model_ids, dataset, batch_size):
        return self._compute_metric_per_sample("spike_counts", models_root, model_ids, dataset, ResultsBuilderMetric.spike_count, batch_size, return_all=True)

    def _compute_metric_per_sample(self, metric_name, models_root, model_ids, dataset, metric, batch_size, **kwargs):
        results_df = trainer.build_metric_df(models_root, model_ids, self._model_loader, dataset, metric, batch_size, **kwargs)
        results_df = results_df.groupby("model_id").sum()["metric_score"] / len(dataset)
        results_df = results_df.to_frame().rename(columns={"metric_score": metric_name})

        return results_df

    def _get_average_duration_per_batch(self, models_root, model_ids):
        durations_list = []

        for model_id in model_ids:
            duration = trainer.load_log(models_root, model_id)["duration"][1:].mean()
            durations_list.append({"model_id": model_id, "duration": duration})

        return pd.DataFrame(durations_list).set_index("model_id")

    def _get_model_ids(self, models_root):
        model_paths = glob.glob(models_root + "/*")
        return [model_path.split("/")[-1] for model_path in model_paths]


class DatasetResultsBuilder(BaseDatasetResultsBuilder):

    def __init__(self, models_root, dataset, batch_size=256, build_activity=True):
        super().__init__(models_root, dataset, batch_size, build_activity, block.trainer.Trainer.hyperparams_mapper, block.trainer.Trainer.model_loader)


class ResultsBuilderMetric:

    @staticmethod
    def accuracy_metric(output, target):
        _, predictions = torch.max(output, 1)
        return (predictions == target).sum().cpu().item()

    @staticmethod
    def spike_count(output, target):
        count = 0

        for layer_spikes in output[1]:
            count += layer_spikes.sum().cpu().item()

        return count
