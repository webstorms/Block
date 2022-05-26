import os
import glob

import torch
import pandas as pd
from brainbox import trainer

from fastsnn import datasets, models


class DatasetResultsQuery:

    def __init__(self, root, fmnist_path, nmnist_path, shd_path):
        self._root = root
        self._fmnist_path = fmnist_path
        self._nmnist_path = nmnist_path
        self._shd_path = shd_path

        self._fmnist_results_df = None
        self._nmnist_results_df = None
        self._shd_results_df = None

    def build(self, batch_size=256):
        self._fmnist_results_df = DatasetResultsBuilder.get_fmnist_results_df(self._root, self._fmnist_path, batch_size)
        self._nmnist_results_df = DatasetResultsBuilder.get_nmnist_results_df(self._root, self._nmnist_path, batch_size)
        self._shd_results_df = DatasetResultsBuilder.get_shd_results_df(self._root, self._shd_path, batch_size)

    def get_results_df(self):
        results_df = pd.concat([self._fmnist_results_df, self._nmnist_results_df, self._shd_results_df])
        results_df["accuracy"] *= 100

        return results_df

    def get_mean_results_df(self):
        results_df = self.get_results_df()

        return results_df.groupby(["fast_layer", "dataset"]).mean().reset_index()

    def get_comparative_results_df(self):
        fmnist_comparative_statistics_df = DatasetResultsQuery._build_comparative_statistics_df(self._fmnist_results_df)
        fmnist_comparative_statistics_df["dataset"] = "F-MNIST"
        nmnist_comparative_statistics_df = DatasetResultsQuery._build_comparative_statistics_df(self._nmnist_results_df)
        nmnist_comparative_statistics_df["dataset"] = "N-MNIST"
        shd_comparative_statistics_df = DatasetResultsQuery._build_comparative_statistics_df(self._shd_results_df)
        shd_comparative_statistics_df["dataset"] = "SHD"

        return pd.concat([fmnist_comparative_statistics_df, nmnist_comparative_statistics_df, shd_comparative_statistics_df])

    @staticmethod
    def _build_comparative_statistics_df(results_df):

        def get_mean_standard_snn_results_df():
            mean_results_df = results_df.groupby(["fast_layer"]).mean()
            mean_results_df = mean_results_df.reset_index()

            mean_standard_snn_results_df = mean_results_df[mean_results_df["fast_layer"] == False][["n_layers", "accuracy", "spike_counts", "duration"]]
            mean_standard_snn_results_df = mean_standard_snn_results_df.set_index("n_layers")

            return mean_standard_snn_results_df

        def get_fastsnn_results_df():
            fastsnn_results_df = results_df[results_df["fast_layer"]][["n_layers", "accuracy", "spike_counts", "duration"]]
            fastsnn_results_df = fastsnn_results_df.set_index("n_layers")

            return fastsnn_results_df

        mean_standard_snn_results_df = get_mean_standard_snn_results_df()
        fastsnn_results_df = get_fastsnn_results_df()

        relative_change_df = mean_standard_snn_results_df / fastsnn_results_df
        inv_relative_change_df = fastsnn_results_df / mean_standard_snn_results_df
        diff_change_df = 100 * (mean_standard_snn_results_df - fastsnn_results_df)

        summary_df = pd.DataFrame({"acc_diff": diff_change_df["accuracy"], "speedup": relative_change_df["duration"], "spike_reduction": 100 * (1 - inv_relative_change_df["spike_counts"])})
        summary_df = summary_df.reset_index()

        return summary_df


class DatasetResultsBuilder:

    @staticmethod
    def get_fmnist_results_df(root, dataset_root, batch_size=256):
        dataset = datasets.FMNISTDataset(dataset_root, train=False)
        return DatasetResultsBuilder._get_dataset_results_df(root, "fmnist", dataset, batch_size)

    @staticmethod
    def get_nmnist_results_df(root, dataset_root, batch_size=256):
        dataset = datasets.NMNISTDataset(dataset_root, train=False, dt=1)
        return DatasetResultsBuilder._get_dataset_results_df(root, "nmnist", dataset, batch_size)

    @staticmethod
    def get_shd_results_df(root, dataset_root, batch_size=256):
        dataset = datasets.SHDDataset(dataset_root, train=False, dt=2)
        return DatasetResultsBuilder._get_dataset_results_df(root, "shd", dataset, batch_size)

    @staticmethod
    def _get_dataset_results_df(root, dataset_name, dataset, batch_size):
        root = os.path.join(root, dataset_name)
        model_ids = DatasetResultsBuilder._get_model_ids(root)

        accuracy_df = DatasetResultsBuilder._get_accuracy(root, model_ids, dataset, batch_size)
        spike_counts_df = DatasetResultsBuilder._get_average_spikes_per_sample(root, model_ids, dataset, batch_size)
        average_duration_per_bath_df = DatasetResultsBuilder._get_average_duration_per_batch(root, model_ids)
        hyperparameters_df = trainer.build_models_df(root, model_ids, ResultsBuilderMapper.hyperparams_mapper)
        results_df = accuracy_df.join(spike_counts_df).join(average_duration_per_bath_df).join(hyperparameters_df)
        results_df = results_df.reset_index()
        results_df["dataset"] = dataset_name

        return results_df

    @staticmethod
    def _get_accuracy(root, model_ids, dataset, batch_size):
        return DatasetResultsBuilder._compute_metric_per_sample(root, ResultsBuilderMetric.accuracy_metric, "accuracy", model_ids, dataset, batch_size)

    @staticmethod
    def _get_average_spikes_per_sample(root, model_ids, dataset, batch_size):
        return DatasetResultsBuilder._compute_metric_per_sample(root, ResultsBuilderMetric.spike_count, "spike_counts", model_ids, dataset, batch_size)

    @staticmethod
    def _compute_metric_per_sample(root, metric, metric_name, model_ids, dataset, batch_size):
        results_df = trainer.build_metric_df(root, model_ids, ResultsBuilderMapper.model_loader, dataset, metric, batch_size)
        results_df = results_df.groupby("model_id").sum()["metric_score"] / len(dataset)
        results_df = results_df.to_frame().rename(columns={"metric_score": metric_name})

        return results_df

    @staticmethod
    def _get_average_duration_per_batch(root, model_ids):
        durations_list = []

        for model_id in model_ids:
            duration = trainer.load_log(root, model_id)["duration"][1:].mean()
            durations_list.append({"model_id": model_id, "duration": duration})

        return pd.DataFrame(durations_list).set_index("model_id")

    @staticmethod
    def _get_model_ids(root):
        model_paths = glob.glob(root + "/*")
        return [model_path.split("/")[-1] for model_path in model_paths]


class ResultsBuilderMetric:

    @staticmethod
    def accuracy_metric(output, target):
        _, predictions = torch.max(output[0], 1)
        return (predictions == target).sum().cpu().item()

    @staticmethod
    def spike_count(output, target):
        count = 0

        for layer_spikes in output[1][:-1]:
            count += layer_spikes.sum().cpu().item()

        return count


class ResultsBuilderMapper:

    @staticmethod
    def model_loader(hyperparams):
        model_params = hyperparams["model"]

        t_len = model_params["t_len"]
        n_in = model_params["n_in"]
        n_out = model_params["n_out"]
        n_hidden = model_params["n_hidden"]
        n_layers = model_params["n_layers"]
        fast_layer = model_params["fast_layer"]
        skip_connections = model_params["skip_connections"]
        bias = model_params["bias"]
        hidden_beta = model_params["hidden_beta"]
        readout_beta = model_params["readout_beta"]

        return models.LinearModel(t_len, n_in, n_out, n_hidden, n_layers, fast_layer, skip_connections, bias, hidden_beta, readout_beta)

    @staticmethod
    def hyperparams_mapper(hyperparams):
        n_layers = hyperparams["model"]["n_layers"]
        fast_layer = hyperparams["model"]["fast_layer"]
        skip_connections = hyperparams["model"]["skip_connections"]

        return {"n_layers": n_layers, "fast_layer": fast_layer, "skip_connections": skip_connections}
