import os
import glob
import time

import torch
import pandas as pd

from fastsnn import datasets, layers, models


class Benchmarker:

    def __init__(self, fast_layer, t_len, input_units, hidden_units, min_r=0, max_r=200, n_samples=11, batch_size=16):
        self._fast_layer = fast_layer
        self._t_len = t_len
        self._input_units = input_units
        self._hidden_units = hidden_units
        self._batch_size = batch_size

        if fast_layer:
            self._model = self._get_fast_model(t_len, input_units, hidden_units)
        else:
            self._model = self._get_vanilla_model(input_units, hidden_units)

        self._data_loader = self._get_data_loader(t_len, input_units, min_r, max_r, batch_size, n_samples*batch_size)
        self._benchmark_results = None

    def benchmark(self, device="cuda"):
        timing_list = []

        self._model = self._model.to(device)

        for i, data in enumerate(self._data_loader):
            # Benchmark forward pass
            data = data.to(device)

            start_time = time.time()
            output = self._model(data)
            torch.cuda.synchronize()
            forward_pass_time = time.time() - start_time

            # Benchmark backward pass
            start_time = time.time()
            fake_target = torch.zeros(output[0].shape, device=output[0].device)
            loss = (output[0] - fake_target).mean()
            loss.backward()
            torch.cuda.synchronize()
            backward_pass_time = time.time() - start_time

            if i > 0:
                timing_row = {"forward_time": forward_pass_time, "backward_time": backward_pass_time}
                timing_list.append(timing_row)

        self._benchmark_results = timing_list

    def to_df(self):
        assert self._benchmark_results is not None

        results = []

        for results_row in self._benchmark_results:
            results.append({**results_row, **self._get_description()})

        return pd.DataFrame(results)

    def save(self, path):
        results_df = self.to_df()
        results_df.to_csv(os.path.join(path, f"{self._get_name()}.csv"), index=False)

    def _get_description(self):
        model_type = "fast" if self._fast_layer else "vanilla"
        return {"type": model_type, "t_len": self._t_len, "units": self._hidden_units, "batch": self._batch_size}

    def _get_data_loader(self, t_len, n_units, min_r, max_r, batch_size, n_samples):
        spikes_dataset = datasets.SyntheticSpikes(t_len, n_units, min_r, max_r, n_samples)
        return torch.utils.data.DataLoader(spikes_dataset, batch_size, shuffle=False)

    def _get_name(self):
        raise NotImplementedError

    def _get_vanilla_model(self, n_units):
        raise NotImplementedError

    def _get_fast_model(self, t_len, n_units):
        raise NotImplementedError


class LinearLayerBenchmarker(Benchmarker):

    def _get_name(self):
        return f"linearlayer_{self._fast_layer}_{self._t_len}_{self._hidden_units}_{self._batch_size}"

    def _get_vanilla_model(self, input_units, hidden_units):
        return layers.LinearLIFNeurons(input_units, hidden_units)

    def _get_fast_model(self, t_len, input_units, hidden_units):
        return layers.LinearFastLIFNeurons(t_len, input_units, hidden_units)


class LinearModelBenchmarker(Benchmarker):

    def __init__(self, fast_layer, t_len, input_units, hidden_units, n_layers, min_r=0, max_r=200, n_samples=11, batch_size=16):
        self._n_layers = n_layers
        super().__init__(fast_layer, t_len, input_units, hidden_units, min_r, max_r, n_samples, batch_size)

    def _get_name(self):
        return f"linearmodel_{self._fast_layer}_{self._t_len}_{self._hidden_units}_{self._n_layers}_{self._batch_size}"

    def _get_description(self):
        return {**super()._get_description(), "n_layers": self._n_layers}

    def _get_vanilla_model(self, input_units, hidden_units):
        return models.LinearModel(None, input_units, 10, hidden_units, self._n_layers, fast_layer=False)

    def _get_fast_model(self, t_len, input_units, hidden_units):
        return models.LinearModel(t_len, input_units, 10, hidden_units, self._n_layers, fast_layer=True)


# Loading benchmark results

class Query:

    def __init__(self, root):
        self._root = root

        self._results_df = self._build_df()

    def get_results(self, **kwargs):
        query = True
        for key, value in kwargs.items():
            query &= self._results_df[key] == value

        if len(kwargs) > 0:
            return self._results_df[query]

        return self._results_df

    def get_speedups(self, **kwargs):
        results_df = self.get_results(**kwargs)

        return self._build_speedups(results_df)

    def _get_paths(self, path):
        raise NotImplementedError

    def _build_speedups(self, results_df):
        pass

    def _build_df(self):
        results_df_list = []

        for path in self._get_paths(self._root):
            results_df_list.append(pd.read_csv(path))

        results_df = pd.concat(results_df_list)
        results_df["total_time"] = results_df["forward_time"] + results_df["backward_time"]

        return results_df


class LayerQuery(Query):

    def _get_paths(self, root):
        return [path for path in glob.glob(f"{root}/*") if "layer" in path]

    def _build_speedups(self, results_df):
        t_lens = list(results_df["t_len"].unique())
        units = list(results_df["units"].unique())
        speedup_list = []

        for t_len in t_lens:
            for unit in units:
                # Query times
                query = (results_df["t_len"] == t_len) & (results_df["units"] == unit)
                vanilla_layer_query = query & (results_df["type"] == "vanilla")
                fast_layer_query = query & (results_df["type"] == "fast")
                vanilla_layer_time = results_df[vanilla_layer_query].mean(numeric_only=True)
                fast_layer_time = results_df[fast_layer_query].mean(numeric_only=True)

                # Compute speedup
                forward_speedup = vanilla_layer_time["forward_time"] / fast_layer_time["forward_time"]
                backward_speedup = vanilla_layer_time["backward_time"] / fast_layer_time["backward_time"]
                pass_speedup = fast_layer_time["forward_time"] / fast_layer_time["backward_time"]
                total_speedup = vanilla_layer_time["total_time"] / fast_layer_time["total_time"]
                speedup_list.append({"t_len": t_len, "units": unit, "forward_speedup": forward_speedup, "backward_speedup": backward_speedup, "total_speedup": total_speedup, "pass_speedup": pass_speedup})

        return pd.DataFrame(speedup_list)





# def _get_model_result_paths(path="/home/luketaylor/PycharmProjects/FastSNN/benchmark_results"):
#     benchmark_results_paths = glob.glob(f"{path}/*")
#
#     return [path for path in benchmark_results_paths if "model" in path]


    # # TODO: Compute all speedups
    # def build_model_speedup_df(path, speedup_type, batch, units):
    #     layer_results_df = build_model_results_df(path)
    #     query = (layer_results_df["batch"] == batch) & (layer_results_df["units"] == units)
    #     layer_results_df = layer_results_df[query]
    #
    #     t_lens = list(layer_results_df["t_len"].unique())
    #     n_layers = list(layer_results_df["n_layers"].unique())
    #
    #     speedup_list = []
    #
    #     for t_len in t_lens:
    #         for n_layer in n_layers:
    #             # Query times
    #             query = (layer_results_df["t_len"] == t_len) & (layer_results_df["n_layers"] == n_layer)
    #             vanilla_layer_query = query & (layer_results_df["type"] == "vanilla")
    #             fast_layer_query = query & (layer_results_df["type"] == "fast")
    #             vanilla_layer_time = layer_results_df[vanilla_layer_query].iloc[1:].mean(numeric_only=True)[speedup_type]
    #             fast_layer_time = layer_results_df[fast_layer_query].iloc[1:].mean(numeric_only=True)[speedup_type]
    #             # Compute speedup
    #             speedup = vanilla_layer_time / fast_layer_time
    #             speedup_list.append({"t_len": t_len, "n_layers": n_layer, "speedup": speedup})
    #
    #     return pd.DataFrame(speedup_list)

