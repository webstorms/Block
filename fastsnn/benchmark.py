import os
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
