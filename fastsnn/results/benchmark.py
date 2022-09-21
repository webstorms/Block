import glob

import numpy as np
import pandas as pd


class BaseBenchmarkQuery:

    def __init__(self, root, batches=[32, 64, 128]):
        self._root = root

        self._results_df = self._build_df()
        self._results_df = pd.concat([self._query_results(batch=b) for b in batches])

    def _build_df(self):
        results_df_list = []

        for path in self._get_paths(self._root):
            results_df_list.append(pd.read_csv(path))

        results_df = pd.concat(results_df_list)
        results_df["total_time"] = results_df["forward_time"] + results_df["backward_time"]

        return results_df

    def _get_paths(self, root):
        return [path for path in glob.glob(f"{root}/*")]

    def _query_results(self, **kwargs):
        query = True
        for key, value in kwargs.items():
            query &= self._results_df[key] == value

        if len(kwargs) > 0:
            return self._results_df[query]

        return self._results_df


class Benchmark2dQuery(BaseBenchmarkQuery):

    def __init__(self, root, batches=[32, 64, 128]):
        super().__init__(root, batches)

    def get_speedups(self, apply_mean_time=False, **kwargs):
        results_df = self._query_results(**kwargs)

        vanilla_times = results_df[results_df["method"] == "standard"].set_index(["t_len", "units", "batch"])[["forward_time", "backward_time", "total_time"]]
        fast_times = results_df[results_df["method"] == "fast_naive"].set_index(["t_len", "units", "batch"])[["forward_time", "backward_time", "total_time"]]

        if apply_mean_time:
            vanilla_times = vanilla_times.groupby(["t_len", "units", "batch"]).mean()
            fast_times = fast_times.groupby(["t_len", "units", "batch"]).mean()

        speedup_df = vanilla_times / fast_times
        speedup_df.rename(columns={"forward_time": "forward_speedup", "backward_time": "backward_speedup", "total_time": "total_speedup"}, inplace=True)

        return speedup_df

    def get_durations(self, units=200, batch_list=[16, 32, 64, 128, 256], log=True, **kwargs):
        layer_results = []
        for batch in batch_list:
            layer_results.append(self._query_results(batch=batch, units=units, **kwargs))
        results_df = pd.concat(layer_results).reset_index()

        results_df["total_time"] = np.log10(results_df["total_time"]) if log else results_df["total_time"]
        results_df.rename(columns={"method": "Model", "batch": "Batch"}, inplace=True)
        results_df["Model"].replace("fast_naive", "FastSNN", inplace=True)
        results_df["Model"].replace("standard", "Standard", inplace=True)

        return results_df

    def get_forward_vs_backward_speedup(self, units=200, batch_list=[16, 32, 64, 128, 256], **kwargs):
        layer_relative_speedup_df = self.get_durations(units, batch_list, log=False, **kwargs).reset_index()
        relative_speedup = layer_relative_speedup_df["backward_time"] / layer_relative_speedup_df["forward_time"]
        layer_relative_speedup_df["relative_speedup"] = relative_speedup
        layer_relative_speedup_df = layer_relative_speedup_df[layer_relative_speedup_df["Model"] == "FastSNN"]

        return layer_relative_speedup_df

    def get_relative_speedups(self, units=200, target_batch=128):
        durations_df = self.get_durations(units=units, log=False)

        # Reference standard SNN times for a particular batch size
        standard_times_df = durations_df[(durations_df["Model"] == "Standard") & (durations_df["Batch"] == target_batch)]
        mean_standard_times_df = standard_times_df.groupby("t_len").mean()

        # Fast times
        fast_times_df = durations_df[(durations_df["Model"] == "FastSNN")].copy()
        fast_times_df["extended_time"] = fast_times_df["total_time"] * (target_batch / fast_times_df["Batch"])
        fast_times_df = fast_times_df.set_index("t_len")

        # Build relative speedup df
        t_list = []
        b_list = []
        s_list = []
        for t in fast_times_df.index.unique():
            speedup = mean_standard_times_df.loc[t]["total_time"] / fast_times_df.loc[t]["extended_time"]
            t_list.extend(fast_times_df.loc[t].index)
            b_list.extend(fast_times_df.loc[t]["Batch"])
            s_list.extend(speedup.values)

        return pd.DataFrame({"t_len": t_list, "Batch": b_list, "speedup": s_list})


class BenchmarkUnitsQuery(Benchmark2dQuery):

    def __init__(self, root, batches=[32, 64, 128]):
        super().__init__(root, batches)


class BenchmarkLayersQuery(BaseBenchmarkQuery):

    def __init__(self, root, batches=[32, 64, 128]):
        super().__init__(root, batches)

    def get_speedups(self, **kwargs):
        results_df = self._query_results(**kwargs)

        vanilla_times = results_df[results_df["method"] == "standard"].set_index(["t_len", "units",  "layers", "batch"])[["forward_time", "backward_time", "total_time"]]
        fast_times = results_df[results_df["method"] == "fast_naive"].set_index(["t_len", "units", "layers", "batch"])[["forward_time", "backward_time", "total_time"]]
        speedup_df = vanilla_times / fast_times
        speedup_df.rename(columns={"forward_time": "forward_speedup", "backward_time": "backward_speedup", "total_time": "total_speedup"}, inplace=True)

        return speedup_df