import os

results_path = "/home/luketaylor/PycharmProjects/FastSNN/benchmark_results"
batch_sizes = [16, 32, 64, 128, 256]

# Run layer benchmarks
t_lens = [2 ** i for i in range(3, 12)]
hidden_units = [i * 100 for i in range(1, 11)]

for batch_size in batch_sizes:
    for t_len in t_lens:
        for hidden_unit in hidden_units:
            print(f"Running layer t_len={t_len} hidden_unit={hidden_unit}...")
            os.system(f"python benchmark.py --path={results_path} --type=layer --fast_layer=False --t_len={t_len} --hidden_units={hidden_unit} --batch_size={batch_size}")
            os.system(f"python benchmark.py --path={results_path} --type=layer --fast_layer=True --t_len={t_len} --hidden_units={hidden_unit} --batch_size={batch_size}")


# Run model benchmarks
# t_lens = [2 ** i for i in range(3, 12)]
# hidden_units = [128, 256, 512]
# n_layers = [i * 5 for i in range(1, 11)]
#
# for t_len in t_lens:
#     for hidden_unit in hidden_units:
#         for n_layer in n_layers:
#             print(f"Running model t_len={t_len} hidden_unit={hidden_unit} n_layer={n_layer}...")
#             os.system(f"python benchmark.py --path={results_path} --type=model --fast_layer=False --t_len={t_len} --hidden_units={hidden_unit} --n_layers={n_layer} --batch_size=128")
#             os.system(f"python benchmark.py --path={results_path} --type=model --fast_layer=True --t_len={t_len} --hidden_units={hidden_unit} --n_layers={n_layer} --batch_size=128")
