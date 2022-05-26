import torch


def tensor_to_coordinates(tensor):
    n, t = tensor.shape[0], tensor.shape[1]
    ts, ns = [], []

    for n_id in range(n):
        for t_id in range(t):
            if tensor[n_id, t_id] == 1:
                ts.append(t_id)
                ns.append(n_id)

    return ts, ns


def mem_tensor_to_lists(tensor):
    return [tensor[i].numpy() for i in range(tensor.shape[0])]


def get_get_output(dataset, model, sample_id):
    x, y = dataset[sample_id]
    with torch.no_grad():
        model_output = model(x.unsqueeze(0).cuda())
    prediction, spike_history, mem_history = model_output
    hidden_spikes = spike_history[0]
    mem_preds = mem_history[1]

    return x, hidden_spikes.cpu()[0], mem_preds.cpu()[0]
