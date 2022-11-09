import os
import time
import logging

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from brainbox import trainer

from block import models, results


class Trainer(trainer.Trainer):

    def __init__(self, root, model, dataset, n_epochs, batch_size, lr, milestones=[-1], gamma=0.1, val_dataset=None, device="cuda", track_activity=False):
        super().__init__(root, model, dataset, n_epochs, batch_size, lr, torch.optim.Adam, device=device, loader_kwargs={"shuffle": True, "pin_memory": True,  "num_workers": 16})
        self._milestones = milestones
        self._gamma = gamma
        self._val_dataset = val_dataset
        self._track_activity = track_activity

        self._times = {"forward_pass": [], "backward_pass": []}
        if track_activity:
            self._activity = []
        self._min_loss = np.inf
        self._milestone_idx = 0
    
    @staticmethod
    def accuracy_metric(output, target):
        _, predictions = torch.max(output, 1)
        return (predictions == target).sum().cpu().item()

    @property
    def times_path(self):
        return os.path.join(self.root, self.id, "times.csv")

    @property
    def activity_path(self):
        return os.path.join(self.root, self.id, "activity.csv")

    def save_model_log(self):
        super().save_model_log()

        # Save times
        times_df = pd.DataFrame(self._times)
        times_df.to_csv(self.times_path, index=False)

        # Save activity
        if self._track_activity:
            activity_df = pd.DataFrame(self._activity)
            activity_df.to_csv(self.activity_path, index=False)

    def loss(self, output, target, model):
        target = target.long()
        loss = F.cross_entropy(output, target)

        return loss

    def train_for_single_epoch(self):
        epoch_loss = 0
        n_samples = 0
        n_correct = 0

        for batch_id, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device).type(self.dtype)
            target = target.to(self.device).type(self.dtype)
            torch.cuda.synchronize()

            # Forward pass
            start_time = time.time()
            if self._track_activity:
                output = self.model(data, return_all=True)
                activity = results.datasets.ResultsBuilderMetric.spike_count(output, None)
                self._activity.append(activity / data.shape[0])
                output = output[0]
            else:
                output = self.model(data)
            torch.cuda.synchronize()
            forward_pass_time = time.time() - start_time
            self._times["forward_pass"].append(forward_pass_time)

            # Compute accuracy
            _, predictions = torch.max(output, 1)
            n_correct += (predictions == target).sum().cpu().item()

            # Compute loss
            loss = self.loss(output, target, self.model)

            # Backward pass
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize()
            backward_pass_time = time.time() - start_time
            self._times["backward_pass"].append(backward_pass_time)

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                epoch_loss += (loss.item() * data.shape[0])
                n_samples += data.shape[0]

        logging.info(f"Train acc: {n_correct/n_samples}")

        if self._val_dataset is not None and len(self.log["train_loss"]) % 10 == 0:
            scores = trainer.compute_metric(self.model, self._val_dataset, Trainer.accuracy_metric, batch_size=self.batch_size)
            logging.info(f"Val acc: {np.sum(scores)/len(self._val_dataset)}")

        return epoch_loss / n_samples

    def on_epoch_complete(self, save):
        if save:
            self.save_model_log()

            epoch_loss = self.log["train_loss"][-1]
            if epoch_loss < self._min_loss:
                logging.info(f"Saving model...")
                self._min_loss = epoch_loss
                self.save_model()

        n_epoch = len(self.log["train_loss"])

        if n_epoch == self._milestones[self._milestone_idx]:
            logging.info(f"Decaying lr...")
            self.lr *= self._gamma
            # Load best model
            self.model = Trainer.load_model(self.root, self.id, self.device, self.dtype)
            self.optimizer = self.optimizer_func(
                self.model.parameters(), self.lr, **self.optimizer_kwargs
            )

            if self._milestone_idx != len(self._milestones) - 1:
                logging.info(f"New milestone target...")
                self._milestone_idx += 1

    def on_training_complete(self, save):
        pass

    @staticmethod
    def load_model(root, id, device, dtype):
        return trainer.load_model(root, id, Trainer.model_loader, device, dtype)

    @staticmethod
    def model_loader(hyperparams):
        model_params = hyperparams["model"]

        name = model_params["name"]
        method = model_params["method"]
        t_len = int(model_params["t_len"])
        heterogeneous_beta = bool(model_params["heterogeneous_beta"])
        beta_requires_grad = bool(model_params["beta_requires_grad"])
        readout_max = bool(model_params["readout_max"])
        single_spike = bool(model_params.get("single_spike", True))

        if name == "YingYangModel":
            return models.YingYangModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        elif name == "LinearMNINSTModel":
            return models.LinearMNINSTModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        elif name == "ConvMNINSTModel":
            return models.ConvMNINSTModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        elif name == "LinearFMNINSTModel":
            return models.LinearFMNINSTModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        elif name == "ConvFMNINSTModel":
            return models.ConvFMNINSTModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        elif name == "NMNISTModel":
            return models.NMNISTModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        elif name == "SHDModel":
            return models.SHDModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        elif name == "CIFAR10Model":
            return models.CIFAR10Model(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)

    @staticmethod
    def hyperparams_mapper(hyperparams):
        method = hyperparams["model"]["method"]
        heterogeneous_beta = hyperparams["model"]["heterogeneous_beta"]
        beta_requires_grad = hyperparams["model"]["beta_requires_grad"]
        readout_max = hyperparams["model"]["readout_max"]
        single_spike = hyperparams["model"]["single_spike"]

        return {"method": method, "heterogeneous_beta": heterogeneous_beta, "beta_requires_grad": beta_requires_grad, "readout_max": readout_max, "single_spike": single_spike}
