import os
import time

import torch
import torch.nn.functional as F
import pandas as pd
from brainbox import trainer


class Trainer(trainer.Trainer):
    
    def __init__(self, root, model, dataset, n_epochs, batch_size, lr, optimizer_func=torch.optim.Adam, device='cuda', dtype=torch.float, grad_clip_type=None, grad_clip_value=None, save_type='SAVE_DICT'):
        super().__init__(root, model, dataset, n_epochs, batch_size, lr, optimizer_func, device, dtype, grad_clip_type, grad_clip_value, save_type)
        self.times = {'forward_pass': [], 'backward_pass': []}

    @property
    def times_path(self):
        return os.path.join(self.root, self.id, 'times.csv')

    def save_model_log(self):
        super().save_model_log()
        times_df = pd.DataFrame(self.times)
        times_df.to_csv(self.times_path, index=False)

    def loss(self, output, target, model):
        target = target.long()
        loss = F.cross_entropy(output, target)

        return loss

    def train_for_single_epoch(self):
        epoch_loss = 0
        n_samples = 0

        for batch_id, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device).type(self.dtype)
            target = target.to(self.device).type(self.dtype)

            # Forward pass
            start_time = time.time()
            output = self.model(data)[0]
            torch.cuda.synchronize()
            forward_pass_time = time.time() - start_time
            self.times['forward_pass'].append(forward_pass_time)

            loss = self.loss(output, target, self.model)

            # Backward pass
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize()
            backward_pass_time = time.time() - start_time
            self.times['backward_pass'].append(backward_pass_time)

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                epoch_loss += (loss.item() * data.shape[0])
                n_samples += data.shape[0]

        self.save_model()

        return epoch_loss / n_samples
