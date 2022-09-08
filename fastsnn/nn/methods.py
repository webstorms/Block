import torch
import torch.nn as nn
import torch.nn.functional as F
from brainbox.models import BBModel


from fastsnn.nn.functional import bconv1d


RETURN_SPIKES = 0
RETURN_SPIKES_AND_MEM = 1
RETURN_ALL = 2


class BaseMethod(BBModel):

    def __init__(self, t_len, spike_func, scale):
        super().__init__()
        self._t_len = t_len
        self._spike_func = spike_func
        self._scale = scale


class MethodStandard(BaseMethod):

    def __init__(self, t_len, spike_func, scale, single_spike=False, integrator=False):
        print(f"single_spike {single_spike} integrator {integrator}")
        super().__init__(t_len, spike_func, scale)
        self._single_spike = single_spike
        self._integrator = integrator

    @property
    def hyperparams(self):
        return {**super().hyperparams, "single_spike": self._single_spike}

    def forward(self, current, beta, v_init=None, return_type=RETURN_SPIKES):
        # Initialise membrane voltage and tracking variables
        spike_mask = None
        mem = None if v_init is None else v_init
        mem_list = []
        spikes_list = []

        for t in range(self._t_len):
            # Update membrane potential
            if mem is None:
                new_mem = current[:, :, t]
            else:
                new_mem = torch.einsum("bn...,n->bn...", mem, beta) + current[:, :, t]

            # To spike or not to spike
            spikes = self._spike_func(new_mem - 1, self._scale)

            if self._single_spike:
                if spike_mask is None:
                    spike_mask = torch.zeros(spikes.shape, device=spikes.device)
                spikes *= (1 - spike_mask)
                if not self._integrator:
                    new_mem *= (1 - spike_mask)
                spike_mask = torch.maximum(spike_mask, spikes)

            spikes_list.append(spikes)

            # Reset membrane potential for spiked neurons
            mem_list.append(new_mem.clone())
            #if not self._single_spike:
            new_mem -= spikes
            mem = new_mem

        if return_type == RETURN_SPIKES:
            return torch.stack(spikes_list, dim=2)
        elif return_type == RETURN_SPIKES_AND_MEM:
            return torch.stack(spikes_list, dim=2), torch.stack(mem_list, dim=2)
        elif return_type == RETURN_ALL:
            return torch.stack(spikes_list, dim=2), torch.stack(mem_list, dim=2), current


class MethodFastNaive(BaseMethod):

    def __init__(self, t_len, spike_func, scale, beta):
        super().__init__(t_len, spike_func, scale)
        self._beta = beta
        self._beta_requires_grad = beta.requires_grad

        n_in = len(beta)
        self._beta_ident_base = nn.Parameter(torch.ones(n_in, t_len), requires_grad=False)
        self._beta_exp = nn.Parameter(torch.arange(t_len).flip(0).unsqueeze(0).expand(n_in, t_len).float(), requires_grad=False)
        self._beta_kernel = nn.Parameter(self._build_beta_kernel(beta), requires_grad=False)

        self._phi_kernel = nn.Parameter((torch.arange(t_len) + 1).flip(0).float().view(1, 1, 1, t_len), requires_grad=False)

    @staticmethod
    def g(faulty_spikes):
        # faulty_spikes[faulty_spikes > 1] = 0

        negate_faulty_spikes = faulty_spikes.clone().detach()
        negate_faulty_spikes[faulty_spikes == 1.0] = 0
        faulty_spikes -= negate_faulty_spikes

        # faulty_spikes[faulty_spikes != 1.0] = 0

        return faulty_spikes

    def forward(self, current, beta, v_init=None, return_type=RETURN_SPIKES):
        if v_init is not None:
            current[:, :, 0] += beta.multiply(v_init)

        pad_current = F.pad(current, pad=(self._t_len - 1, 0)).unsqueeze(1)

        # compute membrane potential without reset
        beta_kernel = self._build_beta_kernel(beta) if self._beta_requires_grad else self._beta_kernel
        conv_func = F.conv2d if len(self._beta) == 1 else bconv1d
        membrane = conv_func(pad_current, beta_kernel)

        # map no-reset membrane potentials to output spikes
        faulty_spikes = self._spike_func(membrane - 1, self._scale)
        pad_spikes = F.pad(faulty_spikes, pad=(self._t_len - 1, 0))
        z = F.conv2d(pad_spikes, self._phi_kernel)
        if return_type == RETURN_ALL:
            z_copy = z.clone()
        spikes = MethodFastNaive.g(z).squeeze(1)

        if return_type == RETURN_SPIKES:
            return spikes
        elif return_type == RETURN_SPIKES_AND_MEM:
            return spikes, membrane.squeeze(1)
        else:
            return spikes, z_copy, faulty_spikes, membrane.squeeze(1), current

    def _build_beta_kernel(self, beta):
        beta_base = beta.unsqueeze(1).multiply(self._beta_ident_base)
        return torch.pow(beta_base, self._beta_exp).unsqueeze(1).unsqueeze(1)
