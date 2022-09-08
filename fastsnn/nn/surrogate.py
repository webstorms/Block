import torch


class FastSigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        
        return grad, None


# TODO: Add Deng SG https://openreview.net/pdf?id=_XNtisL32jv