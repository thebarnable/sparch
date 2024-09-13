import torch

class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x

class SingleSpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020), but allowing only spike to happen in forward().
    """
    @staticmethod
    def forward(ctx, x, v_thresh=0):
        ctx.save_for_backward(x)
        ctx.v_thresh=v_thresh

        x_copy = x.clone()
        x[:, :] = 0
        spike_ids = torch.nonzero(x_copy > v_thresh, as_tuple=False)
        if len(spike_ids) > 0:
            spike_id = spike_ids[torch.randint(len(spike_ids), (1,))][0] # choose random neuron to spike
            x[spike_id[0]][spike_id[1]] = 1
        return x

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= ctx.v_thresh-0.5] = 0
        grad_x[x > ctx.v_thresh+0.5] = 0
        return grad_x, None, None
