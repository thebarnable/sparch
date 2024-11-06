import torch

class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """
    @staticmethod
    def forward(ctx, x, v_thresh=0):
        ctx.save_for_backward(x)
        ctx.v_thresh=v_thresh

        return x.gt(v_thresh).float()

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= ctx.v_thresh-0.5] = 0
        grad_x[x > ctx.v_thresh+0.5] = 0
        return grad_x, None

class SingleSpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020), but allowing only spike to happen in forward().
    """
    @staticmethod
    def forward(ctx, x, v_thresh=0):
        ctx.save_for_backward(x)
        ctx.v_thresh=v_thresh

        """x_copy = x.clone()
        x[:, :] = 0
        spike_ids = torch.nonzero(x_copy > v_thresh, as_tuple=False)
        if len(spike_ids) > 0:
            #spike_id = spike_ids[torch.randint(len(spike_ids), (1,))][0] # choose random neuron to spike
            # Get unique values in the first column and their inverse indices
            unique_x, inverse_indices = torch.unique(spike_ids[:, 0], return_inverse=True)

            # Randomly shuffle each duplicate set and pick the first occurrence
            selected_indices = []
            for i in range(len(unique_x)):
                # Get all indices of current unique x value
                indices = (inverse_indices == i).nonzero(as_tuple=True)[0]
                # Choose one random index from these
                random_index = indices[torch.randint(len(indices), (1,))]
                selected_indices.append(random_index)

            # Construct the result tensor
            indices = spike_ids[torch.tensor(selected_indices).squeeze()]
            if len(indices.shape) == 1:
                indices = indices.unsqueeze(dim=0)
            x[indices[:, 0],indices[:, 1]] = 1"""
            
        x_copy = x.clone() - v_thresh.clone()
        x[:, :] = 0
        x[torch.arange(x.shape[0]), torch.argmax(x_copy, dim=1)] = 1
        x[x_copy<=0] = 0
            
        return x

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= ctx.v_thresh-0.5] = 0
        grad_x[x > ctx.v_thresh+0.5] = 0
        return grad_x, None, None
