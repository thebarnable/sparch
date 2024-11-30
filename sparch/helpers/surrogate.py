# -----------------------------------------------------------------------------
# File Name : main.py
# Purpose:
#
# Author: Tim Stadtmann
#
# Creation Date : 28-08-2024
#
# Copyright : (c) Tim Stadtmann
# License : BSD-3-Clause
#
# Based on snns.py in original Sparch package.
# -----------------------------------------------------------------------------

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
        
        x_copy = x.clone()
        x[:, :] = 0
        x[torch.arange(x.shape[0]), torch.argmax(x_copy, dim=1)] = 1
        x[x_copy <= v_thresh] = 0

        return x

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= ctx.v_thresh-0.5] = 0
        grad_x[x > ctx.v_thresh+0.5] = 0
        return grad_x, None, None
