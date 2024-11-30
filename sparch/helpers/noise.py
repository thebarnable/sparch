# -----------------------------------------------------------------------------
# File Name : noise.py
# Purpose:
#
# Author: Tim Stadtmann
#
# Creation Date : 30-11-2024
#
# Copyright : (c) Tim Stadtmann
# License : BSD-3-Clause
# -----------------------------------------------------------------------------


import torch
try:
    import torch.ao.quantization as quantization
except ImportError:
    import torch.quantization as quantization

def quant(x, quantize):
    if quantize == "":
        return x

    n_bits = int(quantize.split(".")[0])
    n_frac = int(quantize.split(".")[1])

    obs = quantization.observer.MinMaxObserver(quant_min=0, quant_max=2**n_bits-1) #Added quant min and quant max so that it doesn't use 8 bit by default
    obs.to(device=x.device)
    _ = obs(x)
    scale, zero_point = obs.calculate_qparams()
    return (
            torch.clamp(torch.round(x / scale + zero_point), 0, 2**n_bits-1) - zero_point
        ) * scale

def gauss(x, sigma, multiplicative):
    if sigma == 0.0:
        return x

    if multiplicative:
        return x*torch.randn_like(x)*sigma
    else:
        return x+torch.randn_like(x)*sigma