#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the Spiking Neural Network (SNN) baseline is defined using the
surrogate gradient method.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import scipy

from sparch.helpers.plot import plot_network
import sparch.helpers.surrogate as surrogate

class SNN(nn.Module):
    """
    A multi-layered Spiking Neural Network (SNN).

    It accepts input tensors formatted as (batch, time, feat). In the case of
    4d inputs like (batch, time, feat, channel) the input is flattened as
    (batch, time, feat*channel).

    The function returns the outputs of the last spiking or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively, as well
    as the firing rates of all hidden neurons with shape (num_layers*feats).
    """

    def __init__(
        self,
        args
    ):
        super().__init__()

        # Set params
        self.reshape = True if len(args.input_shape) > 3 else False
        self.input_size = float(torch.prod(torch.tensor(args.input_shape[2:])))
        self.layer_sizes = args.layer_sizes
        self.bidirectional = args.bidirectional
        self.use_readout_layer = True
        self.single_spike = args.single_spike
        self.track_balance = args.track_balance

        # Check params
        if args.auto_encoder:
            if args.n_layers != 1 or not args.balance:
                raise ValueError("Auto-encoder requires a single balanced RLIF layer.")
            
        if args.balance and args.model != "RLIF":
            raise ValueError("Balance is only available for RLIF model yet.")
        
        # Register dummy parameter. otherwise optimizer will complain about empty parameter list if nothing is learned
        if args.auto_encoder:
            self.register_parameter("dummy", nn.Parameter(torch.Tensor([0.0])))

            

        # Init trainable parameters
        self.snn = self._init_layers(args)

        # Init arrays for tracking network behavior of last forward() call for batch 0 (firing rates, balance, etc)
        self.spikes = []
        self.voltages = []
        self.currents = []

    def _init_layers(self, args):
        snn = nn.ModuleList([])
        input_size = self.input_size

        # Hidden layers
        for i in range(args.n_layers):
            snn.append(
                globals()[args.model + "Layer"](
                    input_size=int(input_size),
                    hidden_size=int(self.layer_sizes[i]),
                    args=args
                )
            )
            input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        # Readout layer
        if self.use_readout_layer:
            decoder = snn[-1].W.data.detach().T if args.auto_encoder else None
            snn.append(
                ReadoutLayer(
                    input_size=int(input_size),
                    hidden_size=int(self.layer_sizes[-1]),
                    args=args,
                    init_weight=decoder
                )
            )

        return snn

    def forward(self, x):
        # Reset tracking lists
        self.inputs = x
        self.spikes = []
        self.voltages = []
        self.currents_exc = []
        self.currents_inh = []

        # Reshape input tensors to (batch, time, feats) for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError

        # Process all layers
        for snn_lay in self.snn:
            x = snn_lay(x)
            if not snn_lay.__class__ == ReadoutLayer:
                self.spikes.append(x)
                self.voltages.append(snn_lay.v)
                if self.track_balance:
                    if not (torch.isfinite(snn_lay.I_exc).all() and torch.isfinite(snn_lay.I_inh).all()):
                        print("Warning: currents are NaN, setting to 0.0")
                        print("Number of NaNs: ", torch.logical_not(torch.isfinite(snn_lay.I_exc)).sum(), torch.logical_not(torch.isfinite(snn_lay.I_inh)).sum())
                    self.currents_exc.append(snn_lay.I_exc)
                    self.currents_inh.append(snn_lay.I_inh)

        self.spikes = torch.stack(self.spikes)
        self.voltages = torch.stack(self.voltages)
        if self.track_balance:
            self.currents_exc = torch.stack(self.currents_exc)
            self.currents_inh = torch.stack(self.currents_inh)

        # Compute mean firing rate of each spiking neuron
        firing_rates = self.spikes.mean(dim=(1, 2))
        
        if self.track_balance:
            # Assume only one layer => [0]
            median_exc = scipy.signal.medfilt(self.currents_exc[0].cpu().numpy(), kernel_size=(1, 5, 1))
            median_inh = scipy.signal.medfilt(self.currents_inh[0].cpu().numpy(), kernel_size=(1, 5, 1))
            balance_arr = np.array([[np.corrcoef(median_exc[b, :,  d], median_inh[b, :, d])[0][1] for d in range(median_exc.shape[2])] for b in range(median_exc.shape[0])])
            balance_arr = np.nan_to_num(balance_arr, nan=0, posinf=0, neginf=0)
            balance = -np.mean(balance_arr)
                
            self.balance_val = balance

        return x, firing_rates
    
    def plot(self, filename, show=False, lowpass=True):
        plot_network(self.inputs, self.spikes, self.layer_sizes, self.track_balance, self.currents_exc, self.currents_inh, self.voltages, show, lowpass=lowpass, filename=filename)


class LIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons without layer-wise
    recurrent connections (LIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        args
    ):
        super().__init__()

        # Fixed parameters
        self.batch_size = args.batch_size * (1 + args.bidirectional)
        self.bidirectional = args.bidirectional
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.spike_fct = surrogate.SpikeFunctionBoxcar.apply if args.single_spike is False else surrogate.SingleSpikeFunctionBoxcar.apply
        self.threshold = 1.0

        # Trainable parameters
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if args.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_size, momentum=0.05)
            self.normalize = True
        elif args.normalization == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._lif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _lif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (LIF)
            ut = alpha * (ut - st) + (1 - alpha) * Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class adLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons without
    layer-wise recurrent connections (adLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        args
    ):
        super().__init__()

        # Fixed parameters
        self.bidirectional = args.bidirectional
        self.batch_size = args.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = surrogate.SpikeFunctionBoxcar.apply if args.single_spike is False else surrogate.SingleSpikeFunctionBoxcar.apply
        self.threshold = 1.0

        # Trainable parameters
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(hidden_size))
        self.beta = nn.Parameter(torch.Tensor(hidden_size))
        self.a = nn.Parameter(torch.Tensor(hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        # Initialize normalization
        self.normalize = False
        if args.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_size, momentum=0.05)
            self.normalize = True
        elif args.normalization == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._adlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _adlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (adLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] - wt)

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class RLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        args
    ):
        super().__init__()

        # Fixed parameters
        self.bidirectional = args.bidirectional
        self.batch_size = args.batch_size * (1 + self.bidirectional)
        ref_param = np.log(1-args.h*args.lambda_v)
        self.alpha_lim = [np.exp(ref_param*10), np.exp(ref_param/10)] if args.balance and not args.balance_setting in ["S1", "S3"] else [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.single_spike = args.single_spike
        self.spike_fct = surrogate.SpikeFunctionBoxcar.apply if args.single_spike is False else surrogate.SingleSpikeFunctionBoxcar.apply
        self.track_balance = args.track_balance
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.repeat = args.repeat
        self.fix_w_in = args.fix_w_in
        self.fix_w_rec = args.fix_w_rec
        self.fix_tau_rec = args.fix_tau_rec
        
        self.lambda_v = args.lambda_v
        self.lambda_d = args.lambda_d
        self.sigma_v  = args.sigma_v
        self.h        = args.h
        self.mu       = args.mu
        self.nu       = args.nu
        
        self.balance = args.balance
        self.setting = args.balance_setting

        # Trainable parameters        
        if self.fix_w_in:
            self.register_buffer("W", torch.Tensor(hidden_size, input_size))
        else:
            self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        if self.fix_w_rec:
            self.register_buffer("V", torch.Tensor(hidden_size, hidden_size))
        else:
            self.V = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            
        if self.fix_tau_rec:
            self.register_buffer("alpha", torch.Tensor(hidden_size))
        else:
            self.alpha = nn.Parameter(torch.Tensor(hidden_size))
        
        if self.balance:
            self.W.data = torch.bernoulli(torch.full((hidden_size, input_size), 0.7)) * torch.empty(hidden_size, input_size).uniform_(-0.1, 0.1)
            self.V.data = self.W @ self.W.T + self.mu * (self.lambda_d**2) * torch.eye(hidden_size)
            self.alpha.data = torch.full((hidden_size,), 1-self.h*self.lambda_v)
            if args.auto_encoder and args.balance_ae_setting == "S1":
                self.alpha.data = nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            elif args.auto_encoder and args.balance_ae_setting == "S2":
                self.alpha.data = nn.init.constant_(self.alpha, args.ae_fac * (1-self.h*self.lambda_v))
            
            self.v_thresh = 0.5*(self.nu * self.lambda_d + self.mu * self.lambda_d**2 + torch.diagonal(self.V.data))
                
            self.V.data = -self.V.data
        else:
            #torch.manual_seed(20)
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            nn.init.orthogonal_(self.V)
            nn.init.uniform_(self.W, -np.sqrt(1/input_size), np.sqrt(1/input_size))
            
            #print("Alpha: ", self.alpha[:5])
            #print("V: ", self.V[:5, :5])
            #print("W: ", self.W[:5, :5])
            
            self.v_thresh = torch.tensor([1.0])
            
        print("Recurrent Layer Initialization Information:")
        print("v_thresh | min: {:.5f}, max: {:.5f}, mean: {:.5f}".format(self.v_thresh.min(), self.v_thresh.max(), self.v_thresh.mean()))
        print("W        | min: {:.5f}, max: {:.5f}, mean: {:.5f}".format(self.W.data.min(), self.W.data.max(), self.W.data.mean()))
        print("V        | min: {:.5f}, max: {:.5f}, mean: {:.5f}".format(self.V.data.min(), self.V.data.max(), self.V.data.mean()))
        print("alpha    | min: {:.5f}, max: {:.5f}, mean: {:.5f}".format(self.alpha.min(), self.alpha.max(), self.alpha.mean()))

        # Initialize normalization
        self.normalize = False
        if args.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_size, momentum=0.05)
            self.normalize = True
        elif args.normalization == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = torch.matmul(x, self.W.t())
        if self.track_balance:
            with torch.no_grad():
                self.I_inh = torch.matmul(x, torch.where(self.W < 0, self.W, torch.zeros_like(self.W)).t())
                self.I_exc = torch.matmul(x, torch.where(self.W >= 0, self.W, torch.zeros_like(self.W)).t())
                
                if not (torch.isfinite(self.I_exc).all() and torch.isfinite(self.I_inh).all()):
                    print("Warning: currents are NaN, setting to 0.0")
                    print("Number of NaNs: ", torch.logical_not(torch.isfinite(self.I_exc)).sum(), torch.logical_not(torch.isfinite(self.I_inh)).sum())
                    print("Number of NaNs in x: ", torch.logical_not(torch.isfinite(x)).sum())
                    print("Number of NaNs in W: ", torch.logical_not(torch.isfinite(self.W)).sum())

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._rlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)
        gc.collect()
        return s

    def _rlif_cell(self, Wx):

        # Initializations
        sim_time = Wx.shape[1]
        device = Wx.device
        self.v = torch.zeros(Wx.shape[0], Wx.shape[1], Wx.shape[2]).to(device)
        #torch.manual_seed(20)
        ut = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device) if self.balance else torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device) if self.balance else torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = torch.zeros(Wx.shape[0], Wx.shape[1], Wx.shape[2]).to(device)

        v_thresh = self.v_thresh.to(device)
        
        if self.fix_tau_rec:
            alpha = self.alpha
            beta = self.h
        else:
            alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
            beta = 1-alpha
            if self.setting in ["S4", "S5"]:
                beta = (1-alpha) / self.lambda_v

        # Set diagonal elements of recurrent matrix to zero
        if self.fix_w_rec:
            V = self.V
        else:
            V = self.V.clone().fill_diagonal_(0)
            
        #print("Alpha:", alpha[:10])
        #print("Beta:", beta[:10])
        #print("V:", V.data[:10, :10])

        # Loop over time axis
        for t in range(sim_time):            
            # Compute and save membrane potential (RLIF)
            ut = alpha * (ut - (st if not self.fix_w_rec else 0)) + beta * (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut.clone(), v_thresh)
            if self.balance and not self.setting in ["S1", "S2", "S5"]:
                st = st / self.h
                            
            #if (t >= 0 and t <= 20) or(t >= sim_time-10):
            #    print("t=", t, ":", np.where(st.data.cpu().numpy() > 0)[1])
            #    print("\tut: ", ut[0, [76, 83]].data.tolist())

            # Track neuron state and activity
            self.v[:, t, :] = ut.detach()
            s[:, t, :] = st
            
            if self.track_balance:
                with torch.no_grad():
                    i_fast_inh, i_fast_exc = self._signed_matmul(st, V) # note: the resulting i_rec_exc/inh is equivalent to torch.matmul(st, V)
                    self.I_inh[:, t, :] += i_fast_inh
                    self.I_exc[:, t, :] += i_fast_exc

        return s

    def _signed_matmul(self, A, B):
        # Compute C:=A x B for matrices A & B, split up into positive and negative components
        # Returns: C_neg (AxB for negative elements of A, rest set to 0), C_pos (AxB for positive elements of A, rest set to 0)
        return torch.mm(A, torch.where(B<0, B, 0)), torch.mm(A, torch.where(B>=0, B, 0))


class RadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RadLIF).

    Arguments
    ---------
    input_size : int
        Number of features in the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        args
    ):
        super().__init__()

        # Fixed parameters
        self.bidirectional = args.bidirectional
        self.batch_size = args.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = surrogate.SpikeFunctionBoxcar.apply if args.single_spike is False else surrogate.SingleSpikeFunctionBoxcar.apply
        self.threshold = 1.0

        # Trainable parameters
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(hidden_size))
        self.beta = nn.Parameter(torch.Tensor(hidden_size))
        self.a = nn.Parameter(torch.Tensor(hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if args.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_size, momentum=0.05)
            self.normalize = True
        elif args.normalization == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._radlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _radlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (RadLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (
                Wx[:, t, :] + torch.matmul(st, V) - wt
            )

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)


class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    hidden_size : int
        Number of output neurons.
    batch_size : int
        Batch size of the input tensors.
    dropout : float
        Dropout factor (must be between 0 and 1).
    normalization : str
        Type of normalization. Every string different from 'batchnorm'
        and 'layernorm' will result in no normalization.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        args,
        init_weight=None
    ):
        super().__init__()

        # Fixed parameters
        ref_param = np.log(1-args.h*args.lambda_d)
        self.alpha_lim = [np.exp(ref_param*10), np.exp(ref_param/10)] if args.balance and not args.balance_setting in ["S1", "S3"] else [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.h = args.h
        self.lambda_d = args.lambda_d
        self.balance = args.balance
        self.t_crop = args.t_crop
        self.fix_w_out = args.fix_w_out
        self.fix_tau_out = args.fix_tau_out
        
        self.setting = args.balance_setting

        # Trainable parameters
        if self.fix_w_out:
            self.register_buffer("W", torch.Tensor(hidden_size, input_size))
        else:
            self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        if init_weight is not None:
            self.W.data = init_weight
        else:
            #torch.manual_seed(20)
            nn.init.uniform_(self.W, -np.sqrt(1/input_size), np.sqrt(1/input_size))

        if self.fix_tau_out:
            self.register_buffer("alpha", torch.Tensor(hidden_size))
        else:
            self.alpha = nn.Parameter(torch.Tensor(hidden_size))

        if self.balance:
            nn.init.constant_(self.alpha, 1-self.h*self.lambda_d)
            if args.auto_encoder and args.balance_ae_setting == "S1":
                self.alpha.data = nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
            elif args.auto_encoder and args.balance_ae_setting == "S2":
                self.alpha.data = nn.init.constant_(self.alpha, args.ae_fac * (1-self.h*self.lambda_d))
        else:
            #torch.manual_seed(20)
            nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        
        print("Readout Layer Initialization Information:")
        print("Alpha | min: {:.5f}, max: {:.5f}, mean: {:.5f}".format(self.alpha.min(), self.alpha.max(), self.alpha.mean()))
        print("W     | min: {:.5f}, max: {:.5f}, mean: {:.5f}".format(self.W.data.min(), self.W.data.max(), self.W.data.mean()))
        

        # Initialize normalization
        self.normalize = False
        if args.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_size, momentum=0.05)
            self.normalize = True
        elif args.normalization == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = torch.matmul(x, self.W.t())

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        #torch.manual_seed(20)
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        if self.fix_tau_out:
            alpha = self.alpha
            beta = self.h
        else:
            alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
            beta = 1-alpha
            if self.setting in ["S4", "S5"]:
                beta = (1-alpha) / self.lambda_d

        #print("Alpha:", alpha)
        #print("Beta:", beta)
        #print("W", self.W[:,:10])
        # Loop over time axis
        for t in range(Wx.shape[1]):
            # Compute potential (LI)
            ut = alpha * ut + beta * Wx[:, t, :]
            #if t<50 or t > Wx.shape[1] - 10:
            #    print("Wx[t]", Wx[0, t])
            #    print("t=", t, ":", ut[0].data.cpu().tolist())
            if t >= self.t_crop:
                out = out + F.softmax(ut, dim=1)

        return out
