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
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import gc

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
    def forward(ctx, x):
        ctx.save_for_backward(x)

        x_copy = x.clone()
        x[:, :] = 0
        x[torch.arange(x.shape[0]), torch.argmax(x_copy, dim=1)] = 1
        x[x_copy<=0] = 0
        return x

    @staticmethod
    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x


class SNN(nn.Module):
    """
    A multi-layered Spiking Neural Network (SNN).

    It accepts input tensors formatted as (batch, time, feat). In the case of
    4d inputs like (batch, time, feat, channel) the input is flattened as
    (batch, time, feat*channel).

    The function returns the outputs of the last spiking or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively, as well
    as the firing rates of all hidden neurons with shape (num_layers*feats).

    Arguments
    ---------
    input_shape : tuple
        Shape of an input example.
    layer_sizes : int list
        List of number of neurons in all hidden layers
    model_type : str
        Type of neuron model, either 'LIF', 'adLIF', 'RLIF' or 'RadLIF'.
    dropout : float
        Dropout rate (must be between 0 and 1).
    normalization : str
        Type of normalization (batchnorm, layernorm). Every string different
        from batchnorm and layernorm will result in no normalization.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both directions
        is used, which doubles the size of feedforward matrices in layers l>0.
    use_readout_layer : bool
        If True, the final layer is a non-spiking, non-recurrent LIF and outputs
        a cumulative sum of the membrane potential over time. The outputs have
        shape (batch, labels) with no time dimension. If False, the final layer
        is the same as the hidden layers and outputs spike trains with shape
        (batch, time, labels).
    """

    def __init__(
        self,
        args
    ):
        super().__init__()

        self.reshape = True if len(args.input_shape) > 3 else False
        self.input_size = float(torch.prod(torch.tensor(args.input_shape[2:])))
        self.layer_sizes = args.layer_sizes
        self.bidirectional = args.bidirectional
        self.use_readout_layer = True
        self.is_snn = True
        self.balance = args.balance
        self.substeps = args.substeps

        if args.model_type not in ["LIF", "adLIF", "RLIF", "RadLIF"]:
            raise ValueError(f"Invalid neuron type {args.model_type}")

        # Init trainable parameters
        self.snn = self._init_layers(args)

        # Init arrays for tracking network behavior of last forward() call for batch 0 (firing rates, balance, etc)
        self.spikes = []
        self.voltages = []
        self.currents = []

    def _init_layers(self, args):

        snn = nn.ModuleList([])
        input_size = self.input_size
        snn_class = args.model_type + "Layer"

        if self.use_readout_layer:
            num_hidden_layers = len(args.layer_sizes) - 1
        else:
            num_hidden_layers = len(args.layer_sizes)

        # Hidden layers
        for i in range(num_hidden_layers):
            snn.append(
                globals()[snn_class](
                    input_size=int(input_size),
                    hidden_size=int(self.layer_sizes[i]),
                    args=args
                )
            )
            input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        # Readout layer
        if self.use_readout_layer:
            snn.append(
                ReadoutLayer(
                    input_size=int(input_size),
                    hidden_size=int(self.layer_sizes[-1]),
                    args=args
                )
            )

        return snn

    def forward(self, x):
        # Reset tracking lists
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
        for i, snn_lay in enumerate(self.snn):
            if snn_lay.__class__ == RLIFLayer:
                x = snn_lay(x, i==0) # TODO: i==0 super hacky, only works for RLIF currently
            else:
                x = snn_lay(x)
            if not snn_lay.__class__ == ReadoutLayer:
                self.spikes.append(x)
                if snn_lay.balance:
                    self.currents_exc.append(snn_lay.I_exc)
                    self.currents_inh.append(snn_lay.I_inh)
                    self.voltages.append(snn_lay.v)

        # Compute mean firing rate of each spiking neuron
        firing_rates = torch.cat(self.spikes, dim=2).mean(dim=(0, 1))

        return x, firing_rates
    
    def plot(self, filename):
        # define colors
        RED = "#D17171"
        YELLOW = "#F3A451"
        GREEN = "#7B9965"
        BLUE = "#5E7DAF"
        DARKBLUE = "#3C5E8A"
        DARKRED = "#A84646"
        VIOLET = "#886A9B"
        GREY = "#636363"

        # constants for plotting
        LAYER = 0
        BATCH = 0
        NEURON = 0
        N_NEURONS_TO_PLOT=6

        # cast data lists to torch tensors
        spikes = torch.stack(self.spikes)[LAYER, BATCH, :, :]      # layers x batch x time x neurons

        # setup spike raster plot
        t = list(range(0,spikes.shape[0]))
        SCATTER_MIN=0
        SCATTER_MAX=self.layer_sizes[LAYER]
        SCATTER_N_NEURONS=SCATTER_MAX-SCATTER_MIN
        spike_list = torch.argwhere(spikes[:,SCATTER_MIN:SCATTER_MAX]>0)
        
        x_axis = spike_list[:,0].cpu().numpy()
        y_axis = spike_list[:,1].cpu().numpy()
        colors = len(spike_list[:,0])*[BLUE]

        # create plots
        plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
        plt.rc('ytick', labelsize=8) #fontsize of the y tick labels
        fig, axs = plt.subplots(1+N_NEURONS_TO_PLOT*2, 1, sharex=True, gridspec_kw={'height_ratios': 2*N_NEURONS_TO_PLOT*[1] + [5]})
        fig.subplots_adjust(hspace=0)

        # plot
        if self.balance:
            for i in range(0, 2*N_NEURONS_TO_PLOT, 2):
                neuron = i*7
                currents_exc = torch.stack(self.currents_exc)[LAYER, BATCH, :, neuron].cpu()
                currents_inh = torch.stack(self.currents_inh)[LAYER, BATCH, :, neuron].cpu()
                v = torch.stack(self.voltages)[LAYER, BATCH, :, neuron].cpu()
                #b, a = butter(4, 0.005/(0.5*spikes.shape[0]), btype='low', analog=False)
                #currents_exc_lp = np.array(filtfilt(b, a, currents_exc))
                #currents_inh_lp = np.array(filtfilt(b, a, currents_inh))
                axs[i].plot(t, currents_exc, color=BLUE, label="i_exc")
                axs[i].plot(t, -currents_inh, color=RED, label="-i_inh")
                axs[i+1].plot(t, v, color=GREY, label="v")
                axs[i].set_title("neuron " + str(neuron), y=0.5)
                if i==0:
                    axs[i].legend()

        axs[2*N_NEURONS_TO_PLOT].scatter(x_axis, y_axis, c=colors, marker = "o", s=5)
        axs[2*N_NEURONS_TO_PLOT].set_yticks(list(range(0,SCATTER_N_NEURONS,2))[::int(0.5*SCATTER_N_NEURONS/8)])
        scatter_yticklabels = list(range(SCATTER_MIN, SCATTER_MAX,2))
        axs[2*N_NEURONS_TO_PLOT].set_yticklabels(scatter_yticklabels[::int(0.5*SCATTER_N_NEURONS/8)], fontsize=8)

        plt.xlabel('Timesteps')
        #plt.show()
        plt.savefig(filename, dpi=250)
        plt.close()


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
        self.spike_fct = SpikeFunctionBoxcar.apply
        self.substeps = args.substeps
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
        self.drop = nn.Dropout(p=args.pdrop)

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
        self.spike_fct = SpikeFunctionBoxcar.apply
        self.substeps = args.substeps
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
        self.drop = nn.Dropout(p=args.pdrop)

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
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.balance = args.balance
        self.spike_fct = SpikeFunctionBoxcar.apply if args.balance is False else SingleSpikeFunctionBoxcar.apply
        self.substeps = args.substeps
        self.threshold = 1.0
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Trainable parameters
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # if self.track_balance:
        #     self.register_buffer('W_exc', torch.zeros_like(self.W.weight))
        #     self.register_buffer('W_inh', torch.zeros_like(self.W.weight))

        # Initialize normalization
        self.normalize = False
        if args.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_size, momentum=0.05)
            self.normalize = True
        elif args.normalization == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=args.pdrop)

    def forward(self, x, input_layer):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)
        if self.balance:
            with torch.no_grad():
                # self.W_exc = nn.Linear(self.input_size, self.hidden_size, bias=False)
                # self.W_exc.weight.data = torch.where(self.W.weight.data>=0, self.W.weight.data, 0)
                # self.W_inh = nn.Linear(self.input_size, self.hidden_size, bias=False)
                # self.W_inh.weight.data = torch.where(self.W.weight.data<0, self.W.weight.data, 0)
                # Wx_inh = self.W_inh(x)  # = I_in_inh
                # Wx_exc = self.W_exc(x)  # = I_in_exc

                #self.W_exc.copy_(torch.where(self.W.weight >= 0, self.W.weight, torch.zeros_like(self.W.weight)))
                #self.W_inh.copy_(torch.where(self.W.weight >= 0, self.W.weight, torch.zeros_like(self.W.weight)))
                
                Wx_inh = torch.matmul(x, torch.where(self.W.weight < 0, self.W.weight, torch.zeros_like(self.W.weight)).t())
                Wx_exc = torch.matmul(x, torch.where(self.W.weight >= 0, self.W.weight, torch.zeros_like(self.W.weight)).t())

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s, I_rec_inh, I_rec_exc = self._rlif_cell(Wx, input_layer)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        if self.balance:
            self.I_exc = I_rec_exc+torch.repeat_interleave(Wx_exc.detach(), self.substeps, dim=1)
            self.I_inh = I_rec_inh+torch.repeat_interleave(Wx_inh.detach(), self.substeps, dim=1)
        gc.collect()
        return s

    def _rlif_cell(self, Wx, input_layer):

        # Initializations
        substeps = self.substeps if input_layer else 1
        device = Wx.device
        self.v = torch.zeros(Wx.shape[0], substeps*Wx.shape[1], Wx.shape[2]).to(device)
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = torch.zeros(Wx.shape[0], substeps*Wx.shape[1], Wx.shape[2]).to(device)
        I_rec_inh, I_rec_exc = torch.zeros(Wx.shape[0], substeps*Wx.shape[1], Wx.shape[2]).to(device), torch.zeros(Wx.shape[0], substeps*Wx.shape[1], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):
            
            # Finer loop
            for tt in range(substeps):
                # Compute and save membrane potential (RLIF)
                ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V))
                self.v[:, substeps*t + tt, :] = ut.detach()

                # Compute spikes with surrogate gradient
                st = self.spike_fct(ut - self.threshold)
                s[:, substeps*t + tt, :] = st

                # Compute input currents if necessary (note: the resulting i_rec_exc/inh is equivalent to torch.matmul(st, V))
                if self.balance:
                    I_rec_inh[:, substeps*t + tt, :], I_rec_exc[:, substeps*t + tt, :] = self._signed_matmul(st.detach(), V.detach())
                    # TODO: V x st equivalence check

        return s, I_rec_inh, I_rec_exc

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
        self.spike_fct = SpikeFunctionBoxcar.apply
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
        self.drop = nn.Dropout(p=args.pdrop)

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
    ):
        super().__init__()

        # Fixed parameters
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

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
        self.drop = nn.Dropout(p=args.pdrop)

    def forward(self, x):

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

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
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out
