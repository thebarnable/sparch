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
This is where the dataloader is defined for the SHD and SSC datasets.
"""
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpikingDataset(Dataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or
    Spiking Speech Commands (SSC) dataset.

    Arguments
    ---------
    dataset : str
        Name of the dataset, either shd or ssc.
    dataset_folder : str
        Path to folder containing the dataset (h5py file).
    split : str
        Split of the SHD dataset, must be either "train" or "test".
    n_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(
        self,
        dataset,
        dataset_folder,
        split,
        n_steps=100,
        labeled=True,
        repeat=1,
        scale=1
    ):

        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.n_steps = n_steps
        self.n_units = 700
        self.n_classes = 20 if dataset == "shd" else 35
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=self.n_steps)
        self.labeled = labeled
        self.repeat = repeat
        self.scale = scale
        
        self.t_crop = 0

        # Read data from h5py file
        filename = f"{dataset_folder}/{dataset}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]

        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.n_steps, self.n_units])

        x = torch.sparse_coo_tensor(x_idx, x_val, x_size).to(self.device)

        if self.labeled:
            y = self.labels[index]
            return x.to_dense(), y
        else:
            return x.to_dense()

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        ys = torch.LongTensor(ys).to(self.device)

        xs = self.scale * xs.repeat_interleave(self.repeat, axis=1)
        return xs, xlens, ys
    

class CueAccumulationDataset(Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation

    Timing for cue_assignments[0] = [0,0,1,0,1,0,0]:
    t_silent (50ms) silence
    t_cue (100ms)   spikes on first 10 neurons (4% probability)
    t_silent (50ms) silence
    t_cue (100ms)   spikes on first 10 neurons (4% probability)
    t_silent (50ms) silence
    t_cue (100ms)   spikes on second 10 neurons (4% probability)
    ....
    until 2099ms    silence
    t_interval (150ms) spikes on third 10 neurons (4% probability) as recall cue
    """

    def __init__(self, seed=None, labeled=True, repeat=1, scale=1):
        n_cues = 7
        f0 = 40
        t_cue = 100
        t_wait = 1200
        n_symbols = 4 # if 40 neurons: left cue (neurons 0-9), right cue (neurons 10-19), decision cue (neurons 20-29), noise (neurons 30-39)
        p_group = 0.3

        self.repeat = repeat
        self.labeled = labeled
        self.scale = scale
        
        self.dt = 1e-3
        self.t_interval = 150
        self.seq_len = n_cues*self.t_interval + t_wait
        self.t_crop = n_cues * self.t_interval
        self.n_units = 40
        self.n_classes = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel = self.n_units // n_symbols
        prob0 = f0 * self.dt
        t_silent = self.t_interval - t_cue

        length = 200

        # Randomly assign group A and B
        prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
        idx = np.random.choice([0, 1], length)
        probs = np.zeros((length, 2), dtype=np.float32)
        # Assign input spike probabilities
        probs[:, 0] = prob_choices[idx]
        probs[:, 1] = prob_choices[1 - idx]

        cue_assignments = np.zeros((length, n_cues), dtype=int)
        # For each example in batch, draw which cues are going to be active (left or right) -> e.g. cue_assignments[0]=[0,0,1,0,1,0,0]
        for b in range(length):
            cue_assignments[b, :] = np.random.choice([0, 1], n_cues, p=probs[b])

        # Generate input spikes
        input_spike_prob = np.zeros((length, self.seq_len, self.n_units))
        t_silent = self.t_interval - t_cue
        for b in range(length):
            for k in range(n_cues):
                # Input channels only fire when they are selected (left or right)
                c = cue_assignments[b, k]
                input_spike_prob[b, t_silent+k*self.t_interval:t_silent+k *
                                 self.t_interval+t_cue, c*n_channel:(c+1)*n_channel] = prob0

        # Recall cue and background noise
        input_spike_prob[:, -self.t_interval:, 2*n_channel:3*n_channel] = prob0
        input_spike_prob[:, :, 3*n_channel:] = prob0/4.
        input_spikes = self.generate_poisson_noise_np(input_spike_prob, seed)
        self.x = self.scale * torch.tensor(input_spikes).float()
        self.x = self.x.repeat_interleave(self.repeat, axis=1)

        # Generate targets
        self.y = torch.from_numpy((np.sum(cue_assignments, axis=1) > int(n_cues/2)).astype(int))

    def generate_poisson_noise_np(self, prob_pattern, freezing_seed=None):
        if isinstance(prob_pattern, list):
            return [self.generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

        shp = prob_pattern.shape
        rng = np.random.RandomState(freezing_seed)

        spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
        return spikes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.labeled:
            return self.x[index], self.y[index]
        else:
            return self.x[index]
        
    def generateBatch(self, batch):
        if self.labeled:
            xs, ys = zip(*batch)
            xlens = torch.tensor([x.shape[0] for x in xs])
            #ys = torch.LongTensor(ys).to(self.device)
            if len(xs) > 1:
                xs, ys = torch.stack(xs, dim=0), torch.stack(ys, dim=0)
            else:
                xs, ys = xs[0].expand(size=(1,*xs[0].shape)), ys[0].expand(size=(1,*ys[0].shape))

            return xs, xlens, ys
        else:
            xs = batch[0]
            if len(xs.shape) > 2:
                xs = torch.hstack(xs)
            else:
                xs = xs.expand(size=(1, *xs.shape))

            xlens = torch.tensor([x.shape[0] for x in xs])
            return xs, xlens

