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
This is where the dataloaders and defined for the HD and SC datasets.
"""
import logging
import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchaudio_augmentations import ComposeMany
from torchaudio_augmentations import Gain
from torchaudio_augmentations import Noise
from torchaudio_augmentations import PolarityInversion
from torchaudio_augmentations import RandomApply
from torchaudio_augmentations import Reverb

logger = logging.getLogger(__name__)


class HeidelbergDigits(Dataset):
    """
    Dataset class for the original non-spiking Heidelberg Digits (HD)
    dataset. Generated mel-spectrograms use 40 bins by default.

    Arguments
    ---------
    dataset_folder : str
        Path to folder containing the Heidelberg Digits dataset.
    split : str
        Split of the HD dataset, must be either "train" or "test".
    augment : bool
        Whether to perform data augmentation or not.
    min_snr, max_snr : float
        Minimum and maximum amounts of noise if augmentation is used.
    p_noise : float in (0, 1)
        Probability to apply noise if augmentation is used, i.e.,
        proportion of examples to which augmentation is applied.
    """

    def __init__(
        self,
        dataset_folder,
        split,
        augment,
        min_snr=0.0001,
        max_snr=0.9,
        p_noise=0.1,
    ):

        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.dataset_folder = dataset_folder
        filename = self.dataset_folder + "/" + split + "_filenames.txt"
        with open(filename, "r") as f:
            self.file_list = f.read().splitlines()

        # Data augmentation
        if augment and split == "train":
            transforms = [
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(min_snr, max_snr)], p_noise),
                RandomApply([Gain()], p=0.3),
                RandomApply([Reverb(sample_rate=16000)], p=0.6),
            ]
            self.transf = ComposeMany(transforms, num_augmented_samples=1)
        else:
            self.transf = lambda x: x.unsqueeze(dim=0)

        self.n_units   = 40
        self.n_classes = 20

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        # Read waveform
        filename = self.file_list[index]
        x = self.dataset_folder + "/audio/" + filename
        x, _ = torchaudio.load(x)

        # Apply augmentation
        x = self.transf(x).squeeze(dim=0)

        # Compute acoustic features
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=40)

        # Get label (digits 0-9 in eng and germ)
        y = int(filename[-6])
        if filename[5] == "g":
            y += 10

        return x, y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys


class SpeechCommands(Dataset):
    """
    Dataset class for the original non-spiking Speech Commands (SC)
    dataset. Generated mel-spectrograms use 40 bins by default.

    Arguments
    ---------
    dataset_folder : str
        Path to folder containing the Heidelberg Digits dataset.
    split : str
        Split of the HD dataset, must be either "train" or "test".
    augment : bool
        Whether to perform data augmentation or not.
    min_snr, max_snr : float
        Minimum and maximum amounts of noise if augmentation is used.
    p_noise : float in (0, 1)
        Probability to apply noise if augmentation is used, i.e.,
        proportion of examples to which augmentation is applied.
    """

    def __init__(
        self,
        dataset_folder,
        split,
        augment,
        min_snr,
        max_snr,
        p_noise,
    ):

        if split not in ["training", "validation", "testing"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.dataset_folder = dataset_folder
        EXCEPT_FOLDER = "_background_noise_"

        def load_list(filename):
            filepath = os.path.join(self.dataset_folder, filename)
            with open(filepath) as f:
                return [os.path.join(self.dataset_folder, i.strip()) for i in f]

        if split == "training":
            files = sorted(str(p) for p in Path(dataset_folder).glob("*/*.wav"))
            exclude = load_list("validation_list.txt") + load_list("testing_list.txt")
            exclude = set(exclude)
            self.file_list = [
                w for w in files if w not in exclude and EXCEPT_FOLDER not in w
            ]
        else:
            self.file_list = load_list(str(split) + "_list.txt")

        self.labels = sorted(next(os.walk("./" + dataset_folder))[1])[1:]

        # Data augmentation
        if augment and split == "training":
            transforms = [
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(min_snr, max_snr)], p_noise),
                RandomApply([Gain()], p=0.3),
                RandomApply([Reverb(sample_rate=16000)], p=0.6),
            ]
            self.transf = ComposeMany(transforms, num_augmented_samples=1)
        else:
            self.transf = lambda x: x.unsqueeze(dim=0)

        self.n_units   = 40
        self.n_classes = 35

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        # Read waveform
        filename = self.file_list[index]
        x, _ = torchaudio.load(filename)

        # Apply augmentation
        x = self.transf(x).squeeze(dim=0)

        # Compute acoustic features
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=40)

        # Get label
        relpath = os.path.relpath(filename, self.dataset_folder)
        label, _ = os.path.split(relpath)
        y = torch.tensor(self.labels.index(label))

        return x, y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys
