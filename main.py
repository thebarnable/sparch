import unittest
import argparse
import torch
import argparse
import numpy as np
import os
import shutil

from sparch.exp import Experiment
from sparch.parsers.model_config import add_model_options
from sparch.parsers.training_config import add_training_options

RESULTS_FOLDER="exp/tmp"

def run_sample():
    if os.path.exists(RESULTS_FOLDER) and os.path.isdir(RESULTS_FOLDER):
        shutil.rmtree(RESULTS_FOLDER)

    parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    args.seed = 0
    args.new_exp_folder = RESULTS_FOLDER
    args.model_type = "RLIF"
    args.dataset_name = "shd"
    args.data_folder = "SHD"
    args.nb_layers = 2
    args.pdrop = 0
    args.normalization = "none"
    args.balance = True
    args.substeps = 20
    args.plot = True
    args.batch_size = 1
    exp = Experiment(args)
    
    data, _, label = next(iter(exp.train_loader))
    data, label = data.to(exp.device), label.to(exp.device)
    output, firing_rates = exp.net(data)
    exp.net.plot(RESULTS_FOLDER+"/plots/plot.png")

    pred = torch.argmax(output, dim=1)
    acc = torch.mean((label==pred).float())
    spikes = torch.stack(exp.net.spikes, dim=0)
    
if __name__ == '__main__':
    run_sample()
