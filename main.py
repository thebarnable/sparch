import argparse
import torch
import argparse
import os
import shutil
import logging

from sparch.exp import Experiment
from sparch.helpers.parser import add_model_options
from sparch.helpers.parser import add_training_options

RESULTS_FOLDER="exp/tmp"

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Model training on spiking speech commands datasets."
    )
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    print(''.join(f' {k}={v}\n' for k, v in vars(args).items()))
    return args


def exp():
    """
    Runs model training/testing using the configuration specified
    by the parser arguments. Run `python main.py -h` for details.
    """

    # Get experiment configuration from parser
    args = parse_args()

    for i in range(args.trials):
        if args.trials > 1:
            args.seed = i

        # Instantiate class for the desired experiment
        experiment = Experiment(args)

        # Run experiment
        logging.info(f"\n-------- Trial {i+1}/{args.trials} --------\n")
        experiment.forward()

def run_balanced_ae_sample():
    if os.path.exists(RESULTS_FOLDER) and os.path.isdir(RESULTS_FOLDER):
        shutil.rmtree(RESULTS_FOLDER)

    parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    args.seed = 0
    args.new_exp_folder = RESULTS_FOLDER
    args.model = "BalancedRLIF"
    args.dataset = "cue"
    args.n_layers = 1
    args.neurons = 400
    args.dropout = 0
    args.normalization = "none"
    args.track_balance = True
    args.repeat = 20
    args.plot = True
    args.batch_size = 1
    args.auto_encoder = True
    args.single_spike = True
    args.sigma_v = 0.0
    args.dataset_scale = 200
    exp = Experiment(args)

    data, _ = next(iter(exp.train_loader))
    data = data.to(exp.device)
    output, firing_rates = exp.net(data)
    exp.net.plot(RESULTS_FOLDER+"/plots/plot.png", show=True)

def run_sample():
    if os.path.exists(RESULTS_FOLDER) and os.path.isdir(RESULTS_FOLDER):
        shutil.rmtree(RESULTS_FOLDER)

    parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    args.seed = 0
    args.new_exp_folder = RESULTS_FOLDER
    args.model = "RLIF"
    args.dataset = "shd"
    args.dataset_folder = "SHD"
    args.n_layers = 1
    args.neurons = 128
    args.dropout = 0
    args.normalization = "none"
    args.track_balance = True
    args.repeat = 20
    args.plot = True
    args.batch_size = 1
    exp = Experiment(args)
    
    data, _, label = next(iter(exp.train_loader))
    data, label = data.to(exp.device), label.to(exp.device)
    output, firing_rates = exp.net(data)
    exp.net.plot(RESULTS_FOLDER+"/plots/plot.png")

    pred = torch.argmax(output, dim=1)
    acc = torch.mean((label==pred).float())
    
if __name__ == '__main__':
    run_balanced_ae_sample()
