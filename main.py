import argparse
import torch
import argparse
import os
import shutil
import logging
import tqdm
import numpy as np

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
    args.model = "RLIF"
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
    args.bidirectional = False
    args.balance = True
    args.fix_w_in = True
    args.fix_w_rec = True
    args.fix_w_out = True
    args.fix_tau_rec = True
    args.fix_tau_out = True
    exp = Experiment(args)

    data, _ = next(iter(exp.train_loader))
    data = data.to(exp.device)
    exp.net(data)
    print("Balance: ", exp.net.balance_val)
    exp.net.plot(RESULTS_FOLDER+"/plots/plot.png", show=True)
    
def run_balanced_ae_samples(N=5):
    if os.path.exists(RESULTS_FOLDER) and os.path.isdir(RESULTS_FOLDER):
        shutil.rmtree(RESULTS_FOLDER)

    parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    args.seed = 0
    args.new_exp_folder = RESULTS_FOLDER
    args.model = "RLIF"
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
    args.bidirectional = False
    args.balance = True
    args.fix_w_in = True
    args.fix_w_rec = True
    args.fix_w_out = True
    args.fix_tau_rec = True
    args.fix_tau_out = True
    
    #exp = Experiment(args)
    #n_samples = N
    #balance_avg = 0
    #for _ in tqdm.tqdm(range(n_samples), desc="Balanced AE"):
    #    data, _ = next(iter(exp.train_loader))
    #    data = data.to(exp.device)
    #    exp.net(data)
    #    balance_avg += exp.net.balance_val
    
    factors = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.9875, 0.99, 0.9925, 0.995, 1.0]
    balances_vals = []
    pbar = tqdm.tqdm(total=len(factors) * N, desc="Balanced AE")
    i = 0
    for factor in factors:
        args.ae_fac = factor
        exp = Experiment(args)
        args.new_exp_folder = RESULTS_FOLDER + "_" + str(i)
        balances_vals.append([])
        for _ in range(N):
            data, _ = next(iter(exp.train_loader))
            data = data.to(exp.device)
            exp.net(data)
            balances_vals[-1].append(exp.net.balance_val)
            pbar.update(1)
            i += 1
    
    pbar.close()
    balances_vals = np.stack([factors, np.array(balances_vals).mean(axis=1)])
    np.savetxt(RESULTS_FOLDER + "/balances.csv", balances_vals, delimiter=",")       
    
    #print("Balance: ", balance_avg/n_samples)    
    # Plot last sample
    #exp.net.plot(RESULTS_FOLDER+"/plots/plot.png", show=True)

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
    args.dataset = "cue"
    args.n_layers = 1
    args.neurons = 100
    args.dropout = 0
    args.normalization = "none"
    args.track_balance = True
    args.repeat = 20
    args.plot = True
    args.batch_size = 1
    args.auto_encoder = False
    args.single_spike = True
    args.sigma_v = 0.0
    args.dataset_scale = 200
    args.bidirectional = False
    args.balance = False
    args.fix_w_in = False
    args.fix_w_rec = False
    args.fix_w_out = False
    args.fix_tau_rec = False
    args.fix_tau_out = False
    exp = Experiment(args)
    
    data, _, label = next(iter(exp.train_loader))
    data, label = data.to(exp.device), label.to(exp.device)
    output, _ = exp.net(data)
    print("Balance: ", exp.net.balance_val)
    #exp.net.plot(RESULTS_FOLDER+"/plots/plot.png")

    pred = torch.argmax(output, dim=1)
    acc = torch.mean((label==pred).float())
    print("Output: ", output)
    print("Accuracy: ", acc.item())
    print("Predicted: ", pred)
    print("Actual: ", label)
    
if __name__ == '__main__':
    exp()
    #run_sample()
    #run_balanced_ae_samples(N=5)
