import unittest
import argparse
import torch
import argparse
import numpy as np
import os
import shutil
import sys

from sparch.exp import Experiment
from sparch.helpers.parser import add_model_options
from sparch.helpers.parser import add_training_options

E_large=1e-3
E_small=1e-4
FOLDER=".test"

class TestBEEP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if os.path.exists(FOLDER) and os.path.isdir(FOLDER):
            shutil.rmtree(FOLDER)

    # test inference (SRNN.forward()) + gradient calc via eprop (SRNN.grad_batch())
    def test_bptt_sample(self):
        parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
        parser = add_model_options(parser)
        parser = add_training_options(parser)
        args = parser.parse_args()
        args.seed = 0
        args.new_exp_folder = FOLDER+"/test_bptt_sample"
        args.model = "RLIF"
        args.dataset = "shd"
        args.dataset_folder = "SHD"
        args.n_layers = 2
        args.dropout = True
        args.normalization = "batchnorm"
        args.track_balance = True
        exp = Experiment(args)
        
        data, _, label = next(iter(exp.train_loader))
        data, label = data.to(exp.device), label.to(exp.device)
        output, firing_rates = exp.net(data)
        loss = exp.loss_fn(output, label)
        exp.opt.zero_grad()
        loss.backward()
        exp.opt.step()

        pred = torch.argmax(output, dim=1)
        acc = torch.mean((label==pred).float())
        # torch.save({"output": output,
        #             "firing_rates": firing_rates,
        #             "loss": loss,
        #             "acc": acc,
        #             "spikes": exp.net.spikes}, "refs/bptt.pth")
        ref = torch.load("refs/bptt.pth", map_location=exp.device if torch.cuda.is_available() else 'cpu', weights_only=False)
        self.assertLess(torch.abs(ref["spikes"].to(exp.device) - exp.net.spikes).max(), E_small)
        self.assertLess(torch.abs(ref["output"].to(exp.device) - output).max(), E_small)
        self.assertLess(torch.abs(ref["firing_rates"].to(exp.device) - firing_rates).max(), E_small)
        self.assertLess(torch.abs(ref["loss"].to(exp.device) - loss).max(), E_small)
        self.assertLess(torch.abs(ref["acc"].to(exp.device) - acc).max(), E_small)

    def test_beep_sample(self):
        parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
        parser = add_model_options(parser)
        parser = add_training_options(parser)
        args = parser.parse_args()
        args.seed = 0
        args.new_exp_folder = FOLDER+"/test_beep_sample"
        args.model = "BalancedRLIF"
        args.dataset = "shd"
        args.dataset_folder = "SHD"
        args.n_layers = 1
        args.dropout = 0
        args.normalization = "none"
        args.single_spike = True
        args.track_balance = True
        args.dataset_scale = 200
        exp = Experiment(args)
        
        data, _, label = next(iter(exp.train_loader))
        data, label = data.to(exp.device), label.to(exp.device)
        output, firing_rates = exp.net(data)
        loss = exp.loss_fn(output, label)
        exp.opt.zero_grad()
        loss.backward()
        exp.opt.step()

        pred = torch.argmax(output, dim=1)
        acc = torch.mean((label==pred).float())
        # torch.save({"output": output,
        #             "firing_rates": firing_rates,
        #             "loss": loss,
        #             "acc": acc,
        #             "spikes": exp.net.spikes,
        #             "currents_exc": exp.net.currents_exc,
        #             "currents_inh": exp.net.currents_inh,
        #             "voltages": exp.net.voltages}, "refs/beep.pth")
        ref = torch.load("refs/beep.pth", map_location=exp.device if torch.cuda.is_available() else 'cpu', weights_only=False)
        self.assertLess(torch.abs(ref["spikes"].to(exp.device) - exp.net.spikes).max(), E_small)
        self.assertLess(torch.abs(ref["output"].to(exp.device) - output).max(), E_small)
        self.assertLess(torch.abs(ref["firing_rates"].to(exp.device) - firing_rates).max(), E_small)
        self.assertLess(torch.abs(ref["loss"].to(exp.device) - loss).max(), E_small)
        self.assertLess(torch.abs(ref["acc"].to(exp.device) - acc).max(), E_small)
        self.assertLess(torch.abs(ref["currents_exc"].to(exp.device) - exp.net.currents_exc).max(), E_small)
        self.assertLess(torch.abs(ref["currents_inh"].to(exp.device) - exp.net.currents_inh).max(), E_small)
        self.assertLess(torch.abs(ref["voltages"].to(exp.device) - exp.net.voltages).max(), E_small)

    def test_balanced_autoencoder(self):
        parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
        parser = add_model_options(parser)
        parser = add_training_options(parser)
        args = parser.parse_args()
        args.seed = 0
        args.new_exp_folder = FOLDER+"/test_balanced_autoencoder"
        args.model = "BalancedRLIF"
        args.dataset = "cue"
        args.n_layers = 1
        args.neurons = 400
        args.dropout = 0
        args.normalization = "none"
        args.single_spike = True
        args.track_balance = True
        args.repeat = 20
        args.batch_size = 1
        args.auto_encoder = True
        args.sigma_v = 0.0
        args.dataset_scale = 200
        exp = Experiment(args)
        
        data, _ = next(iter(exp.train_loader))
        data = data.to(exp.device)
        output, firing_rates = exp.net(data)

        pred = torch.argmax(output, dim=1)
        # torch.save({"output": output,
        #             "firing_rates": firing_rates,
        #             "spikes": exp.net.spikes}, "refs/balanced_ae.pth")
        ref = torch.load("refs/balanced_ae.pth", map_location=exp.device if torch.cuda.is_available() else 'cpu', weights_only=False)
        self.assertLess(torch.abs(ref["spikes"].to(exp.device) - exp.net.spikes).max(), E_small)
        self.assertLess(torch.abs(ref["output"].to(exp.device) - output).max(), E_small)
        self.assertLess(torch.abs(ref["firing_rates"].to(exp.device) - firing_rates).max(), E_small)  

    # test 2 epochs
    def test_bptt_epoch(self):
        parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
        parser = add_model_options(parser)
        parser = add_training_options(parser)
        args = parser.parse_args()
        args.seed = 0
        args.new_exp_folder = FOLDER+"/test_bptt_epoch"
        args.model = "RLIF"
        args.dataset = "shd"
        args.dataset_folder = "SHD"
        args.n_layers = 2
        args.dropout = 0.1
        args.normalization = "batchnorm"
        args.n_epochs = 2
        print(''.join(f' {k}={v}\n' for k, v in vars(args).items()))
        exp = Experiment(args)
        exp.forward()
        #shutil.copy(args.new_exp_folder+"/results.pth", "refs/bptt_epoch.pth")
        ref = torch.load("refs/bptt_epoch.pth", map_location=exp.device if torch.cuda.is_available() else 'cpu', weights_only=False)
        results = torch.load(args.new_exp_folder+"/results.pth", map_location=exp.device if torch.cuda.is_available() else 'cpu', weights_only=False)
        self.assertLess(torch.abs(torch.Tensor(ref["train_accs"]) - torch.Tensor(results["train_accs"])).max(), E_small)
        self.assertLess(torch.abs(torch.Tensor(ref["train_frs"]) - torch.Tensor(results["train_frs"])).max(), E_small)
        self.assertLess(torch.abs(torch.Tensor(ref["validation_accs"]) - torch.Tensor(results["validation_accs"])).max(), E_small)
        self.assertLess(torch.abs(torch.Tensor(ref["validation_frs"]) - torch.Tensor(results["validation_frs"])).max(), E_small)
        self.assertLess(abs(ref["test_acc"] - results["test_acc"]), E_small)
        self.assertLess(abs(ref["test_fr"] - results["test_fr"]), E_small)
        self.assertLess(abs(ref["best_acc"] - results["best_acc"]), E_small)
        self.assertEqual(ref["best_epoch"], results["best_epoch"])


if __name__ == '__main__':
    # Run all tests via "python testing.py". Run individual tests via "python testing.py <test_name>", e.g. "python testing.py test_bptt_epoch".

    if len(sys.argv) == 1:
        unittest.main()
    else:
        test_name = sys.argv.pop(1)
        suite = unittest.TestSuite()
        suite.addTest(TestBEEP(test_name))
        runner = unittest.TextTestRunner()
        runner.run(suite)
