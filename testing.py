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
        args.model_type = "RLIF"
        args.dataset_name = "shd"
        args.data_folder = "SHD"
        args.nb_layers = 3
        args.pdrop = True
        args.normalization = "batchnorm"
        args.balance = False
        args.substeps = 1
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
        spikes = torch.stack(exp.net.spikes, dim=0)
        # torch.save({"output": output,
        #             "firing_rates": firing_rates,
        #             "loss": loss,
        #             "acc": acc,
        #             "spikes": spikes}, "refs/bptt.pth")
        ref = torch.load("refs/bptt.pth")
        self.assertLess(torch.abs(ref["spikes"].to(exp.device) - spikes).max(), E_small)
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
        args.model_type = "RLIF"
        args.dataset_name = "shd"
        args.data_folder = "SHD"
        args.nb_layers = 2
        args.pdrop = 0
        args.normalization = "none"
        args.balance = True
        args.substeps = 1
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
        spikes = torch.stack(exp.net.spikes, dim=0)
        # torch.save({"output": output,
        #             "firing_rates": firing_rates,
        #             "loss": loss,
        #             "acc": acc,
        #             "spikes": spikes}, "refs/beep.pth")
        ref = torch.load("refs/beep.pth")
        self.assertLess(torch.abs(ref["spikes"].to(exp.device) - spikes).max(), E_small)
        self.assertLess(torch.abs(ref["output"].to(exp.device) - output).max(), E_small)
        self.assertLess(torch.abs(ref["firing_rates"].to(exp.device) - firing_rates).max(), E_small)
        self.assertLess(torch.abs(ref["loss"].to(exp.device) - loss).max(), E_small)
        self.assertLess(torch.abs(ref["acc"].to(exp.device) - acc).max(), E_small)

    # test 2 epochs
    def test_bptt_epoch(self):
        parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
        parser = add_model_options(parser)
        parser = add_training_options(parser)
        args = parser.parse_args()
        args.seed = 0
        args.new_exp_folder = FOLDER+"/test_bptt_epoch"
        args.model_type = "RLIF"
        args.dataset_name = "shd"
        args.data_folder = "SHD"
        args.nb_layers = 3
        args.pdrop = 0.1
        args.normalization = "batchnorm"
        args.balance = False
        args.substeps = 1
        args.nb_epochs = 2
        print(''.join(f' {k}={v}\n' for k, v in vars(args).items()))
        exp = Experiment(args)
        exp.forward()
        #shutil.copy(args.new_exp_folder+"/results.pth", "refs/bptt_epoch.pth")
        ref = torch.load("refs/bptt_epoch.pth")
        results = torch.load(args.new_exp_folder+"/results.pth")
        self.assertLess(torch.abs(torch.Tensor(ref["train_accs"]) - torch.Tensor(results["train_accs"])).max(), E_small)
        self.assertLess(torch.abs(torch.Tensor(ref["train_frs"]) - torch.Tensor(results["train_frs"])).max(), E_small)
        self.assertLess(torch.abs(torch.Tensor(ref["validation_accs"]) - torch.Tensor(results["validation_accs"])).max(), E_small)
        self.assertLess(torch.abs(torch.Tensor(ref["validation_frs"]) - torch.Tensor(results["validation_frs"])).max(), E_small)
        self.assertLess(abs(ref["test_acc"] - results["test_acc"]), E_small)
        self.assertLess(abs(ref["test_fr"] - results["test_fr"]), E_small)
        self.assertLess(abs(ref["best_acc"] - results["best_acc"]), E_small)
        self.assertEqual(ref["best_epoch"], results["best_epoch"])


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestBEEP('test_bptt_epoch'))
    # suite.addTest(TestBEEP('test_bptt_sample'))
    # suite.addTest(TestBEEP('test_beep_sample'))
    
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
