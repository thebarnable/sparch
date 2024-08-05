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

class TestBEEP(unittest.TestCase):
    # test inference (SRNN.forward()) + gradient calc via eprop (SRNN.grad_batch())
    def test_beep_sample(self):
        if os.path.exists(".test") and os.path.isdir(".test"):
            shutil.rmtree(".test")

        parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
        parser = add_model_options(parser)
        parser = add_training_options(parser)
        args = parser.parse_args()
        args.seed = 0
        args.new_exp_folder = ".test"
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
        

if __name__ == '__main__':
    unittest.main()
