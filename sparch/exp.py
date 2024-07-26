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
This is to define the experiment class used to perform training and testing
of ANNs and SNNs on all speech command recognition datasets.
"""
import errno
import logging
import os
import time
import datetime
from datetime import timedelta
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sparch.dataloaders.nonspiking_datasets import load_hd_or_sc
from sparch.dataloaders.spiking_datasets import load_shd_or_ssc
from sparch.models.anns import ANN
from sparch.models.snns import SNN
from sparch.parsers.model_config import print_model_options
from sparch.parsers.training_config import print_training_options

logger = logging.getLogger(__name__)


class Experiment:
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, args):

        # New model config
        self.model_type = args.model_type
        self.nb_layers = args.nb_layers
        self.nb_hiddens = args.nb_hiddens
        self.pdrop = args.pdrop
        self.normalization = args.normalization
        self.use_bias = False
        self.bidirectional = args.bidirectional
        self.balance = args.balance

        # Training config
        self.use_pretrained_model = args.use_pretrained_model
        self.only_do_testing = args.only_do_testing
        self.load_exp_folder = args.load_exp_folder
        self.new_exp_folder = args.new_exp_folder
        self.dataset_name = args.dataset_name
        self.data_folder = args.data_folder
        self.log_tofile = args.log_tofile
        self.save_best = args.save_best
        self.batch_size = args.batch_size
        self.nb_epochs = args.nb_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.use_regularizers = args.use_regularizers
        self.reg_factor = args.reg_factor
        self.reg_fmin = args.reg_fmin
        self.reg_fmax = args.reg_fmax
        self.use_augm = args.use_augm
        self.plot = args.plot
        self.plot_epoch_freq = args.plot_epoch_freq
        self.plot_classes = [0, 2, 4]
        self.plot_class_cnt = {p:0 for p in self.plot_classes}
        self.plot_class_cnt_max = 5
        self.plot_batch_id = 0

        # Set seed
        if args.seed == -1:
            args.seed = int(datetime.datetime.now().timestamp())
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()
        print_model_options(args)
        print_training_options(args)

        logging.info(f"\nSaving results and trained model in {self.exp_folder}\n")

        # Set device
        if args.gpu != -1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

        # Define optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), self.lr)

        # Define learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.opt,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        if not self.only_do_testing:
            # Initialize best accuracy
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_acc, _ = self.valid_one_epoch(self.start_epoch, self.valid_loader, test=False)
                best_epoch = self.start_epoch
            else:
                best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            train_accs, train_frs, validation_accs, validation_frs = [], [], [], []
            logging.info("\n------ Begin training ------\n")

            for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
                train_acc, train_fr = self.train_one_epoch(e)
                valid_acc, valid_fr = self.valid_one_epoch(e, self.valid_loader, test=False)
                self.scheduler.step(valid_acc) # Update learning rate

                # Update best epoch and accuracy
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_epoch = e

                    # Save best model
                    if self.save_best:
                        torch.save(self.net, f"{self.checkpoint_dir}/best_model.pth")
                        logging.info(f"\nBest model saved with valid acc={valid_acc}")

                train_accs.append(train_acc)
                train_frs.append(train_fr)
                validation_accs.append(valid_acc)
                validation_frs.append(valid_fr)
                logging.info("\n-----------------------------\n")

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

            # Loading best model
            if self.save_best:
                self.net = torch.load(
                    f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                )
                logging.info(
                    f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                )
            else:
                logging.info(
                    "Cannot load best model because save_best option is "
                    "disabled. Model from last epoch is used for testing."
                )

        # Test trained model
        logging.info("\n------ Begin Testing ------\n")
        if self.dataset_name in ["sc", "ssc"]:
            test_acc, test_fr = self.valid_one_epoch(e, self.test_loader, test=True)
        else:
            logging.info("\nThis dataset uses the same split for validation and testing.\n")
            test_acc, test_fr = self.valid_one_epoch(e, self.valid_loader, test=True)
        logging.info("\n-----------------------------\n")

        # Save results summary
        results = {}
        results["train_accs"] = train_accs
        results["train_frs"] = train_frs
        results["validation_accs"] = validation_accs
        results["validation_frs"] = validation_frs
        results["test_acc"] = test_acc
        results["test_fr"] = test_acc
        results["best_acc"] = best_acc
        results["best_epoch"] = best_epoch
        torch.save(results, f"{self.exp_folder}/results.pth")

    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.load_path
                )

        # Use given path for new model folder
        elif self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        # Generate a path for new model from chosen config
        else:
            outname = self.dataset_name + "_" + self.model_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_bdir" if self.bidirectional else "_udir"
            outname += "_reg" if self.use_regularizers else "_noreg"
            outname += "_lr" + str(self.lr)
            exp_folder = "exp/" + outname.replace(".", "_") + "/run0"
            while os.path.exists(exp_folder): # if run0 already exists, create run1 (and so on)
                exp_folder=exp_folder[:-1] + str(int(exp_folder[-1])+1)

        # For a new model check that out path does not exist
        if not self.use_pretrained_model and os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder + "/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if self.plot:
            self.plot_dir = exp_folder + "/plots/"
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)

        self.exp_folder = exp_folder

    def init_logging(self):
        """
        This function sets the experimental log to be written either to
        a dedicated log file, or to the terminal.
        """
        if self.log_tofile:
            logging.FileHandler(
                filename=self.log_dir + "exp.log",
                mode="a",
                encoding=None,
                delay=False,
            )
            logging.basicConfig(
                filename=self.log_dir + "exp.log",
                level=logging.INFO,
                format="%(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )

    def init_dataset(self):
        """
        This function prepares dataloaders for the desired dataset.
        """
        # For the spiking datasets
        if self.dataset_name in ["shd", "ssc"]:

            self.nb_inputs = 700
            self.nb_outputs = 20 if self.dataset_name == "shd" else 35

            self.train_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                nb_steps=100,
                shuffle=True,
            )
            self.valid_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                nb_steps=100,
                shuffle=False,
            )
            if self.dataset_name == "ssc":
                self.test_loader = load_shd_or_ssc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    nb_steps=100,
                    shuffle=False,
                )
            if self.use_augm:
                logging.warning(
                    "\nWarning: Data augmentation not implemented for SHD and SSC.\n"
                )

        # For the non-spiking datasets
        elif self.dataset_name in ["hd", "sc"]:

            self.nb_inputs = 40
            self.nb_outputs = 20 if self.dataset_name == "hd" else 35

            self.train_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=True,
            )
            self.valid_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=False,
            )
            if self.dataset_name == "sc":
                self.test_loader = load_hd_or_sc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    use_augm=self.use_augm,
                    shuffle=False,
                )
            if self.use_augm:
                logging.info("\nData augmentation is used\n")

        else:
            raise ValueError(f"Invalid dataset name {self.dataset_name}")

    def init_model(self):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        input_shape = (self.batch_size, None, self.nb_inputs)
        layer_sizes = [self.nb_hiddens] * (self.nb_layers - 1) + [self.nb_outputs]

        if self.use_pretrained_model:
            self.net = torch.load(self.load_path, map_location=self.device)
            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

        elif self.model_type in ["LIF", "adLIF", "RLIF", "RadLIF"]:

            self.net = SNN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                neuron_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
                balance=self.balance
            ).to(self.device)

            logging.info(f"\nCreated new spiking model:\n {self.net}\n")

        elif self.model_type in ["MLP", "RNN", "LiGRU", "GRU"]:

            self.net = ANN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                ann_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
            ).to(self.device)

            logging.info(f"\nCreated new non-spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.model_type}")

        self.nb_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logging.info(f"Total number of trainable parameters is {self.nb_params}")

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        Returns: training accuracy (float), spike rate average (float)
        """
        start = time.time()
        self.net.train()
        losses, accs = [], []
        epoch_spike_rate = 0

        # Loop over batches from train set
        for step, (x, _, y) in enumerate(self.train_loader):

            # Dataloader uses cpu to allow pin memory
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass through network
            output, firing_rates = self.net(x)

            # Compute loss
            loss_val = self.loss_fn(output, y)
            losses.append(loss_val.item())

            # Spike activity
            if self.net.is_snn:
                epoch_spike_rate += torch.mean(firing_rates)

                if self.use_regularizers:
                    reg_quiet = F.relu(self.reg_fmin - firing_rates).sum()
                    reg_burst = F.relu(firing_rates - self.reg_fmax).sum()
                    loss_val += self.reg_factor * (reg_quiet + reg_burst)

            # Backpropagate
            self.opt.zero_grad()
            loss_val.backward()
            self.opt.step()

            # Compute accuracy with labels
            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)

            # Plot if necessary
            label=int(y[self.plot_batch_id])
            if self.plot and (e-1) % self.plot_epoch_freq == 0 and label in self.plot_classes and self.plot_class_cnt[label] < self.plot_class_cnt_max:
                self.net.plot(self.plot_dir+f"epoch{e}_class{label}_{self.plot_class_cnt[label]}.png")
                self.plot_class_cnt[label]+=1

        # Learning rate of whole epoch
        current_lr = self.opt.param_groups[-1]["lr"]
        logging.info(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        logging.info(f"Epoch {e}: Train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        logging.info(f"Epoch {e}: Train acc={train_acc}")

        # Train spike activity of whole epoch
        if self.net.is_snn:
            epoch_spike_rate /= step
            logging.info(f"Epoch {e}: Train mean act rate={epoch_spike_rate}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        logging.info(f"Epoch {e}: Train elapsed time={elapsed}")

        return train_acc, epoch_spike_rate

    def valid_one_epoch(self, e, dataloader, test=False):
        """
        This function tests the model with a single pass over the
        validation/test set (given via dataloader)
        Returns: validation accuracy (float), spike rate average (float)
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0

            # Loop over batches from validation set
            for step, (x, _, y) in enumerate(dataloader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates = self.net(x)

                # Compute loss
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)

            # Validation loss of whole epoch
            mean_acc = np.mean(accs)
            if test:
                logging.info(f"Test loss={np.mean(losses)}")
                logging.info(f"Test acc={mean_acc}")
            else:
                logging.info(f"Epoch {e}: Valid loss={np.mean(losses)}")
                logging.info(f"Epoch {e}: Valid acc={mean_acc}")

            # Validation spike activity of whole epoch
            if self.net.is_snn:
                epoch_spike_rate /= step
                if test:
                    logging.info(f"Test mean act rate={epoch_spike_rate}")
                else:
                    logging.info(f"Epoch {e}: valid mean act rate={epoch_spike_rate}")

            return mean_acc, epoch_spike_rate
