# BSNNs: Balanced and Efficient Spiking Neural Networks

This Pytorch toolkit is a framework for developing and analyzing balanced spiking neural networks (BSNNs).
Please refer to our publication, "Balanced and Efficient Spiking Neural Networks" by T. Stadtmann, J. Paeßens & T. Gemmeke, for more details.

It is based on the [Sparch toolkit](https://github.com/idiap/sparch), introduced in [A Surrogate Gradient Spiking Baseline for Speech Command Recognition](https://doi.org/10.3389/fnins.2022.865897) by A. Bittar and P. Garner (2022).
All original functionality is obtained, refer to the original repository for more details.

## Installation
```
git clone git@github.com:thebarnable/sparch.git
cd sparch
pip install -r requirements.txt
python setup.py install
```

## Run training

All experiments can be started from the main.py script. Use `python main.py --help` to see all parameters.
To reproduce the first set of experiments from the paper (Fig. 4), you can exemplarily run the following commands:

```
# Baseline
python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_multispike --trials 3 --plot --balance-metric lowpass --gpu 0 --repeat 4
# Baseline single spike
python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_singlespike --single-spike --repeat 4 --trials 3 --plot --balance-metric lowpass --gpu 0
# LSM
python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-in --fix-w-rec --fix-tau-out --fix-tau-rec --balance --plot --new-exp-folder lsm --balance-metric lowpass --gpu 0
# CUBA
python main.py --V-scale 0.3 --V-slow-scale 0.2 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-w-in --new-exp-folder cuba_fixed --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0 --slow-dynamics --fix-w-rec
# Train W & alpha
python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --balance --plot --new-exp-folder train_all --balance-metric lowpass --gpu 0
```

## Run balanced network sandbox

We implemented a sandbox script for working with a balanced spiking auto-encoder, `boerlin.py`. 
It re-implements the model introduced by [Boerlin et al.](Predictive Coding of Dynamical Variables in Balanced
Spiking Networks). To run the most simple setup, call: `python boerlin.py --auto-encoder --plot`

## License
The original Sparch toolkit is licensed under the BSD-3 license. 
All files added in this repository are also licensed under the BSD-3 license.
All files with minor changes have an amended copyright header.
All files with significant changes have a new copyright header.
For licensing of all new and significantly changed files, refer to the new `LICENSE.md` file. For all others, refer to the original Sparch repository.
