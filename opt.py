import numpy as np
import optuna
import sqlalchemy
import os
import argparse

from boerlin import main as boerlin_main
from boerlin import parse_args as boerlin_args

CUE=True # if True, use Cue dataset, else SHD

if not CUE: # SHD
    search_space = {
        'n':            lambda trial : trial.suggest_int('n', 1000, 15000, log=False),
        'repeat':       lambda trial : trial.suggest_int('repeat', 1, 200, log=False),
        'input_scale':  lambda trial : trial.suggest_int('input_scale', 10, 500, log=False),
        'weight_scale': lambda trial : trial.suggest_float('weight_scale', 0.1, 10, log=True),
        'weight_sparsity':     lambda trial : trial.suggest_float('binomial', 0.0, 0.3, log=False)
    }
else:  # CUE
    search_space = {
        'n':            lambda trial : trial.suggest_int('n', 100, 500, log=False),
        'repeat':       lambda trial : trial.suggest_int('repeat', 1, 100, log=False),
        'input_scale':  lambda trial : trial.suggest_int('input_scale', 100, 500, log=False),
        'weight_scale': lambda trial : trial.suggest_float('weight_scale', 0.1, 10, log=True),
        'weight_sparsity':     lambda trial : trial.suggest_float('binomial', 0.0, 0.3, log=False)
    }

def network_fn(trial):
    # Define the search space
    args = boerlin_args(default=True)
    
    for key in search_space.keys():
        setattr(args, key, search_space[key](trial))

    args.seed = 0
    args.sigma_s = 0.0
    args.auto_encoder = True
    args.j = -1
    args.data = "cue" if CUE else "shd"
    args.w_init = "rand"
    args.track_balance = True
    args.save_path = "results_"+args.data
    args.save = f"{args.n}_{args.repeat}_{args.input_scale}_{args.weight_scale:3f}_{args.weight_sparsity:3f}"

    results = boerlin_main(args, trial)
    return results[0] # MSE between x and x_snn

def main(opt_args):
    if os.path.exists("results/optuna_%s" % opt_args.study) and opt_args.new_db:
        print("Optimization with name %s already exists, please choose another name" % opt_args.study)
        exit(0)
    sqlalchemy.create_engine(opt_args.db)

    study = optuna.create_study(study_name = opt_args.study, direction='minimize', sampler = optuna.samplers.TPESampler(),
                                load_if_exists = not opt_args.new_db, storage = opt_args.db)

    study.optimize(network_fn, n_trials=opt_args.evals, catch=(Exception,), gc_after_trial=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot script')
    parser.add_argument('--db', type=str, default='sqlite:///optuna.db', help='path to database')
    parser.add_argument('--study', type=str, default='optuna_study', help='name of the optimization study to perform')
    parser.add_argument('--new-db', action='store_true', help='create new database instead of loading existing one')
    parser.add_argument('--evals', type=int, default=100)

    opt_args = parser.parse_args()

    main(opt_args)
