import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean, correlation #cosine, cityblock
import numpy as np
import os
import sys
from pathlib import Path
import math
import yaml
import argparse
import random

RED = "#D17171"
YELLOW = "#F3A451"
GREEN = "#7B9965"
BLUE = "#5E7DAF"
DARKBLUE = "#3C5E8A"
DARKRED = "#A84646"
VIOLET = "#886A9B"
GREY = "#636363"

DATA_FOLDER_REMOTE="/mnt/data4tb2/stadtmann/paper/beep"
DATA_FOLDER_HOME="/home/stadtmann/1_Projects/sparch"
DATA_FOLDER_HO="/home/tim/Projects/beep"
OUTPUT_FOLDER=DATA_FOLDER_HOME+"/plots"

SAVE=True
PLOT=False
SCORE=""

dist_fct = lambda i_exc,i_inh : euclidean(i_exc/i_exc.max(), i_inh/i_inh.max())/np.sqrt(len(i_exc))

def params_from_file_name(filename):
    IDX_DATASET=0
    IDX_NEURON=1
    IDX_LAYER=2
    IDX_DROPOUT=3
    IDX_NORM=5
    IDX_BIAS=6
    IDX_REG=8
    IDX_LR=9
    IDX_ST=11
    IDX_BALANCE=12

    f=filename.split("/")[-1].split("_")
    return f[IDX_DATASET], f[IDX_NEURON], f[IDX_LAYER].split("lay")[0], f[IDX_LAYER].split("lay")[1], float(f[IDX_DROPOUT][-1]+"."+f[IDX_DROPOUT+1]), f[IDX_NORM], int(f[IDX_ST].split("st")[1])

def recurse_dir(path):
    folders = []
    for (dirpath, dirs, files) in os.walk(path):
        # potential result folder if 
        # - contains run0 or trial_0 folders
        # - contains result.pth directly
        if "run0" in dirs or "trial_0" in dirs or ("results.pth" in files and "run" not in dirpath and "trial_" not in dirpath):
            folders.append(dirpath)

    return folders

def plot_results(path):
    path = os.path.abspath(path)
    folders = recurse_dir(path)
    fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]}, figsize=(30,15))
    fig.subplots_adjust(hspace=0)

    colors  = [BLUE,RED,GREEN,YELLOW,VIOLET, DARKRED, DARKBLUE, GREY]
    for i,folder in enumerate(folders):
        if not args.new_name:
            dataset, neuron, n_layers, n_neurons, dropout, norm, st = params_from_file_name(folder)
            print(f"Results for {n_layers} layers, {n_neurons} {neuron} neurons on {dataset} (dropout = {dropout}, norm = {norm}, {st}x repeated inputs)")
        else:
            print(f"Results for {folder}")

        validation_results, test_results = {"acc": [], "fr": [], "balance": []}, {"acc": [], "fr": [], "balance": []}
        skip=False
        for j,metric in enumerate(["acc", "fr", "balance"]):
            for trial_folder in os.walk(folder):
                if "results.pth" in trial_folder[2]:
                    validation_data = torch.load(trial_folder[0]+"/results.pth")["validation_"+metric+"s"]
                    test_data = torch.load(trial_folder[0]+"/results.pth")["test_"+metric]
                    if metric=="fr":
                        validation_data=np.array(validation_data).tolist()
                        #test_data=test_data

                    validation_results[metric].append(torch.tensor(validation_data))
                    test_results[metric].append(test_data)
                
            if len(validation_results["acc"]) == 0:
                skip=True
                continue

            validation_results[metric] = torch.stack(validation_results[metric])
            x = list(range(0,validation_results[metric].shape[1]))
            y_mean = validation_results[metric].mean(axis=0)
            y_ci = validation_results[metric].std(axis=0) #1.96 * np.std(results["score"], axis=0)/np.sqrt(len(x))
            axs[j].plot(x, y_mean, color=colors[i%8], label=folder)
            #plt.plot(x, results["train_score"].mean(axis=0), color=colors[i%8], alpha=.1)
            axs[j].fill_between(x, (y_mean-y_ci), (y_mean+y_ci), color=colors[i%8], alpha=.1)
            axs[j].set_ylabel(metric)
            #axs[j].set_title(metric, y=0.5)

        if skip:
            print("Skipping",folder,"because it contains no results!")
            continue

        validation_accs, test_accs = validation_results["acc"], test_results["acc"]
        print(f"(1) Highest validation accuracy (total): {validation_accs.max()*100:.2f}%")
        print(f"(2) Highest validation accuracy (avg over trial): {validation_accs.mean(axis=0).max()*100:.2f}%")
        print(f"(3) Average validation accuracy over last 5 epochs (avg over trial): {validation_accs.mean(axis=0)[-5:].mean()*100:.2f}%")
        print(f"(4) Test accuracy (trial with (1)): {test_accs[validation_accs.max(dim=1)[0].argmax()]*100:.2f}%")
        print("")

    plt.xlabel('Epoch')
    #plt.ylabel('Validation Accuracies')
    plt.legend()
    if SAVE:
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_FOLDER+"/validation_accs.png")
    if PLOT:
        plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot script')
    parser.add_argument('path', type=str, help='Either a folder containing a results.pth or a parent folder of one')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--save', action='store_true', help='Save plots')
    parser.add_argument('--score', default='validation_accs', help='Values to plot on x-axis (only used in some functions; can be anything from results.pth)')
    parser.add_argument('--function', default='', help='Plot function to call')
    parser.add_argument('--new-name', action='store_true', help='Use new name for plots')
    args = parser.parse_args()

    PLOT=args.plot
    SAVE=args.save
    SCORE=args.score

    if args.function != "":
        locals()[args.function]()
    else:
        plot_results(args.path)