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
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

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
    
def plot_results():
    # results: ['train_accs', 'train_frs', 'validation_accs', 'validation_frs', 'test_acc', 'test_fr', 'best_acc', 'best_epoch']

    # folders = [os.path.join(DATA_FOLDER_HOME+"/results/dvs_beep", file) for file in os.listdir(DATA_FOLDER_HOME+"/results/dvs_beep")]
    # folders += [os.path.join(DATA_FOLDER_HOME+"/results/dvs_eprop", file) for file in os.listdir(DATA_FOLDER_HOME+"/results/dvs_eprop")]
    # folders += [DATA_FOLDER_HOME+"/results/optuna_dvs_eprop_HPO/id=356756794"]
    #folders = [os.path.join(DATA_FOLDER_HO+"/results/dvs_eprop", file) for file in os.listdir(DATA_FOLDER_HO+"/results/dvs_beep")]
    #folders = [DATA_FOLDER_HO+"/exp"]
    folders = [os.path.join(DATA_FOLDER_HOME+"/exp", file) for file in os.listdir(DATA_FOLDER_HOME+"/exp")]
    colors  = [BLUE,RED,GREEN,YELLOW,VIOLET, DARKRED, DARKBLUE, GREY]

    for i,folder in enumerate(folders):
        #print(f"Analyzing {folder}... ")

        if not os.path.isdir(folder):
            print("Skipping",folder,"because not a directory")
            continue
        
        dataset, neuron, n_layers, n_neurons, dropout, norm, st = params_from_file_name(folder)
        print(f"Results for {n_layers} layers, {n_neurons} {neuron} neurons on {dataset} (dropout = {dropout}, norm = {norm}, {st}x repeated inputs)")

        validation_accs = []
        test_accs = []
        for trial_folder in os.walk(folder):
            if "results.pth" in trial_folder[2]:
                validation_accs.append(torch.tensor(torch.load(trial_folder[0]+"/results.pth")["validation_accs"]))
                test_accs.append(torch.load(trial_folder[0]+"/results.pth")['test_acc'])
            
        if len(validation_accs) == 0:
            print("Skipping",folder,"because it contains no results!")
            continue

        validation_accs = torch.stack(validation_accs)
        x = list(range(0,validation_accs.shape[1]))
        y_mean = validation_accs.mean(axis=0)
        y_ci = validation_accs.std(axis=0) #1.96 * np.std(results["score"], axis=0)/np.sqrt(len(x))
        plt.plot(x, y_mean, color=colors[i%8], label=folder)
        #plt.plot(x, results["train_score"].mean(axis=0), color=colors[i%8], alpha=.1)
        plt.fill_between(x, (y_mean-y_ci), (y_mean+y_ci), color=colors[i%8], alpha=.1)

        print(f"(1) Highest validation accuracy (total): {validation_accs.max()*100:.2f}%")
        print(f"(2) Highest validation accuracy (avg over trial): {validation_accs.mean(axis=0).max()*100:.2f}%")
        print(f"(3) Average validation accuracy over last 5 epochs (avg over trial): {validation_accs.mean(axis=0)[-5:].mean()*100:.2f}%")
        print(f"(4) Test accuracy (trial with (1)): {test_accs[validation_accs.max(dim=1)[0].argmax()]*100:.2f}%")
        print("")

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracies')
    plt.legend()
    if SAVE:
        plt.savefig(OUTPUT_FOLDER+"/validation_accs.png")
    if PLOT:
        plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot script')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--save', action='store_true', help='Save plots')
    parser.add_argument('--score', default='validation_accs', help='Values to plot on x-axis (only used in some functions; can be anything from results.pth)')
    parser.add_argument('--function', default='', help='Plot function to call')
    args = parser.parse_args()

    PLOT=args.plot
    SAVE=args.save
    SCORE=args.score

    if args.function != "":
        locals()[args.function]()
    else:
        plot_results()