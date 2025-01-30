import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean, correlation #cosine, cityblock
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sparch.dataloaders.spiking_datasets import CueAccumulationDataset
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
DARKGREEN = "#46663C"
BLUE = "#5E7DAF"
LIGHTBLUE ="#8FA3C7"
LIGHTBLUE2 = "#16b0d9"
DARKBLUE = "#3C5E8A"
DARKRED = "#A84646"
VIOLET = "#886A9B"
GREY = "#636363"
LIGHTGREY = "#c9c5c5"
TURQUOISE = "#4CB5AE"
ORANGE = "#FF9F45"
BLACK = "#000000"
PLOT=True
SCORE=""
OUTPUT="paper_plots"
colors  = [BLUE,YELLOW,RED,GREEN,VIOLET, DARKRED, DARKBLUE, GREY, BLACK]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

def percentage_formatter(x, pos):
    return f"{x * 100:.0f}" 

def recurse_dir(path):
    folders = []
    for (dirpath, dirs, files) in os.walk(path):
        # potential result folder if 
        # - contains run0 or trial_0 folders
        # - contains result.pth directly
        if "run0" in dirs or "trial_0" in dirs or ("results.pth" in files and "run" not in dirpath and "trial_" not in dirpath):
            folders.append(dirpath)

    return folders

def plot_unbalanced():
#--n 40 --data cue --w-init rand --seed 5 --alpha 0.999 --auto-encoder --track-balance --plot-neuron 5 --plot --plot-input-raster --repeat 1 --sigma-s 0 --sigma-v 0 --repeat 10 --plot-dim 1 --save balanced.png
#--n 40 --data cue --w-init rand --seed 5 --alpha 0.999 --auto-encoder --track-balance --plot-neuron 5 --plot --plot-input-raster --repeat 1 --sigma-s 0 --repeat 1 --sigma-v 0.1 --plot-dim 1 --save unbalanced.png
    # define constants for leaky integrator example & unpack args for convenience
    N = 40
    scale = 200
    dataset = CueAccumulationDataset(0, False)
    np.random.seed(0)
    
    def sim(substeps, sigma_v, alpha):
        c = scale*dataset[0].cpu().numpy()
        c = c.repeat(substeps, axis=0)[:, 30:40]
        t = c.shape[0]
        J = c.shape[1]
        # solve LDS with forward Euler and exact solution
        x = np.zeros([t, J])
        for k in range(t-1):
            x[k+1] = alpha*x[k] + (1-alpha)*c[k]  # explicit euler

        # set other weights
        w_out = np.random.binomial(1, 0.7, size=(J,N)) * np.random.uniform(-(1-0.999)/0.001, (1-0.999)/0.001, size=(J, N))
        w_in   = w_out.T.copy()   # NxJ
        w_fast = w_out.T @ w_out  # NxN

        v_thresh = 0.5*(np.diagonal(w_fast)) # np.linalg.norm(w_out,axis=0)
        v_rest = np.full(N, 0, dtype=float)

        w_fast = -w_fast

        w_fast /= (1-0.999)
        w_out /= (1-0.999)

        w_fast_neg = np.where(w_fast<0, w_fast, 0)
        w_fast_pos = np.where(w_fast>=0, w_fast, 0)
        w_in_neg = np.where(w_in<0, w_in, 0)
        w_in_pos = np.where(w_in>=0, w_in, 0)

        # solve EBN with forward euler
        x_snn = np.zeros([t, J])
        v     = np.full([t, N], v_rest)
        r     = np.zeros([t, N])
        o     = np.zeros([t, N])
        i_fast= np.zeros([t, N])
        i_in  = np.zeros([t, N])

        i_inh = np.zeros([t, N])
        i_exc = np.zeros([t, N])
        for k in range(t-1):
            i_fast_inh = np.matmul(w_fast_neg, o[k])
            i_fast_exc = np.matmul(w_fast_pos, o[k])
            i_in_inh = np.matmul(w_in_neg, np.where(c[k]>=0, c[k], 0))  # c can be negative, so we need to use np.where for it to make sure we only use negative weights
            i_in_inh += np.matmul(w_in_pos, np.where(c[k]<0, c[k], 0))
            i_in_exc = np.matmul(w_in_neg, np.where(c[k]<0, c[k], 0))
            i_in_exc += np.matmul(w_in_pos, np.where(c[k]>=0, c[k], 0))
            i_inh[k] = i_fast_inh + i_in_inh
            i_exc[k] = i_fast_exc + i_in_exc

            i_fast[k] = i_fast_exc + i_fast_inh
            i_in[k]   = i_in_exc + i_in_inh

            # update membrane voltage
            v[k+1] = alpha * v[k] + (1-alpha)*(i_in[k] + i_fast[k]) + sigma_v * np.random.randn(*v[k].shape)

            # update rate
            r[k+1] = alpha * r[k] + o[k]

            # update output
            x_snn[k+1] = alpha * x_snn[k] + (1-alpha)*np.matmul(w_out,o[k]) #np.matmul(w_out, r[k+1])

            # spikes
            spike_ids = np.asarray(np.argwhere(v[k+1] > v_thresh))
            if len(spike_ids) > 0:
                spike_id = np.random.choice(spike_ids[:, 0])  # spike_ids.shape = (Nspikes, 1) -> squeeze away second dimension (cant use np.squeeze() for arrays for (1,1) though)
                o[k+1][spike_id] = 1

        return c, x, x_snn, o, i_exc, i_inh

    fig, axs = plt.subplots(4, 2, sharex=False, gridspec_kw={'height_ratios': [1, 2, 3, 2]}, figsize=(10,7))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    for i in range(0,2):
        if i==0:
            alpha = 0.99
            substeps = 1
            noise = 0.1
            t = 2250*substeps
            c, x, x_snn, o, i_exc, i_inh = sim(substeps, noise, alpha)
        else:
            alpha = 0.999
            substeps = 10
            noise = 0
            t = 2250*substeps
            c, x, x_snn, o, i_exc, i_inh = sim(substeps, noise, alpha)

        # define colors
        RED = "#D17171"
        YELLOW = "#F3A451"
        GREEN = "#7B9965"
        BLUE = "#5E7DAF"
        DARKBLUE = "#3C5E8A"
        DARKRED = "#A84646"
        VIOLET = "#886A9B"
        GREY = "#636363"
        BLACK = "#000000"

        # create plots
        t_max = t
        t = list(range(0,t_max))

        # plot inputs 
        spikes = np.argwhere(c>0)
        x_axis = spikes[:,0] # x-axis: spike times
        y_axis = spikes[:,1] # y-axis: spiking neuron ids
        colors = len(x_axis)*[BLUE]
        axs[0][i].scatter(x_axis, y_axis, c=colors, marker = "o", s=10, clip_on=False)

        # plot outputs
        b, a = butter(4, 0.1, btype='low', analog=False)
        x_snn_pl = 0.5*x_snn[:, 0] if i==0 else x_snn[:,0]
        axs[1][i].plot(t, x[:, 0], color=GREY, label="x_0", linestyle="solid", clip_on=True, linewidth=2.0)
        axs[1][i].plot(t, x_snn_pl, color=YELLOW, label="x_snn_0", linestyle="solid", clip_on=True, linewidth=2.0)

        spikes = np.argwhere(o>0)
        x_axis = spikes[:,0] # x-axis: spike times
        y_axis = spikes[:,1]# y-axis: spiking neuron ids
        colors = len(spikes[:,0])*[BLUE]
        axs[2][i].scatter(x_axis, y_axis, c=colors, marker = "o", s=10, clip_on=False)

        b, a = butter(4, 0.1, btype='low', analog=False)
        i_exc_plot = i_exc[:, 5]
        i_inh_plot = i_inh[:, 5]
        i_exc_plot = np.array(filtfilt(b, a, i_exc_plot))
        i_inh_plot = np.array(filtfilt(b, a, i_inh_plot))
        axs[3][i].plot(t, i_exc_plot, color=BLUE, label="i_exc", linewidth=2.0)
        axs[3][i].plot(t, -i_inh_plot, color=RED, label="-i_inh", linewidth=2.0)

        # style
        ## x axes
        for j in range(0,3):
            axs[j][i].set_xticks([])
            axs[j][i].tick_params(axis='x', bottom=False, labelbottom=False)
            axs[j][i].spines['bottom'].set_visible(False)
            axs[j][i].spines['top'].set_visible(False)
            axs[j][i].spines['right'].set_visible(False)
        axs[3][i].set_xticks([0, t_max], ["0", str(int(t_max/substeps))])
        axs[3][i].set_xlim(0, t_max)
        axs[3][i].tick_params(axis='x', length=15, width=2.0, labelsize=15)
        axs[3][i].tick_params(axis='x', which='minor', length=5, width=0.5)
        axs[3][i].spines['bottom'].set_position(('outward', 10))
        axs[3][i].spines['bottom'].set_linewidth(2.0)
        axs[3][i].xaxis.set_label_coords(0.0, -0.3)
        axs[3][i].set_xlabel("Time [ms]", fontsize=15, fontweight='bold')
        axs[3][i].spines['top'].set_visible(False)
        axs[3][i].spines['right'].set_visible(False)

        ## y axes
        if i==0:
            axs[0][i].set_yticks([0, 10])
            axs[0][i].set_ylim(0, 10)
            axs[0][i].tick_params(axis='y', length=15, width=2.0, labelsize=15)
            axs[0][i].tick_params(axis='y', which='minor', length=5, width=0.5)
            axs[0][i].spines['left'].set_position(('outward', 10))
            axs[0][i].spines['left'].set_linewidth(2.0)
            axs[0][i].yaxis.set_minor_locator(AutoMinorLocator(10))
            axs[0][i].yaxis.set_label_coords(-0.1, 0.5)
            #axs[0].set_ylabel("c", fontsize=15, fontweight='bold')

            axs[1][i].set_yticks([0, 8]) #max(max(x[:,0]), max(x_snn[:,0]))])
            axs[1][i].set_ylim(0, 8) #max(max(x[:,0]), max(x_snn[:,0])))
            axs[1][i].tick_params(axis='y', length=15, width=2.0, labelsize=15)
            axs[1][i].tick_params(axis='y', which='minor', length=5, width=0.5)
            axs[1][i].spines['left'].set_position(('outward', 10))
            axs[1][i].spines['left'].set_linewidth(2.0)
            #axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
            axs[1][i].yaxis.set_label_coords(-0.1, 0.5)
            #axs[1].set_ylabel("s₀, ŝ₀", fontsize=15, fontweight='bold')

            axs[2][i].set_yticks([0, 40])
            axs[2][i].set_ylim(0, 40)
            axs[2][i].tick_params(axis='y', length=15, width=2.0, labelsize=15)
            axs[2][i].tick_params(axis='y', which='minor', length=5, width=0.5)
            axs[2][i].spines['left'].set_position(('outward', 10))
            axs[2][i].spines['left'].set_linewidth(2.0)
            axs[2][i].yaxis.set_minor_locator(AutoMinorLocator(40))
            axs[2][i].yaxis.set_label_coords(-0.1, 0.5)
            #axs[2].set_ylabel("o", fontsize=15, fontweight='bold')

            axs[3][i].set_yticks([0, 550])#max(i_exc_plot.max(), -i_inh_plot.max())])
            axs[3][i].set_ylim(0, 550)#max(i_exc_plot.max(), -i_inh_plot.max()))
            axs[3][i].tick_params(axis='y', length=15, width=2.0, labelsize=15)
            axs[3][i].tick_params(axis='y', which='minor', length=5, width=0.5)
            axs[3][i].spines['left'].set_position(('outward', 10))
            axs[3][i].spines['left'].set_linewidth(2.0)
            #axs[3].yaxis.set_minor_locator(AutoMinorLocator(10))
            axs[3][i].yaxis.set_label_coords(-0.1, 0.5)
            #axs[3].set_ylabel("i₀", fontsize=15, fontweight='bold')
        else:
            for j in range(0,4):
                axs[j][i].set_yticks([])
                axs[j][i].tick_params(axis='y', bottom=False, labelbottom=False)
                axs[j][i].spines['left'].set_visible(False)


    Path("paper_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("paper_plots/balanced.pdf", format='pdf', transparent=True)
    plt.savefig("paper_plots/balanced.svg", format='svg', transparent=True)
    plt.savefig("paper_plots/balanced.png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()

def plot_noise():
    quants = list(range(4,13,1))
    gauss = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    folders_quant = ["results/noise_tests_fix2/quant"+str(i) for i in quants]
    folders_quant_ref = ["results/noise_tests_ref/quant"+str(i) for i in quants]
    folders_gauss = ["results/noise_tests_fix2/quant6_adc6_gauss"+str(i) for i in gauss]
    folders_gauss_ref = ["results/noise_tests_ref/quant6_adc6_gauss"+str(i) for i in gauss]

    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    fig.subplots_adjust(hspace=0.3)

    # quant plot
    ## get data
    # x, y, y_ref = [], [], []
    # for i,(folder, folder_ref) in enumerate(zip(folders_quant, folders_quant_ref)):
    #     accs, accs_ref = [], []
    #     for trial_folder in os.walk(folder):
    #         if "results.pth" in trial_folder[2]:
    #             results=torch.load(trial_folder[0]+"/results.pth", weights_only=False)
    #             accs.append(results["test_acc"])
    #     # for trial_folder in os.walk(folder_ref):
    #     #     if "results.pth" in trial_folder[2]:
    #     #         results=torch.load(trial_folder[0]+"/results.pth", weights_only=False)
    #     #         accs_ref.append(results["test_acc"])
    #     x.append(quants[i])
    #     print(np.mean(accs))
    #     y.append(np.mean(accs))
    #     # y_ref.append(np.mean(accs_ref))
            
    # ax[0].plot(x, y_ref, color=BLACK, label="Baseline", linewidth=2.5, linestyle="solid", clip_on=False)
    y     = [0.9158, 0.9142, 0.9119, 0.9023, 0.9119, 0.8623, 0.8207, 0.6701, 0.5600] # from claix: noise_tests_fix2/quantX
    y_ref = [0.9214, 0.9214, 0.8595, 0.7865, 0.7373, 0.6960, 0.7206, 0.6507, 0.5036] # from claix: noise_tests_ref_fix/quantX
    x = list(reversed(quants)) #list(reversed(x))  # highest #bits first
    ax[0].plot(x, y, color=BLUE, label="BSNN", linewidth=2.5, linestyle="solid", clip_on=False)
    ax[0].plot(x, y_ref, color=BLACK, label="Baseline", linewidth=2.5, linestyle="solid", clip_on=False)
    ax[0].axhline(y=0.9143, color=LIGHTBLUE, ls='--', lw=2, label='BSNN w/o quant: 91.43%', clip_on=False)
    ax[0].axhline(y=0.9357, color=GREY, ls='--', lw=2, label='Baseline w/o quant: 93.57%', clip_on=False)
    #plt.vlines(x=[0, 150*7*4, 150*7*4+1050*4, 150*7*4+1050*4+150*4], ymin=0, ymax=500, colors=GREY, ls='--', lw=2, label='vline_multiple - full height', clip_on=False)

    ## x axis
    xlims = [max(quants), min(quants)]
    ax[0].set_xticks([12, 4])
    ax[0].set_xlim(12, 4)
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(len(x)-1))
    ax[0].tick_params(axis='x', length=15, width=2.0, labelsize=15)
    ax[0].tick_params(axis='x', which='minor', length=10, width=0.5)
    ax[0].spines['bottom'].set_position(('outward', 15))
    ax[0].spines['bottom'].set_linewidth(2.0)
    ax[0].xaxis.set_label_coords(0.0, -0.15)
    ax[0].set_xlabel("# Bits", fontsize=15, fontweight='bold')

    ## y axis
    ylims = [0.5, 1]
    ax[0].set_yticks(ylims)
    ax[0].set_ylim(ylims[0], ylims[1])
    ax[0].tick_params(axis='y', length=15, width=2.0, labelsize=15)
    ax[0].tick_params(axis='y', which='minor', length=10, width=0.5)
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[0].spines['left'].set_position(('outward', 15))
    ax[0].spines['left'].set_linewidth(2.0)
    ax[0].yaxis.set_label_coords(-0.1, 0.5)
    ax[0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax[0].set_ylabel("Accuracy [%]", fontsize=15, fontweight='bold')

    ## other axes
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ## legend

    # noise plot
    ## get data
    # x, y, y_ref = [], [], []
    # for i,(folder, folder_ref) in enumerate(zip(folders_gauss, folders_gauss_ref)):
    #     accs, accs_ref = [], []
    #     for trial_folder in os.walk(folder):
    #         if "results.pth" in trial_folder[2]:
    #             results=torch.load(trial_folder[0]+"/results.pth", weights_only=False)
    #             accs.append(results["test_acc"])
    #     # for trial_folder in os.walk(folder_ref):
    #     #     if "results.pth" in trial_folder[2]:
    #     #         results=torch.load(trial_folder[0]+"/results.pth", weights_only=False)
    #     #         accs_ref.append(results["test_acc"])
    #     x.append(gauss[i])
    #     y.append(np.mean(accs))
    #     # y_ref.append(np.mean(accs_ref))

    x       = list(reversed(gauss))
    y6b     = [0.8300, 0.8119, 0.8023, 0.7119, 0.6623, 0.6207, 0.5701, 0.5600] # from claix: noise_tests_fix2/quant6_adc6_gaussX
    y6b_ref = [0.6906, 0.7595, 0.6865, 0.6873, 0.5960, 0.5206, 0.5507, 0.5036] # from claix: noise_tests_ref_fix/quant6_adc6_gaussX   
    y8b     = [0.9001, 0.9333, 0.8923, 0.8919, 0.7998, 0.7575, 0.6311, 0.6600]
    y8b_ref = [0.7110, 0.7002, 0.7365, 0.7101, 0.6920, 0.6139, 0.5204, 0.5129]
    # # ax[1].plot(x, y_ref, color=BLACK, label="Baseline", linewidth=2.5, linestyle="solid", clip_on=False)
    ax[1].plot(x, y6b, color=BLUE, label="BSNN (6b)", linewidth=2.5, linestyle="solid", clip_on=False)
    ax[1].plot(x, y6b_ref, color=BLACK, label="Baseline (6b)", linewidth=2.5, linestyle="solid", clip_on=False)
    ax[1].axhline(y=0.8207, color=LIGHTBLUE, ls='--', lw=2, label='BSNN w/o noise: '+str(100*0.8207)+'%', clip_on=False)
    ax[1].axhline(y=0.7206, color=GREY, ls='--', lw=2, label='Baseline w/o noise: '+str(100*0.7206)+'%', clip_on=False)

    ax[1].plot(x, y8b, color=BLUE, label="BSNN (8b)", linewidth=2.5, linestyle="solid", clip_on=False,  marker='o', markersize=8,)
    ax[1].plot(x, y8b_ref, color=BLACK, label="Baseline (8b)", linewidth=2.5, linestyle="solid", clip_on=False,  marker='o', markersize=8,)
    ax[1].axhline(y=0.9119, color=LIGHTBLUE, ls='--', lw=2, label='BSNN w/o noise: '+str(100*0.9119)+'%', clip_on=False,  marker='o', markersize=8,)
    ax[1].axhline(y=0.7373, color=GREY, ls='--', lw=2, label='Baseline w/o noise: '+str(f"{100*0.7373}:.4f")+'%', clip_on=False,  marker='o', markersize=8,)

    ## x axis
    xlims = [min(gauss), max(gauss)]
    ax[1].set_xscale('log')
    ax[1].set_xticks(gauss)
    ax[1].set_xlim(xlims[0], xlims[1])
    ax[1].tick_params(axis='x', length=15, width=2.0, labelsize=15)
    ax[1].tick_params(axis='x', which='minor', length=10, width=0.5)
    ax[1].spines['bottom'].set_position(('outward', 15))
    ax[1].spines['bottom'].set_linewidth(2.0)
    ax[1].xaxis.set_label_coords(0.0, -0.2)
    ax[1].set_xlabel("σ", fontsize=15, fontweight='bold')

    ## y axis
    ylims = [0.5, 1]
    ax[1].set_yticks(ylims)
    ax[1].set_ylim(ylims[0], ylims[1])
    ax[1].tick_params(axis='y', length=15, width=2.0, labelsize=15)
    ax[1].tick_params(axis='y', which='minor', length=5, width=0.5)
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
    ax[1].spines['left'].set_position(('outward', 15))
    ax[1].spines['left'].set_linewidth(2.0)
    ax[1].yaxis.set_label_coords(-0.1, 0.5)
    ax[1].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax[1].set_ylabel("Accuracy [%]", fontsize=15, fontweight='bold')

    ## other axes
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)    
    
    ## legend
    ax[0].legend(loc='lower left', bbox_to_anchor=(0.0,0.0), fontsize=15, ncol=1)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0.0,0.0), fontsize=15, ncol=2)

    # save and plot
    Path("paper_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("paper_plots/noise.pdf", format='pdf', transparent=True)
    plt.savefig("paper_plots/noise.svg", format='svg', transparent=True)
    plt.savefig("paper_plots/noise.png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()

def plot_balance_fr():
    folders = [
        "results/paper/baseline_multispike", 
        "results/paper/baseline_singlespike", 
        "results/paper/lsm", 
        #"results/paper/train_all",
        #"results/paper/train_tau_out", #x
        #"results/paper/train_tau_rec",
        "results/paper/train_taurec_tauout", #x
        "results/paper/train_win",   #x 
        #"results/paper/train_wrec", 
        "results/paper/train_wrec_win",
        #"results/paper/cuba",
        #"results/paper/cuba_refit",
        #"results/paper/refit",
    ]   
    # folders = [
    #     "results/paper/baseline_multispike", 
    #     "results/paper/baseline_singlespike", 
    #     "results/paper/lsm", 
    #     "results/paper/train_all",
    #     #"results/paper/train_tau_out", 
    #     "results/paper/train_tau_rec", 
    #     "results/paper/train_taurec_tauout",
    #     "results/paper/train_win", 
    #     "results/paper/train_wrec", 
    #     "results/paper/train_wrec_win",
    #     "results/paper/cuba",
    #     #"results/paper/cuba_refit",
    #     #"results/paper/refit",
    # ]
    colors = {
        "train_all": BLUE,
        "train_tau_out": BLUE,
        "train_tau_rec": GREEN,
        "train_taurec_tauout": GREEN,
        "train_win": RED,
        "train_wrec": BLUE,
        "train_wrec_win": BLUE,
        "baseline_multispike": BLACK,
        "baseline_singlespike": BLACK,
        "lsm": BLACK,
        "cuba": BLUE,
        "cuba_refit": BLUE,
        "refit": BLUE
    }
    #x_none, y_none, x_bad, x_good, y_bad, y_good = [], [], [], [], [], []
    x_array, y_array, accs_array = [], [], []
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10,6))
    for i,folder in enumerate(folders):
        #print(f"Results for {folder}")
        exp = folder.split("/")[-1]

        for trial_folder in os.walk(folder):
            if "results.pth" in trial_folder[2]:
                #print(f"Loading {trial_folder[0]}/results.pth")
                results=torch.load(trial_folder[0]+"/results.pth", weights_only=False)
                x = [results["test_balance_low"].tolist()] + np.array(results["train_balances_low"]).tolist() + np.array(results["validation_balances_low"]).tolist()
                y = [results["test_fr"].tolist()] + np.array(results["train_frs"]).tolist() +np.array(results["validation_frs"]).tolist()
                accs = [results["test_acc"].tolist()] + np.array(results["train_accs"]).tolist() +np.array(results["validation_accs"]).tolist()

                for i,acc in enumerate(accs):
                    x_array.append(x[i])
                    y_array.append(y[i])
                    accs_array.append(acc)
                    # if acc < 0.6:
                    #     x_none.append(x[i])
                    #     y_none.append(y[i])
                    # else:
                    #     x_good.append(x[i])
                    #     y_good.append(y[i])

                # if results["test_acc"] < 0.9:
                #     print(f"Bad accuracy for {folder}")
    #ax.scatter(x_none, y_none, c=len(x_none)*[BLACK], marker = "o", s=10, clip_on=False)
    #ax.scatter(x_bad, y_bad, c=len(x_bad)*[BLUE], marker = "o", s=10, clip_on=False)
    #ax.scatter(x_good, y_good, c=len(x_good)*[BLUE], marker = "o", s=10, clip_on=False)
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["#D62727", "#2C82C9"])
    sc = ax.scatter(x_array, y_array, c=accs_array, cmap=custom_cmap, marker = "o", s=30, clip_on=False, vmin=0.5, vmax=1.0)
    cbar = plt.colorbar(sc)
    #cbar.set_label("Accuracy [%]", fontsize=15, fontweight='bold')
    #cbar.ax.yaxis
    cbar.ax.set_yticks([0.5, 1.0])
    cbar.ax.set_ylim(0.5, 1.0)
    cbar.ax.tick_params(axis='y', length=15, labelsize=15)
    cbar.ax.tick_params(axis='y', which='minor', length=10, width=0.5)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    # cbar.ax.spines['left'].set_position(('outward', 15))
    # cbar.ax.spines['left'].set_linewidth(2.0)
    cbar.ax.yaxis.set_label_coords(2.5, 0.5)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    cbar.ax.set_ylabel("Accuracy [%]", fontsize=15, fontweight='bold')

    # x axis
    xlims = [0, 1.0]
    ax.set_xticks(xlims)
    ax.set_xlim(xlims[0], xlims[1])
    ax.tick_params(axis='x', length=15, width=2.0, labelsize=15)
    ax.tick_params(axis='x', which='minor', length=5, width=0.5)
    ax.spines['bottom'].set_position(('outward', 15))
    ax.spines['bottom'].set_linewidth(2.0)
    #ax.yaxis.set_minor_locator(AutoMinorLocator(sample_dim/2))
    ax.xaxis.set_label_coords(0.0, -0.075)
    ax.set_xlabel("Balance", fontsize=15, fontweight='bold')

    # y axis
    ylims = [1e-4, 1e-1]
    ylims_str = ["0.1", "100"]
    ax.set_yscale('log')
    ax.set_yticks(ylims, ylims_str)
    ax.set_ylim(ylims[0], ylims[1])
    ax.tick_params(axis='y', length=15, width=2.0, labelsize=15)
    ax.tick_params(axis='y', which='minor', length=5, width=0.5)
    ax.spines['left'].set_position(('outward', 15))
    ax.spines['left'].set_linewidth(2.0)
    # ax.yaxis.set_minor_locator(AutoMinorLocator(sample_dim/2))
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel("Firing Rate [Hz]", fontsize=15, fontweight='bold')

    # ax.fill_between(np.linspace(0, 1, 10), 1e-4, 1e-2, color=GREY, alpha=.1, hatch='//', linewidth=1.5)  # 1e-4 spikes/timestep = 0.1 spikes/second; 1e-2 s/t = 10 spikes/second
    # ax.annotate(
    #     'Typical range in human brain',
    #     xy=(0.8, 1e-2),
    #     xytext=(0.45, 1.2e-2),
    #     #arrowprops=dict(facecolor=GREY, edgecolor=GREY, shrink=0.05),
    #     color = GREY,
    #     fontsize=15,
    #     fontweight='bold'
    # )

    # other axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.margins(x=.01, y=.01)

    Path("paper_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("paper_plots/balance_fr.pdf", format='pdf', transparent=True)
    plt.savefig("paper_plots/balance_fr.svg", format='svg', transparent=True)
    plt.savefig("paper_plots/balance_fr.png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def plot_balance_fr2():
    folders = [
        "results/paper/baseline_multispike", 
        "results/paper/baseline_singlespike", 
        "results/paper/lsm", 
        "results/paper/train_all",
        #"results/paper/train_tau_out", 
        #"results/paper/train_tau_rec", 
        "results/paper/train_taurec_tauout",
        "results/paper/train_win", 
        "results/paper/train_wrec", 
        "results/paper/train_wrec_win",
        #"results/paper/cuba",
        #"results/paper/cuba_refit",
        #"results/paper/refit",
    ]   
    # folders = [
    #     "results/paper/baseline_multispike", 
    #     "results/paper/baseline_singlespike", 
    #     "results/paper/lsm", 
    #     "results/paper/train_all",
    #     #"results/paper/train_tau_out", 
    #     "results/paper/train_tau_rec", 
    #     "results/paper/train_taurec_tauout",
    #     "results/paper/train_win", 
    #     "results/paper/train_wrec", 
    #     "results/paper/train_wrec_win",
    #     "results/paper/cuba",
    #     #"results/paper/cuba_refit",
    #     #"results/paper/refit",
    # ]
    colors = {
        "train_all": BLUE,
        "train_tau_out": BLUE,
        "train_tau_rec": BLUE,
        "train_taurec_tauout": BLUE,
        "train_win": BLUE,
        "train_wrec": BLUE,
        "train_wrec_win": BLUE,
        "baseline_multispike": BLACK,
        "baseline_singlespike": BLACK,
        "lsm": BLUE,
        "cuba": BLUE,
        "cuba_refit": BLUE,
        "refit": BLUE
    }
    x = {
        "train_all": [],
        "train_tau_out": [],
        "train_tau_rec": [],
        "train_taurec_tauout": [],
        "train_win": [],
        "train_wrec": [],
        "train_wrec_win": [],
        "baseline_multispike": [],
        "baseline_singlespike": [],
        "lsm": [],
        "cuba": [],
        "cuba_refit": [],
        "refit": []
    }
    y = {
        "train_all": [],
        "train_tau_out": [],
        "train_tau_rec": [],
        "train_taurec_tauout": [],
        "train_win": [],
        "train_wrec": [],
        "train_wrec_win": [],
        "baseline_multispike": [],
        "baseline_singlespike": [],
        "lsm": [],
        "cuba": [],
        "cuba_refit": [],
        "refit": []
    }

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10,6))
    for i,folder in enumerate(folders):
        #print(f"Results for {folder}")
        exp = folder.split("/")[-1]

        for trial_folder in os.walk(folder):
            if "results.pth" in trial_folder[2]:
                #print(f"Loading {trial_folder[0]}/results.pth")
                results=torch.load(trial_folder[0]+"/results.pth", weights_only=False)
                if results["test_acc"] < 0.9:
                    print(f"Bad accuracy for {folder}")
                    continue
                y[exp].extend([results["test_fr"].tolist()])
                y[exp].extend(np.array(results["train_frs"]).tolist())
                y[exp].extend(np.array(results["validation_frs"]).tolist())

                x[exp].extend([results["test_balance_low"].tolist()])
                x[exp].extend(np.array(results["train_balances_low"]).tolist())
                x[exp].extend(np.array(results["validation_balances_low"]).tolist())


            #exp=folder.split("/")[-1]
            #axs[j].plot(x, y_mean, color=colors[exp], label=legend_labels[exp], linewidth=2.5, linestyle=linestyles[exp])
        # if x[exp][-1] > 0.7:
        #     print(f"Large balance for {folder}")
        # if x[exp][-1] < 0.3:
        #     print(f"Small balance for {folder}")

        # if x[exp][-1] < 0.6 and x[exp][-1] > 0.5 and y[exp][-1] < 0.00025 and y[exp][-1] > 0.00018:
        #     print(f"Weird {folder}") 

    for k,v in x.items():
        ax.scatter(x[k], y[k], c=len(x[k])*[colors[k]], marker = "o", s=10, clip_on=False)

    # x axis
    xlims = [0, 1.0]
    ax.set_xticks(xlims)
    ax.set_xlim(xlims[0], xlims[1])
    ax.tick_params(axis='x', length=15, width=2.0, labelsize=15)
    ax.tick_params(axis='x', which='minor', length=5, width=0.5)
    ax.spines['bottom'].set_position(('outward', 15))
    ax.spines['bottom'].set_linewidth(2.0)
    #ax.yaxis.set_minor_locator(AutoMinorLocator(sample_dim/2))
    ax.xaxis.set_label_coords(0.0, -0.075)
    ax.set_xlabel("Balance", fontsize=15, fontweight='bold')

    # y axis
    ylims = [1e-4, 1e-1]
    ax.set_yscale('log')
    ax.set_yticks(ylims)
    ax.set_ylim(ylims[0], ylims[1])
    ax.tick_params(axis='y', length=15, width=2.0, labelsize=15)
    ax.tick_params(axis='y', which='minor', length=5, width=0.5)
    ax.spines['left'].set_position(('outward', 15))
    ax.spines['left'].set_linewidth(2.0)
    # ax.yaxis.set_minor_locator(AutoMinorLocator(sample_dim/2))
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel("Firing Rate [Hz]", fontsize=15, fontweight='bold')

    # other axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.margins(x=.01, y=.01)

    Path("paper_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("paper_plots/balance_fr.pdf", format='pdf', transparent=True)
    plt.savefig("paper_plots/balance_fr.svg", format='svg', transparent=True)
    plt.savefig("paper_plots/balance_fr.png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()

def plot_boerlin_sample():
    networks = [
        "spikes_local/baseline/plots/epoch10_class0_1.png_spikes.pth", 
        "spikes_local/lsm/plots/epoch1_class0_0.png_spikes.pth",
        "spikes_local/cuba_oldreset/plots/epoch1_class0_0.png_spikes.pth"
    ]

    fig, axs = plt.subplots(1+len(networks)+1, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.3]+[1]*len(networks)}, figsize=(7,10))

    for i,ax in enumerate(axs):
        if i==0:
            print(f"Loading dataset")
            dataset = CueAccumulationDataset(0, False)
            sample = dataset[0].cpu().numpy()
            sample = sample.repeat(4, axis=0)
            sample_time = sample.shape[0]
            sample_dim = sample.shape[1]
        elif i>1:
            print(f"Loading network {networks[i-2]}")
            sample = torch.load(networks[i-2], map_location='cpu', weights_only=False).detach().numpy()
            sample_time = sample.shape[0]
            sample_dim = sample.shape[1]
            if sample_time==2250:
                sample = sample.repeat(4, axis=0)
                sample_time = 2250*4

        if i!=1:
            print(f"Sample time: {sample_time}")
            print(f"Sample dim: {sample_dim}")

            # add data
            spikes = np.argwhere(sample>0)
            x = spikes[:,0] # x-axis: spike times
            y = spikes[:,1] # y-axis: spiking neuron ids
            colors = len(x)*[BLUE]
            ax.scatter(x, y, c=colors, marker = "o", s=8, clip_on=False)

        # x axis
        if ax == axs[-1]:
            ax.set_xticks([0, sample_time], labels=["0", "2250"])
            ax.set_xlim(0, sample_time)
            ax.tick_params(axis='x', length=15, width=2.0, labelsize=15)
            ax.tick_params(axis='x', which='minor', length=5, width=0.5)
            ax.spines['bottom'].set_position(('outward', 15)) 
            ax.spines['bottom'].set_linewidth(2.0)
            #ax.yaxis.set_minor_locator(AutoMinorLocator(sample_dim/2))
            ax.xaxis.set_label_coords(0.5, -0.5)
            ax.set_xlabel("Time [ms]", fontsize=15, fontweight='bold')
        else:
            ax.set_xticks([])
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)

        # y axis
        if ax != axs[1]:
            ax.set_yticks([0, sample_dim/2, sample_dim])
            ax.set_ylim(0, sample_dim)
            ax.tick_params(axis='y', length=15, width=2.0, labelsize=15)
            ax.tick_params(axis='y', which='minor', length=5, width=0.5)
            ax.spines['left'].set_position(('outward', 15)) 
            ax.spines['left'].set_linewidth(2.0)
            ax.yaxis.set_minor_locator(AutoMinorLocator(sample_dim/10))
        else:
            ax.set_yticks([])
            ax.tick_params(axis='y', bottom=False, labelbottom=False)
            ax.spines['left'].set_visible(False)
        #ax.yaxis.set_label_coords(-0.1, 0.5)
        # if i==0:
        #     ax.set_ylabel("Input\nNeurons", fontsize=15, fontweight='bold')
        # else:
        #     ax.set_ylabel("Recurrent\nNeurons", fontsize=15, fontweight='bold')

        # other axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # plot
    plt.vlines(x=[0, 150*7*4, 150*7*4+1050*4, 150*7*4+1050*4+150*4], ymin=0, ymax=500, colors=GREY, ls='--', lw=2, label='vline_multiple - full height', clip_on=False)
    Path("paper_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("paper_plots/cue_example.pdf", format='pdf', transparent=True)
    plt.savefig("paper_plots/cue_example.svg", format='svg', transparent=True)
    plt.savefig("paper_plots/cue_example.png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.close()

def plot_balance_example():
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]}, figsize=(30,15))
    fig.subplots_adjust(hspace=0)
    colors  = [BLUE,RED,GREEN,YELLOW,VIOLET, DARKRED, DARKBLUE, GREY]
    b, a = butter(4, 0.1, btype='low', analog=False)

    t = 1000
    x = list(range(t))
    x      = x[10:t-10]
    
    i_exc0 = np.abs(np.random.randn(t))
    #i_inh0 = np.random.randn(t) #-1.5*i_exc0+np.random.randn(t)*0.5 # *0.8+3
    i_inh0 = -np.abs(np.random.randn(t))
    i_inh0 *= (np.linalg.norm(i_exc0) / np.linalg.norm(i_inh0))

    i_exc0 = np.array(filtfilt(b, a, i_exc0))
    i_inh0 = np.array(filtfilt(b, a, i_inh0))
    i_exc0 = (i_exc0)[10:t-10]
    i_inh0 = (i_inh0)[10:t-10]

    i_exc1 = np.random.randn(t)*0.8+1
    i_inh1 = -i_exc1+np.random.randn(t)*0.08
    i_exc1 = np.array(filtfilt(b, a, i_exc1))
    i_inh1 = np.array(filtfilt(b, a, i_inh1))
    i_exc1 = (i_exc1)[10:t-10]
    i_inh1 = (i_inh1)[10:t-10]

    axs[0].plot(x, i_exc0, color=BLUE, linewidth=6.5)
    axs[0].plot(x, i_inh0, color=RED, linewidth=6.5)
    axs[0].plot(x, i_exc0+i_inh0, color=GREY, linewidth=6.5, linestyle='dashed')
    axs[1].plot(x, i_exc1, color=BLUE, linewidth=6.5)
    axs[1].plot(x, i_inh1, color=RED, linewidth=6.5)
    axs[1].plot(x, i_exc1+i_inh1, color=GREY, linewidth=6.5, linestyle='dashed')

    plt.axis('off')
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    Path("paper_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("paper_plots/balance_example.pdf", format='pdf', transparent=True)
    plt.savefig("paper_plots/balance_example.svg", format='svg', transparent=True)
    plt.savefig("paper_plots/balance_example.png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()

def plot_results_cue():
    folders = [
        "results/paper/baseline_multispike", 
        "results/paper/baseline_singlespike", 
        "results/paper/lsm", 
        "results/paper/train_all",
        #"results/paper/train_tau_out", 
        #"results/paper/train_tau_rec", 
        "results/paper/train_taurec_tauout",
        #"results/paper/train_win", 
        #"results/paper/train_wrec", 
        #"results/paper/train_wrec_win",
        "results/paper/cuba",
        #"results/paper/cuba_refit",
        #"results/paper/refit",
    ]    
    legend_labels = {
        "baseline_multispike": "Baseline", 
        "baseline_singlespike": "Baseline (one spike per timestep)", 
        "lsm": "LSM", 
        "train_all": "Train W & α",
        "train_tau_out": "train_tau_out",
        "train_tau_rec": "train_tau_rec",
        "train_taurec_tauout": "Train α",
        "train_win": "train_win",
        "train_wrec": "train_wrec",
        "train_wrec_win": "train_wrec_win",
        "cuba": "CUBA",
        "cuba_refit": "cuba_refit",
        "refit": "refit",
    }
    metric_labels = {"acc": "Accuracy [%]", "fr": "Firing Rate [Hz]", "balance": "Balance"}
    metric_axs = {0: "acc", 1: "fr", 2: "balance"}
    colors = {
        "train_all": DARKBLUE,
        "train_tau_out": GREY,
        "train_tau_rec": BLUE,
        "train_taurec_tauout": LIGHTBLUE2,
        "train_win": RED,
        "train_wrec": VIOLET,
        "train_wrec_win": DARKBLUE,
        "baseline_multispike": BLACK,
        "baseline_singlespike": BLACK,
        "lsm": ORANGE,
        "cuba": DARKRED,
        "cuba_refit": DARKBLUE,
        "refit": YELLOW
    }
    linestyles = {
        "train_all": "solid",
        "train_tau_out": "solid",
        "train_tau_rec": "solid",
        "train_taurec_tauout": "solid",
        "tauout": "solid",
        "train_win": "solid",
        "train_wrec": "solid",
        "train_wrec_win": "solid",
        "baseline_multispike": "solid",
        "baseline_singlespike": "dotted",
        "lsm": "solid",
        "cuba": "solid",
        "cuba_refit": "solid",
        "refit": "solid"
    }
    exp_name="results_cue_lsm"
    ignore="none"
    ylims = {"acc": [0.5, 1.0], "fr": [1e-4, 1e-1], "balance": [0.0, 0.5, 1.0]}
    ylims_str = {"acc": ["0.5", "1.0"], "fr": ["0.1", "100"], "balance": ["0.0", "0.5", "1.0"]} # fr[spikes/second] = fr[spikes/timestep]/h -> x 1000

    fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]}, figsize=(10,10))
    fig.subplots_adjust(hspace=0.2)
    
    fontsize = 16
    labelsize = 15
    folders = [f for f in folders if ignore not in f]
    for i,folder in enumerate(folders):
        print(f"Results for {folder}")

        validation_results, test_results = {"acc": [], "fr": [], "balance": []}, {"acc": [], "fr": [], "balance": []}
        skip=False

        for j,metric in enumerate(["acc", "fr", "balance"]):
            for trial_folder in os.walk(folder):
                if "results.pth" in trial_folder[2]:
                    print(f"Loading {trial_folder[0]}/results.pth")
                    try:
                        validation_data = torch.load(trial_folder[0]+"/results.pth",weights_only=False)["validation_"+metric+"s"]
                        test_data = torch.load(trial_folder[0]+"/results.pth", weights_only=False)["test_"+metric]
                    except:
                        if metric=="balance":
                            validation_data = torch.load(trial_folder[0]+"/results.pth", weights_only=False)["validation_"+metric+"s_low"]
                            test_data = torch.load(trial_folder[0]+"/results.pth", weights_only=False)["test_"+metric+"_low"]
                    if metric=="fr":
                        validation_data=np.array(validation_data).tolist()
                        #test_data=test_data
                    elif metric=="acc":
                        if test_data > 0.99:
                            continue

                    validation_results[metric].append(torch.tensor(validation_data))
                    test_results[metric].append(test_data)

            if len(validation_results["acc"]) == 0:
                skip=True
                continue

            validation_results[metric] = torch.stack(validation_results[metric])
            x = list(range(1,validation_results[metric].shape[1]+1))
            y_mean = validation_results[metric].mean(axis=0)
            y_ci = validation_results[metric].std(axis=0) #1.96 * np.std(results["score"], axis=0)/np.sqrt(len(x))

            exp=folder.split("/")[-1]
            axs[j].plot(x, y_mean, color=colors[exp], label=legend_labels[exp], linewidth=2.5, linestyle=linestyles[exp], clip_on=False)
            #plt.plot(x, results["train_score"].mean(axis=0), color=colors[i%8], alpha=.1)
            axs[j].fill_between(x, (y_mean-y_ci), (y_mean+y_ci), color=colors[exp], alpha=.1)
            axs[j].set_ylabel(metric_labels[metric], fontsize=fontsize, fontweight='bold')

            if metric=="fr":
                x_fr = x

        if skip:
            print("Skipping",folder,"because it contains no results!")
            continue

        validation_accs, test_accs = validation_results["acc"], test_results["acc"]
        validation_frs, test_fr = validation_results["fr"], test_results["fr"]
        print(f"(1) Highest validation accuracy (total): {validation_accs.max()*100:.2f}%")
        print(f"(2) Highest validation accuracy (avg over trial): {validation_accs.mean(axis=0).max()*100:.2f}%")
        print(f"(3) Average validation accuracy over last 5 epochs (avg over trial): {validation_accs.mean(axis=0)[-5:].mean()*100:.2f}%")
        print(f"(4) Test accuracy (trial with (1)): {test_accs[validation_accs.max(dim=1)[0].argmax()]*100:.2f}%")
        print(f"(5) Average firing rate (total): {validation_frs.mean():.6f}Hz")
        print(f"(6) Test firing rate (trial with (1)): {float(test_fr[0]):.6f}Hz")
        print(f"(7) Average firing rate x ts: {(validation_frs.mean())*2250.0:.6f}Hz")
        print(f"(8) Test firing rate x ts: {float(test_fr[0])*2250.0*4:.6f}Hz")
        print("")

    axs[0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
    axs[0].tick_params(axis='y', which='minor', length=5, width=0.5)
    axs[1].set_yscale('log')
    axs[1].tick_params(axis='y', which='minor', length=5, width=0.5)
    #axs[1].fill_between(x_fr, len(x_fr)*[1e-4], len(x_fr)*[1e-2], color=GREY, alpha=.1, hatch='//', linewidth=1.5)  # 1e-4 spikes/timestep = 0.1 spikes/second; 1e-2 s/t = 10 spikes/second

    axs[2].set_xlabel('Epoch', fontsize=fontsize, fontweight='bold')
    axs[2].xaxis.set_label_coords(0.5, -0.2)

    axs[1].annotate(
        'Typical range in human brain',
        xy=(5, 1e-2),
        xytext=(4, 2e-1),
        arrowprops=dict(facecolor=GREY, edgecolor=GREY, shrink=0.05),
        color = GREY,
        fontsize=15,
        fontweight='bold'
    )

    # line = Line2D([0,0], [1, 1], color=BLACK, linewidth=6, linestyle='-')
    # axs[1].add_line(line)

    # axs[1].annotate(
    #     '-',
    #     xy=(1, 1e-4),
    #     xytext=(1, 1e-2),
    #     arrowprops=dict(facecolor=GREY, edgecolor=GREY, linewidth=8, arrowstyle='-'),
    #     fontsize=15
    # )

    for i,ax in enumerate(axs):
        metric = metric_axs[i]
        x_values, y_values = [], []
        for line in ax.get_lines():
            x_values.extend(line.get_xdata())
            y_values.extend(line.get_ydata())

        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = ylims[metric][0], ylims[metric][-1]

        if i==2:
            ax.set_xticks([x_min, x_max])
            ax.set_xlim(x_min , x_max)
            ax.spines['bottom'].set_position(('outward', 15))
            ax.spines['bottom'].set_linewidth(2.0)
        else:
            ax.set_xticks([])
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)
        
        ax.set_yticks(ylims[metric], ylims_str[metric])
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='both', length=15, width=2.0, labelsize=labelsize)
        ax.spines['left'].set_position(('outward', 15)) 
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.margins(x=.01, y=.01)
        ax.yaxis.set_label_coords(-0.11, 0.5)
    
    
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), fontsize=fontsize, ncol=2)
    Path("paper_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("paper_plots/"+exp_name+".pdf", format='pdf', transparent=True)
    plt.savefig("paper_plots/"+exp_name+".svg", format='svg', transparent=True)
    plt.savefig("paper_plots/"+exp_name+".png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot script')
    parser.add_argument('--function', '-f', default='', help='Plot function to call')
    args = parser.parse_args()

    if args.function != "":
        locals()[args.function]()
    else:
        plot_results_cue()