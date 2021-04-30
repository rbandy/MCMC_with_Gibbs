#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams.update({'font.size': 18})
import seaborn as sns
import math
from statsmodels.graphics.tsaplots import plot_acf
import os

def remove_from_chain(thetas, burnin, lag):
    """
    remove [0, burnin) items from the beginning of the chain
    only keep every lag-th item
    changes dimensions of thetas-- new_thetas rows: rate(s), cols: steps
    """
    if burnin > 0:
        thetas = np.delete(thetas, np.arange(burnin), 0)
    # discard steps according to lag
    new_thetas = []
    for i in range(len(thetas[0])):
        ki = thetas[:,i]
        if lag > 1:
            ki = np.asarray(ki[::lag])
        new_thetas = np.append(new_thetas,ki)
    return new_thetas.reshape( ( len(thetas[0]), -1) )

def graph_chain(thetas, graph_endings, burnin, lag, n_s):

    # chain
    print(burnin)
    print("reduced size", n_s)
    chain_type = "(raw chain)"
    #print("tot", thetas)
    full_chain = np.copy(thetas)
    full_chain = remove_from_chain(full_chain, burnin, 1)
    thetas = remove_from_chain(thetas, burnin, lag)
    print(len(thetas[0]))
    if lag > 1:
        chain_type = "(filtered chain)"

    cols = n_s
    rows = 1

    fig, axs = plt.subplots(rows, cols,figsize=(5*cols, 5))
    if rows > 1:
        fig, axs = plt.subplots(rows, cols,figsize=(3*cols, 3*rows))
    
    # chain
    for i in range(0, len(thetas)):
        r = i // cols
        c = i % cols
        ki = thetas[i]
        step = np.arange(len(ki))
        if rows == 1:
            axs[c].plot(step, ki)
            axs[c].set_xlabel("Number of Positions")
            axs[c].set_ylabel("$\\delta_{%s%s}$"%(0, c+1))
        elif cols == 1:
            axs[r].plot(step, ki)
            axs[r].set_xlabel("Number of Positions")
            axs[r].set_ylabel("$\\delta_{%s%s}$"%(r+1, c+1))
        else:
            axs[r,c].plot(step, ki) 
            axs[r,c].set_xlabel("Number of Positions")
            axs[r,c].set_ylabel("$\\delta_{%s%s}$"%(r+1, c+1))
    fig.suptitle("MCMC Chain Positions " + chain_type)  
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("graphs/chain_" + graph_endings)
    # plt.show()
    plt.close()

    # kde
    fig, axs = plt.subplots(rows, cols,figsize=(5*cols, 5))
    if rows > 1:
        fig, axs = plt.subplots(rows, cols,figsize=(3*cols, 3*rows))
    for i in range(0, len(thetas)):
        r = i // cols
        c = i % cols
        ki = thetas[i]
        step = np.arange(len(ki))
        if rows == 1:
            sns.kdeplot(ki, ax=axs[c])
            axs[c].set_xlabel("$\\delta_{%s%s}$"%(0, c+1))
            axs[c].set_ylabel("KDE")
        else:
            sns.kdeplot(ki, ax=axs[r, c])
            axs[r,c].set_xlabel("$\\delta_{%s%s}$"%(r+1, c+1))
            axs[r,c].set_ylabel("KDE")
    fig.suptitle("Parameter Kernel Density Estimation " + chain_type)  
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.show()
    plt.savefig("graphs/kde_" + graph_endings)
    plt.close()

def graph_hyper(thetas, graph_endings, burnin, lag, n_s, var_name):
    cols = n_s
    rows=1
    fig, axs = plt.subplots(rows, cols,figsize=(5*cols, 5))
    
    # chain
    for i in range(0, len(thetas)):
        r = i // cols
        c = i % cols
        ki = thetas[i]
        step = np.arange(len(ki))
        axs[c].plot(step, ki)
        axs[c].set_xlabel("Number of Positions")
        axs[c].set_ylabel("${%s}_{%s%s}$"%(var_name, 0, c+1))
    fig.suptitle("MCMC Chain Positions")  
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("graphs/chain_" + graph_endings)
    # plt.show()
    plt.close()

    # histo
    fig, axs = plt.subplots(rows, cols,figsize=(5*cols, 5))
    for i in range(0, len(thetas)):
        r = i // cols
        c = i % cols
        ki = thetas[i]
        step = np.arange(len(ki))
        # sns.kdeplot(ki, ax=axs[c])
        sns.histplot(data=ki, ax=axs[c], kde=True, stat="probability")
        axs[c].set_xlabel("$\\sigma^2_{%s%s}$"%(0, c+1))
        # axs[c].set_ylabel("Kernel Density Estimation")
    fig.suptitle("Marginal Distributions of $\\boldsymbol{{%s}_0}$"%(var_name))  
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.show()
    plt.savefig("graphs/kde_" + graph_endings)
    plt.close()

def graph_sip(chain_file, d, graph_endings, burnin=0, lag=1, n_s=4, hyper_file=None):
    tot = np.loadtxt(chain_file) #.view(complex)
    if d == 1:
        tot = np.array([tot])
        tot = np.reshape(tot, (-1, 1))
    graph_chain(tot, graph_endings, burnin, lag, n_s)

    if hyper_file is not None:
        tot = np.loadtxt(hyper_file) #.view(complex)
        if d == 1:
            tot = np.array([tot])
            tot = np.reshape(tot, (-1, 1))
        tot = np.array(remove_from_chain(tot, burnin, lag))
        mot = tot[0:n_s, :]
        sot = tot[n_s:, :]
        var_name = ["\\mu", "\\sigma^2"]
        ending = "sigma_" + graph_endings
        graph_hyper(sot, ending, burnin, lag, n_s, var_name[1])
        ending = "mu_" + graph_endings
        graph_hyper(mot, ending, burnin, lag, n_s, var_name[0])

if __name__ == "__main__":
    dataFile = "inputs/info.txt"
    info = np.loadtxt(dataFile,comments="%")
    n_S = int(info[0])
    n_s = int(info[1])
    n_phis_cal = int(info[2])
    n_phis_val = int(info[3])
    n_times = int(info[4])
    obs_error = float(info[5])
    inad_type = int(info[6])

    chain_file = "outputs/sip_raw_chain2.dat"
    hyper_file = "outputs/sip_hyper_raw_chain2.dat"
    d = n_s
    burnin = 10000
    lag = 20
    stub = "case2_" '%s' '-s' '%s' '-phi' '%s' %(n_S,n_s,n_phis_cal+n_phis_val)
    graph_endings = stub + ".png"
    graph_sip(chain_file, d, graph_endings, burnin, lag, n_s, hyper_file)