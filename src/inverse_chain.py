#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
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
    # labels = ["$\\delta_{11}$", "$\\delta_{12}$", "$\\delta_{13}$", "$\\delta_{14}$",
    #           "$\\delta_{21}$", "$\\delta_{22}$", "$\\delta_{23}$", "$\\delta_{24}$",
    #           "$\\delta_{31}$", "$\\delta_{23}$", "$\\delta_{33}$", "$\\delta_{34}$",
    #           "$\\delta_{41}$", "$\\delta_{42}$", "$\\delta_{43}$", "$\\delta_{44}$"]

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
    # cols = 2
    rows = n_s
    #(len(labels) // cols)
    # rows = 

    fig, axs = plt.subplots(rows, cols,figsize=(5*cols, 5))
    if rows > 1:
        fig, axs = plt.subplots(rows, cols,figsize=(3*cols, 3*rows))
    
    # chain
    for i in range(0, len(thetas)):
        # print("curr val:",i)
        r = i // cols
        c = i % cols
        ki = thetas[i]
        step = np.arange(len(ki))
        if rows == 1 and cols == 1:
            plt.plot(step, ki) # , label=labels[i])
            plt.xlabel("Number of Positions")
            #axs[c].set_ylabel("Rate Coefficient Values")
            # plt.ylabel("$\\delta_{%s%s}$"%(r+1, c+1))
            plt.ylabel("$a_{%s%s}$"%(1, 2))
        elif rows == 1:
            axs[c].plot(step, ki) # , label=labels[i])
            #axs[c].legend()
            axs[c].set_xlabel("Number of Positions")
            #axs[c].set_ylabel("Rate Coefficient Values")
            axs[c].set_ylabel("$\\delta_{%s%s}$"%(c, r+1))
        elif cols == 1:
            axs[r].plot(step, ki)
            axs[r].set_xlabel("Number of Positions")
            axs[r].set_ylabel("$\\delta_{%s%s}$"%(r+1, c+1))
        else:
            axs[r,c].plot(step, ki) #, label=labels[i])
            #axs[r,c].legend()
            axs[r,c].set_xlabel("Number of Positions")
            #axs[r,c].set_ylabel("Rate Coefficient Values")
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
        if rows == 1 and cols == 1:
            sns.kdeplot(ki)
            #axs[c].legend()
            # plt.xlabel("$\\delta_{%s%s}$"%(r+1, c+1))
            plt.ylabel("$a_{%s%s}$"%(1, 2))
            plt.ylabel("KDE")
        elif rows == 1:
            sns.kdeplot(ki, ax=axs[c])
            #axs[c].legend()
            axs[c].set_xlabel("$\\delta_{%s%s}$"%(c, r+1))
            axs[c].set_ylabel("KDE")
        # elif cols == 1:
        #     sns.kdeplot(ki, label=labels[i], ax=axs[r])
        #     #axs[c].legend()
        #     axs[r].set_xlabel("$\\delta_{%s%s}$"%(r+1, c+1))
        #     axs[r].set_ylabel("KDE")
        else:
            sns.kdeplot(ki, ax=axs[r, c])
            #axs[r,c].legend()
            axs[r,c].set_xlabel("$\\delta_{%s%s}$"%(r+1, c+1))
            axs[r,c].set_ylabel("KDE")
    fig.suptitle("Parameter Kernel Density Estimation " + chain_type)  
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.show()
    plt.savefig("graphs/kde_" + graph_endings)
    plt.close()

def graph_sip(chain_file, d, graph_endings, burnin=0, lag=1, n_s=4):
    tot = np.loadtxt(chain_file) #.view(complex)
    if d == 1:
        tot = np.array([tot])
        tot = np.reshape(tot, (-1, 1))
    graph_chain(tot, graph_endings, burnin, lag, n_s)

if __name__ == "__main__":
    # os.system("./post_proc.sh")

    # dataFile = "../inputs/info.txt"
    # info = np.loadtxt(dataFile,comments="%")
    n_S = 2
    n_s = 1
    n_phis_cal = 1
    n_phis_val = 0
    n_times = 11
    var = 0.1
    inad_type = 0

    chain_file = "outputs/sip_raw_chain.dat"
    d = n_s * 2
    if inad_type == 5:
        d = n_s**2
    elif inad_type == 0:
        d = n_s
    burnin = 0
    lag = 1
    stub = "a_12" '%s' '-s' '%s' '-phi' '%s' %(n_S,n_s,n_phis_cal+n_phis_val)
    graph_endings = stub + ".png"
    graph_sip(chain_file, d, graph_endings, burnin, lag, n_s)