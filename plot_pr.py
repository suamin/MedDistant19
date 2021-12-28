#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copied from: https://raw.githubusercontent.com/fenchri/dsre-vae/main/src/helpers/plot_pr.py

"""
Created on 13/09/2020

author: fenia
"""

import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd
import random
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import json
import pickle as pkl
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from matplotlib.colors import ListedColormap


def plot_pr(dataset):
    sns.set(style="whitegrid")
    sns.set_context("paper")
    fig = plt.figure()

    c = [u'#4878d0', u'#ee854a', u'#6acc64', u'#d65f5f', u'#956cb4', u'#8c613c', u'#dc7ec0', u'#797979', u'#d5bb67', u'#82c6e2']
    if dataset == 'nyt10':
        names = ['pcnnatt', 'jointnre', 'reside', 'intra-inter', 'distre', 'baseline', 'vae', 'vae_prior', 'bert_sent_avg']
        color = ['#797979', '#8c613c', '#dc7ec0', '#d5bb67', '#6acc64', '#ee854a', '#4878d0', '#d65f5f', '#ffd500']
        marker = ['d', 's', '^', '*', 'v', '<', '>', 'o', 'x']
    elif dataset == 'nyt10_570k':
        names = ['pcnnatt', 'reside', 'intra-inter', 'baseline', 'vae', 'vae_prior']
        color = ['#797979', '#dc7ec0', '#d5bb67', '#ee854a', '#4878d0', '#d65f5f']
        marker = ['s', '^', '*', 'v', '<', '>', 'o']
    elif dataset == 'med_distant':
        names = ['bert_bag_avg',]
        color = ['#797979',]
        marker = ['s',]
    else:
        names = ['baseline', 'vae', 'vae_prior', 'bert_sent_avg']
        color = ['#ee854a', '#4878d0', '#d65f5f', '#ffd500']
        marker = ['<', '>', 'o', 'x']

    for i, name in enumerate(names):

        if dataset == 'nyt10':
            path = os.path.join('nyt10', '520K', name)
        elif dataset == 'nyt10_570k':
            path = os.path.join('nyt10', '570K', name)
        elif dataset == 'med_distant':
            path = os.path.join('med_distant', name)
        else:
            path = os.path.join('wikidistant', name)

        if name in ['distre', 'reside', 'intra-inter', 'jointnre']:
            prec = np.load(os.path.join('../pr_curves/', path, 'precision.npy'))
            rec = np.load(os.path.join('../pr_curves/', path, 'recall.npy'))

        elif name in ['pcnnatt']:
            points = np.load(os.path.join('../pr_curves/', path, 'nyt10_test_pr.npz'))
            prec = points['precision']
            rec = points['recall']

        elif name in ['baseline', 'vae', 'vae_prior']:
            points = np.load(os.path.join('../pr_curves/', path, 'test_pr.npz'))
            prec = points['precision']
            rec = points['recall']

        elif name in ['bert_sent_avg']:
            with open(os.path.join('../pr_curves/', path, 'pr_metrics.pkl'), "rb") as rf:
                metrics = pkl.load(rf)
            prec = metrics['np_prec']
            rec = metrics['np_rec']

        elif name in ['bert_bag_avg']:
            prec = np.load(os.path.join('pr_curves/', path, 'precision.npy'))
            # prec = prec[:9000]
            rec = np.load(os.path.join('pr_curves/', path, 'recall.npy'))
            # rec = rec[:9000]
        else:
            rec = 0
            prec = 0

        area = auc(rec, prec)
        print(f'Name: {name}, Area: {area}')
        if name == 'distre':
            name = 'DISTRE'
        elif name == 'pcnnatt':
            name = 'PCNN-ATT'
        elif name == 'reside':
            name = 'RESIDE'
        elif name == 'intra-inter':
            name = 'Intra-Inter'
        elif name == 'jointnre':
            name = 'JointNRE'
        elif name == 'bert_sent_avg':
            name = 'BERT+sent+AVG'
        elif name == 'bert_bag_avg':
            name = 'BERT+bag+AVG'

        if name == 'vae':
            plt.plot(rec, prec, label=r'$\mathcal{N}(0, I)$', lw=1, marker=marker[i], color=color[i], markevery=0.2, ms=6)
        elif name == 'vae_prior':
            plt.plot(rec, prec, label=r'$\mathcal{N}(\mu, I)$', lw=1, marker=marker[i], color=color[i], markevery=0.2, ms=6)
        else:
            plt.plot(rec, prec, label=name, lw=1, marker=marker[i], color=color[i], markevery=0.2, ms=6)

    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.7])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc="upper right", prop={'size': 12})

    plot_path = f'plots/pr_curves_{dataset}'
    fig.savefig(plot_path+'.png', bbox_inches='tight')
    fig.savefig(plot_path+'.pdf', bbox_inches='tight')
    print('Precision-Recall plot saved at: {}'.format(plot_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['nyt10', 'nyt10_570k', 'wikidistant', 'med_distant'])
    args = parser.parse_args()
    plot_pr(args.dataset)
