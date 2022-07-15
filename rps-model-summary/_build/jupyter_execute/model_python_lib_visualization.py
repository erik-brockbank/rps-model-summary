#!/usr/bin/env python
# coding: utf-8

# # `model_visualization.py`
# 
# TODO add quick explanation of this library

# In[1]:


"""
Library for visualizing model output
These functions are primarily graphing functions, with some supporting functions as well
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def groupby_f_data(f_data, colname, bins, rounds):
    """
    group by filtered data with player outcome and calculate the win percentage
    colname will be either 'player_outcome' or 'agent_outcome' for plotting human or agent results
    """
    modified_f_data = f_data.dropna()
    labs = [str(int(round(a * (rounds / bins), 0))) for a in range(1, bins + 1)]
    modified_f_data['bin'] = pd.cut(modified_f_data.loc[:, ('round_index')], bins, labels = labs)
    grouped_data = modified_f_data[['bot_strategy', 'player_id','bin', colname]].groupby(
        ['bot_strategy', 'player_id', 'bin'])[colname].value_counts('count').rename('pct').reset_index()

    return grouped_data


def win_summary(grouped_data, colname):
    """
    filter out the win data and add mean, SD, and SEM
    colname will be either 'player_outcome' or 'agent_outcome' for plotting human or agent results
    """
    win_data = grouped_data[grouped_data[colname] == 'win'].reset_index()
    win_summary = win_data[['bot_strategy', 'bin', 'pct']].groupby(
        ['bot_strategy', 'bin'])['pct'].agg(
            [np.mean, np.std, stats.sem]).reset_index()

    return win_summary


def plot_win_rates(data):
    """
    generate plot displaying win rates against each bot, binned by rounds
    """
    sns.set_style(style='white')
    data['bot_strategy'] = data['bot_strategy'].replace([
        'prev_move_positive', 'prev_move_negative',
        'opponent_prev_move_positive', 'opponent_prev_move_nil',
        'win_nil_lose_positive', 'win_positive_lose_negative',
        'outcome_transition_dual_dependency'
    ],
    [
        'Previous move (+)', 'Previous move (-)',
        'Opponent previous move (+)', 'Opponent previous move (0)',
        'Win-stay-lose-positive', 'Win-positive-lose-negative',
        'Outcome-transition dual dependency'
    ])

    f, ax = plt.subplots(figsize=(15, 10))
    g = sns.scatterplot(
        x = "bin", y = "mean", hue = "bot_strategy",
        hue_order = [
            'Previous move (+)', 'Previous move (-)',
            'Opponent previous move (+)', 'Opponent previous move (0)',
            'Win-stay-lose-positive', 'Win-positive-lose-negative',
            'Outcome-transition dual dependency'
        ],
        palette="deep", s = 200, ax = ax, data = data)

    plt.errorbar(data.get('bin'), data.get('mean'), yerr = data.get('sem'),
        fmt = '.', ecolor='0.5', color='0.5',
        capsize = 10 , elinewidth = 1, capthick = 1)
    plt.ylim(0, 1.0)
    plt.title('Win percentage against bot strategies')
    plt.xlabel('Trial round')
    plt.ylabel('Mean win percentage')
    plt.axhline(y = 1/3, color = 'r', linestyle = '--')

    return g

