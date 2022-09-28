#!/usr/bin/env python
# coding: utf-8

# # `RL_model_visualization.py`

# In[1]:


# """
# Library for visualizing model output
# These functions are primarily graphing functions, with some supporting functions as well
# """

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import stats
# import seaborn as sns



# N_ROUNDS = 300
# def groupby_f_data(f_data, colname, bins):
#     """
#     group by filtered data with player outcome and calculate the win percentage
#     colname will be either 'player_outcome' or 'agent_outcome' for plotting human or agent results
#     """
#     modified_f_data = f_data.dropna()
#     labs = [str(int(round(a * (N_ROUNDS / bins), 0))) for a in range(1, bins + 1)]
#     modified_f_data= modified_f_data.assign(bin= pd.cut(modified_f_data.loc[:, ('round_index')], bins, labels = labs))
#     grouped_data = modified_f_data[['bot_strategy', 'player_id','bin', colname]].groupby(
#         ['bot_strategy', 'player_id', 'bin'])[colname].value_counts('count').rename('pct').reset_index()
    
#     return grouped_data


# def win_summary(grouped_data, colname):
#     """
#     filter out the win data and add mean, SD, and SEM
#     colname will be either 'player_outcome' or 'agent_outcome' for plotting human or agent results
#     """
#     win_data = grouped_data[grouped_data[colname] == 'win'].reset_index()
#     win_summary = win_data[['bot_strategy', 'bin', 'pct']].groupby(
#         ['bot_strategy', 'bin'])['pct'].agg(
#             [np.mean, np.std, stats.sem]).reset_index()
    
#     return win_summary


# def plot_win_rates(data):
#     """
#     generate plot displaying win rates against each bot, binned by rounds
#     """
#     sns.set_style(style='white')
#     data['bot_strategy'] = data['bot_strategy'].replace([
#         'prev_move_positive', 'prev_move_negative', 
#         'opponent_prev_move_positive', 'opponent_prev_move_nil',
#         'win_nil_lose_positive', 'win_positive_lose_negative',
#         'outcome_transition_dual_dependency'
#     ],
#     [
#         'Previous move (+)', 'Previous move (-)',
#         'Opponent previous move (+)', 'Opponent previous move (0)',
#         'Win-stay-lose-positive', 'Win-positive-lose-negative',
#         'Outcome-transition dual dependency'
#     ])
    
#     f, ax = plt.subplots(figsize=(15, 10))
#     g = sns.scatterplot(
#         x = "bin", y = "mean", hue = "bot_strategy", 
#         hue_order = [
#             'Previous move (+)', 'Previous move (-)',
#             'Opponent previous move (+)', 'Opponent previous move (0)',
#             'Win-stay-lose-positive', 'Win-positive-lose-negative',
#             'Outcome-transition dual dependency'
#         ],
#         palette="deep", s = 200, ax = ax, data = data)
    
#     plt.errorbar(data.get('bin'), data.get('mean'), yerr = data.get('sem'), 
#         fmt = '.', ecolor='0.5', color='0.5',
#         capsize = 10 , elinewidth = 1, capthick = 1)
#     plt.ylim(0, 1.0)
#     plt.title('Win percentage against bot strategies')
#     plt.xlabel('Trial round')
#     plt.ylabel('Mean win percentage')
#     plt.axhline(y = 1/3, color = 'r', linestyle = '--')
    
#     plt.savefig(os.path.join('img', 'critical_trial_intervention_dist.png'), dpi=300, bbox_inches='tight', transparent=True)
    
#     return


# In[2]:


"""
Library for visualizing model output
These functions are primarily graphing functions, with some supporting functions as well
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from utils import N_ROUNDS


def groupby_f_data(f_data, colname, bins):
    """
    group by filtered data with player outcome and calculate the win percentage
    colname will be either 'player_outcome' or 'agent_outcome' for plotting human or agent results
    """
    modified_f_data = f_data.dropna()
    labs = [str(int(round(a * (N_ROUNDS / bins), 0))) for a in range(1, bins + 1)]
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


def plot_win_rates(data, img_name):
    """
    generate plot displaying win rates against each bot, binned by rounds
    """
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
    
    data['bin'] = data['bin'].replace(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'])
    
    palette = {
      'Opponent previous move (0)':'#A569BD',
      'Opponent previous move (+)':'#8E44AD', 
      'Outcome-transition dual dependency':'#A04000', 
      'Previous move (-)':'#85C1E9', 
      'Previous move (+)':'#2874A6', 
      'Win-stay-lose-positive':'#A9DFBF', 
      'Win-positive-lose-negative':'#229954'
  }
    
    hue_order = ['Previous move (-)','Previous move (+)','Opponent previous move (0)','Opponent previous move (+)',
              'Win-stay-lose-positive','Win-positive-lose-negative','Outcome-transition dual dependency']
    
    f, ax = plt.subplots(figsize=(15, 10))
    g = sns.pointplot(
        x = "bin", y = "pct", hue = "bot_strategy", scale = 2.5,
        palette=palette, s = 400, ax = ax, data = data,hue_order = hue_order)
    
    plt.ylim(0, 1.0)
    ax.set_xticklabels(["30","","","","150","","","","","300"])
    ax.set_yticks([0,0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0])
    ax.set_yticklabels(["0","","","","","0.5","","","","","1"])
    plt.xlabel('Trial round',fontdict={'fontsize':25},fontweight="bold")
    plt.ylabel('Mean win percentage',fontdict={'fontsize':25},fontweight="bold")
    plt.axhline(y = 1/3, color = 'grey', linestyle = '--',linewidth = 5)
    for label in (ax.get_xticklabels()+ax.get_yticklabels()):
        label.set_fontsize(15)
        label.set_fontweight("bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.legend(fontsize = 15)
    sns.despine()
    
#     plt.savefig(os.path.join('img', img_name), dpi=300, bbox_inches='tight', transparent=True)
    
    return g


# In[ ]:




