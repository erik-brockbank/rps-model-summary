#!/usr/bin/env python
# coding: utf-8

# # `RL_model_utils.py`

# In[1]:


"""
Library for general purpose functions related to data processing in RL model
"""

import os
import pandas as pd
import numpy as np


###############
### GLOBALS ###
###############

DEFAULT_FILE = 'rps_v2_clean.csv'
TRANSITIONS = ["up", "down", "stay"]
OUTCOMES = ["win", "tie", "loss"]
N_ROUNDS = 300




#######################
### Data processing ###
#######################

def read_rps_data(fp = DEFAULT_FILE):
    """
    Read in the data! Yay!
    """
    return pd.read_csv(fp)


def add_col(df, col_names, value = 0):
    """
    Assign a new column with values initialized as `value`
    col_names = list of new col names
    """
    for name in col_names:
        df[name] = value
    return df


def drop_bot_rows(df):
    """
    Filter out human rows from data frame
    """
    df = df[df['is_bot'] != 1]
    return df


def separate_df(df):
    """
    Create a list of sub-datafames with unque game ids
    """
    experiments = []
    for game in df['game_id'].unique():
        e = df.loc[df['game_id'] == game]
        e = e.reset_index(drop=True)
        experiments.append(e)
    return experiments



############################################
### Evaluate transitions, outcomes, etc. ###
############################################

def evaluate_outcome(player_move, opponent_move):
    """
    Determine outcome based on player move and opponent move
    TODO check that both moves are in OUTCOME_LOOKUP
    """
    OUTCOME_LOOKUP = np.transpose(
        pd.DataFrame.from_dict({
            'rock':     ['tie', 'lose', 'win'],
            'paper':    ['win', 'tie', 'lose'],
            'scissors': ['lose', 'win', 'tie']
        }, orient='index', columns=['rock', 'paper', 'scissors'])
    )

    return OUTCOME_LOOKUP[player_move][opponent_move]

