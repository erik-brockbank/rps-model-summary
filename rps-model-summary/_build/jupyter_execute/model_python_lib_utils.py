#!/usr/bin/env python
# coding: utf-8

# # `model_utils.py`
# 
# TODO add quick explanation of this libary

# In[1]:


"""
Library for general purpose functions related to data processing
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


def points_lookup(agent_move, opponent_move): 
    """
    Returns 0, -1, or 3 (agent's points) based on move and opponent move
    """
    POINTS_LOOKUP = pd.DataFrame.from_dict({
        'rock':     [0, 3, -1],
        'paper':    [-1, 0, 3],
        'scissors': [3, -1, 0]
    }, orient='index', columns=['rock', 'paper', 'scissors'])

    return POINTS_LOOKUP[agent_move][opponent_move]


def transition_lookup(previous_move, current_move):
    """
    Get transition based on previous move and current move
    """
    TRANSITION_LOOKUP = pd.DataFrame.from_dict({
        'rock':     ['stay', 'down', 'up'],
        'paper':    ['up', 'stay', 'down'],
        'scissors': ['down', 'up', 'stay']
    }, orient='index', columns=['rock', 'paper', 'scissors'])

    return TRANSITION_LOOKUP[previous_move][current_move]



def transition_move_lookup(transition_type, previous_move):
    """
    Returns next move for a given transition type and previous move
    """
    TRANSITION_MOVE_LOOKUP = pd.DataFrame.from_dict({
        'rock':     ['paper', 'scissors', 'rock'],
        'paper':    ['scissors', 'rock', 'paper'],
        'scissors': ['rock', 'paper', 'scissors']
    }, orient='index', columns=['transition_up', 'transition_down', 'transition_stay'])

    return TRANSITION_MOVE_LOOKUP[transition_type][previous_move]




