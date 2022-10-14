#!/usr/bin/env python
# coding: utf-8

# # `model_event_counts.py`
# 
# TODO add quick explanation of this library

# In[1]:


"""
Functions for adding event counts
"""


import pandas as pd

# from utils import *






#####################################
### Opponent previous move counts ###
#####################################

def get_opponent_move(sub_df):
    """
    fills in the `opponent_move` column
    """
    for i in range(len(sub_df)):
        if i%2 == 0:
            sub_df.at[i, 'opponent_move'] = sub_df.at[i + 1, 'player_move']
        else:
            sub_df.at[i, 'opponent_move'] = sub_df.at[i - 1, 'player_move']

    return sub_df


def count_bot_prev_moves(sub_df):
    """
    adds a tally for what the opponent played in prev round
    """
    for i in range(2, len(sub_df)):
        sub_df.at[i, 'opponent_rock_count'] = sub_df.at[i - 2, 'opponent_rock_count']
        sub_df.at[i, 'opponent_paper_count'] = sub_df.at[i - 2, 'opponent_paper_count']
        sub_df.at[i, 'opponent_scissors_count'] = sub_df.at[i - 2, 'opponent_scissors_count']

        if sub_df.at[i - 2, 'opponent_move'] == "rock":
            sub_df.at[i, 'opponent_rock_count'] += 1
        elif sub_df.at[i - 2, 'opponent_move'] == "paper":
            sub_df.at[i, 'opponent_paper_count'] += 1
        elif sub_df.at[i - 2, 'opponent_move'] == "scissors":
            sub_df.loc[i, 'opponent_scissors_count'] += 1

    return sub_df



##################################
### Opponent transition counts ###
##################################

def get_player_prev_move(sub_df):
    """
    get the previous moves
    and add these moves as a new column to the orgininal dataframe
    """
    for i in range(2, len(sub_df)):
        if sub_df.iloc[i]['round_index']-1 == sub_df.iloc[i-2]['round_index']:
            sub_df.at[i, 'previous_move'] = sub_df.at[i-2, 'player_move']
    return sub_df


def get_opponent_prev_move(sub_df):
    """
    get the opponent's previous moves
    and add these moves as a new column to the orgininal dataframe
    """
    for i in range(2, len(sub_df)):
        if sub_df.iloc[i]['round_index']-1 == sub_df.iloc[i-2]['round_index']:
            sub_df.at[i, 'opponent_previous_move'] = sub_df.at[i-2, 'opponent_move']
    return sub_df


def count_bot_transitions(sub_df):
    """
    tally for bot transitions from previous move to current move
    """
    for i in range(3, len(sub_df), 2):
        bot_pre = sub_df.get('previous_move').iloc[i]
        bot_curr = sub_df.get('player_move').iloc[i]
        if bot_pre and bot_pre != 'none' and not pd.isna(bot_pre) and \
            bot_curr and bot_curr != 'none' and not pd.isna(bot_curr):

            sub_df.at[i,'up_transition_count'] = sub_df.at[i-2, 'up_transition_count']
            sub_df.at[i,'stay_transition_count'] = sub_df.at[i-2, 'stay_transition_count']
            sub_df.at[i,'down_transition_count'] = sub_df.at[i-2, 'down_transition_count']

            result = transition_lookup(bot_pre, bot_curr)
            if result == 'up':
                sub_df.at[i,'up_transition_count'] += 1
            elif result == 'stay':
                sub_df.at[i,'stay_transition_count'] += 1
            elif result == 'down':
                sub_df.at[i,'down_transition_count'] += 1

    return sub_df


def count_bot_cournot_transitions(sub_df):
    """
    tally for bot Cournot transitions from (human) opponent previous move to bot current move
    """
    for i in range(3,len(sub_df),2):
        human_pre=sub_df.get('opponent_previous_move').iloc[i]
        bot_curr=sub_df.get('player_move').iloc[i]
        if human_pre and human_pre != 'none' and not pd.isna(human_pre) and \
            bot_curr and bot_curr != 'none' and not pd.isna(bot_curr):

            sub_df.at[i,'cournot_up_transition_count'] = sub_df.at[i-2,'cournot_up_transition_count']
            sub_df.at[i,'cournot_stay_transition_count'] = sub_df.at[i-2,'cournot_stay_transition_count']
            sub_df.at[i,'cournot_down_transition_count'] = sub_df.at[i-2,'cournot_down_transition_count']

            result = transition_lookup(human_pre, bot_curr)
            if result=='up':
                sub_df.at[i,'cournot_up_transition_count'] += 1
            elif result=='stay':
                sub_df.at[i,'cournot_stay_transition_count'] += 1
            elif result=='down':
                sub_df.at[i,'cournot_down_transition_count'] += 1

    return sub_df





#################################
### Outcome transition counts ###
#################################

def get_opponent_prev_outcome(sub_df):
    """
    get the opponent's previous outcomes
    and add these outcomes as a new column to the orgininal dataframe
    """
    for i in range(2, len(sub_df)):
        if sub_df.iloc[i]['round_index']-1 == sub_df.iloc[i-2]['round_index']:
            sub_df.at[i, 'previous_outcome'] = sub_df.at[i-2, 'player_outcome']
    return sub_df


def count_bot_outcome_transitions(sub_df):
    """
    tally based on previous_outcome (win, lose, tie) and bot's own transition (up, down, stay)

    """
    for i in range(3, len(sub_df), 2):
        bot_previous_outcome = sub_df.get('previous_outcome').iloc[i]
        previous_move = sub_df.get('previous_move').iloc[i]
        player_move = sub_df.get('player_move').iloc[i]
        # get bot's transition_outcome from bot's previous move and current move
        bot_transition = transition_lookup(previous_move, player_move)
        if not pd.isna(bot_previous_outcome):
            update_cols = [f'{outcome}_{trans}_count' for outcome in OUTCOMES for trans in TRANSITIONS]
            for col in update_cols:
                sub_df.at[i, col] = sub_df.at[i-2, col]

            col_name = f'{bot_previous_outcome}_{bot_transition}_count'
            sub_df.at[i, col_name] += 1

    return sub_df


#################################################
### Outcome transition dual dependency counts ###
#################################################

def get_previous_current_transitions(sub_df):
    """
    look for bot's previous/current transition for each round for outcome_trans_dual_depend_count function
    """
    for i in range(5,len(sub_df),2):
        previous_move = sub_df.get('previous_move').iloc[i]
        cur_move = sub_df.get('player_move').iloc[i]
        pre_pre_move=sub_df.get('previous_move').iloc[i-2]
        # get bot's transition_outcome from bot's previous move and current move
        cur_transition = transition_lookup(previous_move,cur_move)
        previous_transition = transition_lookup(pre_pre_move,previous_move)

        sub_df.at[i, 'previous_transition'] = previous_transition
        sub_df.at[i, 'current_transition'] = cur_transition
    return sub_df



def count_outcome_trans_dual_depend(sub_df):
    for i in range(5, len(sub_df), 2):
        bot_previous_outcome = sub_df.get('previous_outcome').iloc[i]
        previous_transition = sub_df.get('previous_transition').iloc[i]
        current_transition = sub_df.get('current_transition').iloc[i]
        if not pd.isna(bot_previous_outcome):
            update_cols = [f'{prev_trans}_{outcome}_{trans}_count' for prev_trans in TRANSITIONS for outcome in OUTCOMES for trans in TRANSITIONS]
            for col in update_cols:
                sub_df.at[i, col] = sub_df.at[i-2, col]

            col_name = f'{previous_transition}_{bot_previous_outcome}_{current_transition}_count'
            sub_df.at[i, col_name] += 1

    return sub_df




# Convert event counts to "opponent_x_count" in all human rows
def change_to_human(sub_df, new_columns, original_columns):
    """
    fill in column in every human row that track how many of each transitions its bot opponent made up to that point
    new_columns: list of new column names
    original_columns: list of original column names (in corresponding order with new column names)
    NB: order of new_columns and original_columns must be matched
    """
    for i in range(2, len(sub_df)):
        if sub_df.iloc[i]['is_bot'] == 0 and sub_df.iloc[i]['round_index']-1 == sub_df.iloc[i-1]['round_index']:
            for (new_c, old_c) in zip(new_columns, original_columns):
                sub_df.at[i, new_c] = sub_df.at[i-1, old_c]
        else:
            continue
    return sub_df





# Unify counts above
def get_event_counts(df, experiments):
    """
    """
    for e in experiments:
        # opponent previous move tallying
        e = get_opponent_move(e)
        e = count_bot_prev_moves(e)
        # transition tallying
        e = get_player_prev_move(e)
        e = get_opponent_prev_move(e)
        e = count_bot_transitions(e)
        e = count_bot_cournot_transitions(e)
        # outcome-based transition tallying
        e = get_opponent_prev_outcome(e)
        e = count_bot_outcome_transitions(e)
        # dual transition-outcome tallying
        e = get_previous_current_transitions(e)
        e = count_outcome_trans_dual_depend(e)
        # move bot tallies to human rows
        e = change_to_human(
            e,
            new_columns=[
                'opponent_previous_outcome', 'opponent_previous_transition', 'opponent_prev2_transition',
                # transition counts
                'opponent_up_transition_count', 'opponent_down_transition_count', 'opponent_stay_transition_count',
                'opponent_cournot_up_transition_count', 'opponent_cournot_down_transition_count', 'opponent_cournot_stay_transition_count',
                # outcome-transition counts
                'opponent_win_up_count', 'opponent_win_down_count', 'opponent_win_stay_count',
                'opponent_loss_up_count', 'opponent_loss_down_count', 'opponent_loss_stay_count',
                'opponent_tie_up_count', 'opponent_tie_down_count', 'opponent_tie_stay_count',
                # dual transition outcome counts
                'opponent_up_win_up_count', 'opponent_up_win_down_count', 'opponent_up_win_stay_count',
                'opponent_up_loss_up_count', 'opponent_up_loss_down_count', 'opponent_up_loss_stay_count',
                'opponent_up_tie_up_count', 'opponent_up_tie_down_count', 'opponent_up_tie_stay_count',
                'opponent_down_win_up_count', 'opponent_down_win_down_count', 'opponent_down_win_stay_count',
                'opponent_down_loss_up_count', 'opponent_down_loss_down_count', 'opponent_down_loss_stay_count',
                'opponent_down_tie_up_count', 'opponent_down_tie_down_count', 'opponent_down_tie_stay_count',
                'opponent_stay_win_up_count', 'opponent_stay_win_down_count', 'opponent_stay_win_stay_count',
                'opponent_stay_loss_up_count', 'opponent_stay_loss_down_count', 'opponent_stay_loss_stay_count',
                'opponent_stay_tie_up_count', 'opponent_stay_tie_down_count', 'opponent_stay_tie_stay_count'

            ],
            original_columns = [
                'player_outcome', 'current_transition', 'previous_transition',
                # transition counts
                'up_transition_count', 'down_transition_count', 'stay_transition_count',
                'cournot_up_transition_count', 'cournot_down_transition_count', 'cournot_stay_transition_count',
                # outcome-transition counts
                'win_up_count', 'win_down_count', 'win_stay_count',
                'loss_up_count', 'loss_down_count', 'loss_stay_count',
                'tie_up_count', 'tie_down_count', 'tie_stay_count',
                # dual transition outcome counts
                'up_win_up_count', 'up_win_down_count', 'up_win_stay_count',
                'up_loss_up_count', 'up_loss_down_count', 'up_loss_stay_count',
                'up_tie_up_count', 'up_tie_down_count', 'up_tie_stay_count',
                'down_win_up_count', 'down_win_down_count', 'down_win_stay_count',
                'down_loss_up_count', 'down_loss_down_count', 'down_loss_stay_count',
                'down_tie_up_count', 'down_tie_down_count', 'down_tie_stay_count',
                'stay_win_up_count', 'stay_win_down_count', 'stay_win_stay_count',
                'stay_loss_up_count', 'stay_loss_down_count', 'stay_loss_stay_count',
                'stay_tie_up_count', 'stay_tie_down_count', 'stay_tie_stay_count'
            ]
        )

    return experiments

