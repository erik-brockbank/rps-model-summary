#!/usr/bin/env python
# coding: utf-8

# # `RL_model_reward.py`

# In[1]:


"""
Functions for adding reward values
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



#########################
### human reward move ###
#########################

def human_reward_move(sub_df):
    """
    tally human rewards on its rock, paper, or scissors moves.
    3 points for a win, 0 for a tie, -1 for a loss
    """
    # reward dictionary
    dic_reward={'win':3,'tie':0,'loss':-1}
    # initialization reward of each game
    dic_move={'rock':0,'paper':0,'scissors':0}

    # loop through the human rows
    for i in range(0,len(sub_df),2):
        # fetch player outcomes
        outcome=sub_df.get('player_outcome').iloc[i]
        # fetch player moves
        move=sub_df.get('player_move').iloc[i]
        # avoid all the nans and 'none's in player_move
        if move != 'none'and not pd.isna(move):
            # tally all the point associated with rewards
            dic_move[move]+=dic_reward[outcome]
            # get the correspounding move's reward column name
            col_name=move+'_reward'

            # store reward points in the reward column
            sub_df.at[i,'rock_reward']=dic_move['rock']
            sub_df.at[i,'paper_reward']=dic_move['paper']
            sub_df.at[i,'scissors_reward']=dic_move['scissors']
    return sub_df

#############################################
### human reward on past and current move ###
#############################################

def human_reward_past_cur_move(sub_df):
    """
    tally human reward based on its past move and current move.
    """
    dic_reward={'win':3,'tie':0,'loss':-1}
    # initialize dic_move to store reward for each combination
    dic_move={'rock':{'rock':0,'paper':0,'scissors':0},'paper':{'rock':0,'paper':0,'scissors':0},
    'scissors':{'rock':0,'paper':0,'scissors':0}}
    for i in range(2,len(sub_df),2):
        # human current outcome
        outcome=sub_df.get('player_outcome').iloc[i]
        # human previous move
        pre_move=sub_df.get('player_move').iloc[i-2]
        # human current move
        cur_move=sub_df.get('player_move').iloc[i]
        # ignore Nan cells
        if pre_move != 'none'and not pd.isna(pre_move) and\
         cur_move != 'none'and not pd.isna(cur_move):
            # get reward for move combination of current round
            dic_move[pre_move][cur_move]+=dic_reward[outcome]
            # fill in reward
            sub_df.at[i,'rock_rock_reward']=dic_move['rock']['rock']
            sub_df.at[i,'rock_paper_reward']=dic_move['rock']['paper']
            sub_df.at[i,'rock_scissors_reward']=dic_move['rock']['scissors']
            sub_df.at[i,'paper_rock_reward']=dic_move['paper']['rock']
            sub_df.at[i,'paper_paper_reward']=dic_move['paper']['paper']
            sub_df.at[i,'paper_scissors_reward']=dic_move['paper']['scissors']
            sub_df.at[i,'scissors_rock_reward']=dic_move['scissors']['rock']
            sub_df.at[i,'scissors_paper_reward']=dic_move['scissors']['paper']
            sub_df.at[i,'scissors_scissors_reward']=dic_move['scissors']['scissors']
    return sub_df

###############################################################
### human reward on opponent past move and own current move ###
###############################################################

def human_reward_oppo_past_cur_move(sub_df):
    """
    tally rewards based on the combination of opponent's past move and
    human current move
    """
    dic_reward={'win':3,'tie':0,'loss':-1}
    # initialize dic_move to store reward of move combination
    dic_move={'rock':{'rock':0,'paper':0,'scissors':0},'paper':{'rock':0,'paper':0,'scissors':0},
    'scissors':{'rock':0,'paper':0,'scissors':0}}
    for i in range(2,len(sub_df),2):
        # outcome of current round
        outcome=sub_df.get('player_outcome').iloc[i]
        # opponent previous move
        oppo_pre_move=sub_df.get('player_move').iloc[i-1]
        # human current move
        cur_move=sub_df.get('player_move').iloc[i]
        # get rid of Nan cells
        if oppo_pre_move != 'none'and not pd.isna(oppo_pre_move) \
        and cur_move != 'none'and not pd.isna(cur_move):
            # get reward of move combination
            dic_move[oppo_pre_move][cur_move]+=dic_reward[outcome]
            # fill in reward value
            sub_df.at[i,'opponent_rock_rock_reward']=dic_move['rock']['rock']
            sub_df.at[i,'opponent_rock_paper_reward']=dic_move['rock']['paper']
            sub_df.at[i,'opponent_rock_scissors_reward']=dic_move['rock']['scissors']
            sub_df.at[i,'opponent_paper_rock_reward']=dic_move['paper']['rock']
            sub_df.at[i,'opponent_paper_paper_reward']=dic_move['paper']['paper']
            sub_df.at[i,'opponent_paper_scissors_reward']=dic_move['paper']['scissors']
            sub_df.at[i,'opponent_scissors_rock_reward']=dic_move['scissors']['rock']
            sub_df.at[i,'opponent_scissors_paper_reward']=dic_move['scissors']['paper']
            sub_df.at[i,'opponent_scissors_scissors_reward']=dic_move['scissors']['scissors']
    return sub_df

def oppo_past_human_past_cur_move(sub_df):
    """
    reward based on opponent past move, human past move and human current move.
    """
    name=['rock','paper','scissors']
    cols=[f'opponent_{oppo}_{past}_{cur}_reward'for oppo in name for past in name for cur in name]
    sub_df=add_col(sub_df, cols, value =0)
    
    dic_reward={'win':3,'tie':0,'loss':-1}

    for i in range(2,len(sub_df),2):
        outcome=sub_df.get('player_outcome').iloc[i]
        oppo_pre_move=sub_df.get('player_move').iloc[i-1]
        agent_pre_move=sub_df.get('player_move').iloc[i-2]
        cur_move=sub_df.get('player_move').iloc[i]
        
        
        if not pd.isna(oppo_pre_move) and not pd.isna(agent_pre_move) and not pd.isna(cur_move):
            sub_df.loc[i,cols]=sub_df.loc[i-2,cols]
            col_name='opponent_'+oppo_pre_move+'_'+agent_pre_move+'_'+cur_move+'_reward'
            sub_df.loc[i,col_name]+=dic_reward[outcome]        
    return sub_df

