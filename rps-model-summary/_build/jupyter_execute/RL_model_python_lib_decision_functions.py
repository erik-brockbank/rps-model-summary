#!/usr/bin/env python
# coding: utf-8

# # `RL_model_decision_functions.py`

# In[1]:


"""
Functions for converting counts to probabilities, evaluating agent decision and outcomes
"""

# from utils import *
import random
from random import choices


#########################
## softmax probability ##
#########################

def softmax(x, beta = 1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)

def get_softmax_probabilities(df, columns):
    """
    create a softmax dataframe to store the probabilities for choosing
    rock, paper, or scissors move.
    This general sofm function is used for human_reward_move model.
    """
    vals = df[columns]
    vals = vals.apply(softmax,axis=1,beta=4)
    return vals

def get_softmax_probabilities_3b(df):
    """
    create a softmax dataframe to store the probabilities for choosing
    rock, paper, or scissors move.
    This sofm function is used for human_reward_past_cur_move model.
    """
    distribution = []
    # assign vals to store reward list
    # Since first round has no previous moves, we add even probability to it
    vals=[[0.33,0.33,0.33]]
    for i in range(2,df.shape[0],2):
        # agent's previous move
        pre_move=df.get('player_move').iloc[i-2]
        # avoid Nan pre_move
        if pre_move != 'none' and not pd.isna(pre_move):
            # get the reward combination with specific pre_move
            reward_cols=[f'{pre_move}_rock_reward',f'{pre_move}_paper_reward',f'{pre_move}_scissors_reward']
            val = df[reward_cols].iloc[i].tolist()
            vals.append(val) # append reward value to vals
        else:
            # if there's no pre_move, use the last reward list (val)
            val=vals[-1]
            vals.append(val)
    # convert reward list to softmax probability
    soft_max=[softmax(x) for x in vals]
    # create a softmax distribution dataframe
    sofm = pd.DataFrame(soft_max, columns = ['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors'])

    # only human rows
    df_new = df[df.is_bot == 0].reset_index()
    # concate human rows and softmax distribution
    df_new = pd.concat([df_new,sofm], axis = 1)
    return df_new

def get_softmax_probabilities_3c(df):
    '''
    generate softmax probability distribution of each round so we can sample moves from the distribution
    '''
    # df.dropna(axis = 0)
    distribution = []
    vals=[[0.33,0.33,0.33]] # has deleted one default prob list
    for i in range(2,df.shape[0],2):
        pre_move=df.get('player_move').iloc[i-1] # -1 instead of -2 since opponent_pre
        if pre_move != 'none' and not pd.isna(pre_move):
            reward_cols=[f'opponent_{pre_move}_rock_reward',f'opponent_{pre_move}_paper_reward',f'opponent_{pre_move}_scissors_reward']
            val = df[reward_cols].iloc[i].tolist()
            vals.append(val)
        else:
            val=vals[-1]
            vals.append(val)
    soft_max=[softmax(x) for x in vals]
    sofm = pd.DataFrame(soft_max, columns = ['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors'])

    # strip only human df outside of the function
    df_new = df[df.is_bot == 0].reset_index()
    df_new = pd.concat([df_new,sofm], axis = 1)
    return df_new

def get_softmax_probabilities_mix(df_agent_past,df_opponent_past):
    """
    choose human_reward_past or opponent_reward_past based on which strategy has higher reward
    """
    
    distribution = []
    vals=[[0.33,0.33,0.33]] # has deleted one default prob list
    for i in range(2,max(df_agent_past.shape[0],df_agent_past.shape[0]),2):
        oppo_pre_move=df_agent_past.get('player_move').iloc[i-1]
        agent_pre_move=df_agent_past.get('player_move').iloc[i-2]
        
        if agent_pre_move != 'none' and oppo_pre_move!= 'none' and not pd.isna(oppo_pre_move) and not pd.isna(agent_pre_move):
            agent_reward_cols=[f'{agent_pre_move}_rock_reward',
                         f'{agent_pre_move}_paper_reward',
                         f'{agent_pre_move}_scissors_reward'] #df_agent_past only has bot 0
            oppo_reward_cols=[f'opponent_{oppo_pre_move}_rock_reward',
                         f'opponent_{oppo_pre_move}_paper_reward',
                         f'opponent_{oppo_pre_move}_scissors_reward']# df_opponent_past has bot=0&1,so index not match
            val_agent=df_agent_past[agent_reward_cols].iloc[i].tolist()
            val_oppo = df_opponent_past[oppo_reward_cols].iloc[i].tolist()
            if sum(val_agent)>sum(val_oppo):
                val=val_agent
            else:
                val=val_oppo
            vals.append(val)
        else:
            val=vals[-1]
            vals.append(val)
            
    soft_max=[softmax(x) for x in vals] 
    sofm = pd.DataFrame(soft_max, columns = ['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors'])
    
    # strip only human df outside of the function
    df_new = df_agent_past[df_agent_past.is_bot == 0].reset_index() 
    df_new = pd.concat([df_new,sofm], axis = 1)
    return df_new

#############################
### Move choice functions ###
#############################

def pick_move(df, sofm):
    """
    pick agent move based of the softmax probability distribution
    """
    moves = np.array([])
    for i in range(df.shape[0]):
        move_choices = ['rock', 'paper', 'scissors']
        distribution = sofm.iloc[i].tolist()
        chosen_move = choices(move_choices, distribution)
        moves = np.append(moves, chosen_move)
    df = df.assign(agent_move = moves)

    return df

def pick_move_3b(df):
    moves = np.array([])
    for i in range(df.shape[0]):
        move_choices = ['rock', 'paper', 'scissors']
        distribution = df[['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors']].iloc[i].tolist() # get ith [rock_prob,paper_prob,scissors_prob] from input df
        chosen_move = random.choices(move_choices, distribution)
        moves = np.append(moves, chosen_move)
    df = df.assign(agent_move = moves) # agent_move stores sampled moves
    return df

def pick_move_3c(df):
    '''
    sample agent move based on softmax distribution
    '''
    moves = np.array([])
    for i in range(df.shape[0]):
        move_choices = ['rock', 'paper', 'scissors']
        distribution = df[['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors']].iloc[i].tolist() # get ith [rock_prob,paper_prob,scissors_prob] from input df
        chosen_move = random.choices(move_choices, distribution)
        moves = np.append(moves, chosen_move)
    df = df.assign(agent_move = moves) # agent_move stores sampled moves
    return df

def pick_move_3d(df):
    moves = np.array([])
    for i in range(df.shape[0]):
        move_choices = ['rock', 'paper', 'scissors']
        distribution = df[['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors']].iloc[i].tolist() # get ith [rock_prob,paper_prob,scissors_prob] from input df 
        chosen_move = random.choices(move_choices, distribution) 
        moves = np.append(moves, chosen_move)
    df = df.assign(agent_move = moves) # agent_move stores sampled moves
    return df

def assign_agent_outcomes(df):
    """
    Assign outcomes for the agent based on agent move choices.
    df should include only human rows, since agent outcomes are irrelevant for simulating bots
    """
    df.assign(agent_outcome = '')
    df=df.assign(agent_outcome=df.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1))
    return df


# In[ ]:




