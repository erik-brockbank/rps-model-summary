#!/usr/bin/env python
# coding: utf-8

# # `model_decision_functions.py`
# 
# TODO add quick explanation of this library

# In[1]:


"""
Functions for converting counts to probabilities, evaluating agent decision and outcomes
"""

# from utils import *
from random import choices




# Basic 3-column probability calculator
# TODO consider revising args to be cols=[], names=[]
def add_prob(df, col_1, col_2, col_3, name1, name2, name3):
    """
    Add probability values as new columns
    """
    count_1 = df[col_1]
    count_2 = df[col_2]
    count_3 = df[col_3]

    total = count_1 + count_2 + count_3

    prob_1 = count_1 / total
    prob_2 = count_2 / total
    prob_3 = count_3 / total

    w_prob = df

    w_prob = add_col(w_prob, [name1], prob_1)
    w_prob = add_col(w_prob, [name2], prob_2)
    w_prob = add_col(w_prob, [name3], prob_3)

    return w_prob


# Below supports outcome and dual transition probabilities
def calculate_prob(num_1, num_2, num_3):
    """Calculating 3 probabilities"""
    total = num_1 + num_2 + num_3

    prob_1 = num_1 / total
    prob_2 = num_2 / total
    prob_3 = num_3 / total
    prob_list = [prob_1, prob_2, prob_3]
    return prob_list



################################
### Expected value functions ###
################################

def calculate_move_ev(df, agent_move):
    """Compute expected values for agent_move"""
    if agent_move == 'rock':
        ev = (df['p_opponent_rock'] * 0) + (df['p_opponent_paper'] * -1) + (df['p_opponent_scissors'] * 3)
    elif agent_move == 'paper':
        ev = (df['p_opponent_rock'] * 3) + (df['p_opponent_paper'] * 0) + (df['p_opponent_scissors'] * -1)
    elif agent_move == 'scissors' :
        ev = (df['p_opponent_rock'] * -1) + (df['p_opponent_paper'] * 3) + (df['p_opponent_scissors'] * 0)

    return ev


def ev_move_baserate(df_agent):
    df_agent['ev_move_baserate_rock'] = calculate_move_ev(df_agent, 'rock')
    df_agent['ev_move_baserate_paper'] = calculate_move_ev(df_agent, 'paper')
    df_agent['ev_move_baserate_scissors'] = calculate_move_ev(df_agent, 'scissors')
    return df_agent


def ev_general(df_agent, agent_move, ev_col_name, prob1, prob2, prob3, move):
    #inputs: dataframe, agent move (string), new col name (string), three probability vectors, move (string): either opponent_previous_move or previous_move(cournot)
    """more general ev function: used for EV of outcome transition and dual dependency transitions"""
    df_agent['filler_1'] = points_lookup(agent_move, transition_move_lookup("transition_up", df_agent[move])).tolist()
    df_agent['filler_2'] = points_lookup(agent_move, transition_move_lookup("transition_down", df_agent[move])).tolist()
    df_agent['filler_3'] = points_lookup(agent_move, transition_move_lookup("transition_stay", df_agent[move])).tolist()
    df_agent['prod_1'] = df_agent[prob1] * df_agent.filler_1
    df_agent['prod_2'] = df_agent[prob2] * df_agent.filler_2
    df_agent['prod_3'] = df_agent[prob3] * df_agent.filler_3
    df_agent['ev_transitions'] = df_agent['prod_1'] + df_agent['prod_2'] + df_agent['prod_3']

    df_agent[ev_col_name] = df_agent['ev_transitions']
    df_agent = df_agent.drop(['filler_1', 'filler_2', 'filler_3',
                              'prod_1', 'prod_2', 'prod_3', 'ev_transitions'], axis=1)
    return df_agent


def ev_transitions(df_agent):
    df_agent = ev_general(df_agent, 'rock', 'ev_transition_rock', 'p_opponent_transition_up', 'p_opponent_transition_down', 'p_opponent_transition_stay', 'opponent_previous_move')
    df_agent = ev_general(df_agent, 'paper', 'ev_transition_paper', 'p_opponent_transition_up', 'p_opponent_transition_down', 'p_opponent_transition_stay', 'opponent_previous_move')
    df_agent = ev_general(df_agent, 'scissors', 'ev_transition_scissors', 'p_opponent_transition_up', 'p_opponent_transition_down', 'p_opponent_transition_stay', 'opponent_previous_move')
    return df_agent


def ev_cournot(df_agent):
    df_agent = ev_general(df_agent, 'rock', 'ev_cournot_transition_rock', 'p_opponent_cournot_transition_up', 'p_opponent_cournot_transition_down', 'p_opponent_cournot_transition_stay', 'previous_move')
    df_agent = ev_general(df_agent, 'paper', 'ev_cournot_transition_paper', 'p_opponent_cournot_transition_up', 'p_opponent_cournot_transition_down', 'p_opponent_cournot_transition_stay', 'previous_move')
    df_agent = ev_general(df_agent, 'scissors', 'ev_cournot_transition_scissors', 'p_opponent_cournot_transition_up', 'p_opponent_cournot_transition_down', 'p_opponent_cournot_transition_stay', 'previous_move')
    return df_agent



def ev_previous_outcome(df_agent):
    """adds 3 ev columns based on previous outcomes"""
    df_agent['key'] = list(range(len(df_agent)))

    df_wins = df_agent[df_agent['opponent_previous_outcome']=='win']
    df_loss = df_agent[df_agent['opponent_previous_outcome']=='loss']
    df_tie =  df_agent[df_agent['opponent_previous_outcome']=='tie']

    #adding columns: w/ ev of rock, paper, scissors
    df_wins = ev_general(df_wins, 'rock', 'ev_outcome_transition_rock', 'p_opponent_win_up', 'p_opponent_win_down', 'p_opponent_win_stay', 'opponent_previous_move')
    df_wins = ev_general(df_wins, 'paper', 'ev_outcome_transition_paper', 'p_opponent_win_up', 'p_opponent_win_down', 'p_opponent_win_stay', 'opponent_previous_move')
    df_wins = ev_general(df_wins, 'scissors', 'ev_outcome_transition_scissors', 'p_opponent_win_up', 'p_opponent_win_down', 'p_opponent_win_stay', 'opponent_previous_move')

    #adding columns: w/ ev of rock, paper, scissors
    df_loss = ev_general(df_loss, 'rock', 'ev_outcome_transition_rock', 'p_opponent_loss_up', 'p_opponent_loss_down', 'p_opponent_loss_stay', 'opponent_previous_move')
    df_loss = ev_general(df_loss, 'paper','ev_outcome_transition_paper', 'p_opponent_loss_up', 'p_opponent_loss_down', 'p_opponent_loss_stay', 'opponent_previous_move')
    df_loss = ev_general(df_loss, 'scissors', 'ev_outcome_transition_scissors','p_opponent_loss_up', 'p_opponent_loss_down', 'p_opponent_loss_stay', 'opponent_previous_move')

    #adding columns: w/ ev of rock, paper, scissors
    df_tie = ev_general(df_tie, 'rock', 'ev_outcome_transition_rock', 'p_opponent_tie_up', 'p_opponent_tie_down', 'p_opponent_tie_stay', 'opponent_previous_move')
    df_tie = ev_general(df_tie, 'paper', 'ev_outcome_transition_paper', 'p_opponent_tie_up', 'p_opponent_tie_down', 'p_opponent_tie_stay', 'opponent_previous_move')
    df_tie = ev_general(df_tie, 'scissors', 'ev_outcome_transition_scissors', 'p_opponent_tie_up', 'p_opponent_tie_down', 'p_opponent_tie_stay', 'opponent_previous_move')

    #merging tables
    stacked_unordered = pd.concat([df_wins, df_loss, df_tie])
    stacked_ordered = stacked_unordered.sort_values(by=['key'], ascending=True)
    df_agent = df_agent.merge(stacked_ordered, 'outer')
    df_agent = df_agent.drop(['key'], axis=1)

    return df_agent


def ev_previous_outcome_previous_transition(df_agent):
    """adds 3 ev columns based on previous outcome and previous transition"""
    df_agent['key'] = list(range(len(df_agent)))

    df_wins_up = df_agent[(df_agent['opponent_previous_outcome']=='win') & (df_agent['opponent_previous_transition']=='up')]
    df_wins_down = df_agent[(df_agent['opponent_previous_outcome']=='win') & (df_agent['opponent_previous_transition']=='down')]
    df_wins_stay = df_agent[(df_agent['opponent_previous_outcome']=='win') & (df_agent['opponent_previous_transition']=='stay')]

    df_loss_up = df_agent[(df_agent['opponent_previous_outcome']=='loss') & (df_agent['opponent_previous_transition']=='up')]
    df_loss_down = df_agent[(df_agent['opponent_previous_outcome']=='loss') & (df_agent['opponent_previous_transition']=='down')]
    df_loss_stay = df_agent[(df_agent['opponent_previous_outcome']=='loss') & (df_agent['opponent_previous_transition']=='stay')]

    df_tie_up = df_agent[(df_agent['opponent_previous_outcome']=='tie') & (df_agent['opponent_previous_transition']=='up')]
    df_tie_down = df_agent[(df_agent['opponent_previous_outcome']=='tie') & (df_agent['opponent_previous_transition']=='down')]
    df_tie_stay = df_agent[(df_agent['opponent_previous_outcome']=='tie') & (df_agent['opponent_previous_transition']=='stay')]

    df_wins_up = ev_general(df_wins_up, 'rock', 'ev_outcome_dual_depend_rock', 'p_opponent_up_win_up', 'p_opponent_up_win_down', 'p_opponent_up_win_stay', 'opponent_previous_move')
    df_wins_up = ev_general(df_wins_up, 'paper', 'ev_outcome_dual_depend_paper', 'p_opponent_up_win_up', 'p_opponent_up_win_down', 'p_opponent_up_win_stay', 'opponent_previous_move')
    df_wins_up = ev_general(df_wins_up, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_up_win_up', 'p_opponent_up_win_down', 'p_opponent_up_win_stay', 'opponent_previous_move')

    df_wins_down = ev_general(df_wins_down, 'rock', 'ev_outcome_dual_depend_rock', 'p_opponent_down_win_up', 'p_opponent_down_win_down', 'p_opponent_down_win_stay', 'opponent_previous_move')
    df_wins_down = ev_general(df_wins_down, 'paper', 'ev_outcome_dual_depend_paper', 'p_opponent_down_win_up', 'p_opponent_down_win_down', 'p_opponent_down_win_stay', 'opponent_previous_move')
    df_wins_down = ev_general(df_wins_down, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_down_win_up', 'p_opponent_down_win_down', 'p_opponent_down_win_stay', 'opponent_previous_move')

    df_wins_stay = ev_general(df_wins_stay, 'rock', 'ev_outcome_dual_depend_rock', 'p_opponent_stay_win_up', 'p_opponent_stay_win_down', 'p_opponent_stay_win_stay', 'opponent_previous_move')
    df_wins_stay = ev_general(df_wins_stay, 'paper', 'ev_outcome_dual_depend_paper', 'p_opponent_stay_win_up', 'p_opponent_stay_win_down', 'p_opponent_stay_win_stay', 'opponent_previous_move')
    df_wins_stay = ev_general(df_wins_stay, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_stay_win_up', 'p_opponent_stay_win_down', 'p_opponent_stay_win_stay', 'opponent_previous_move')

    df_loss_up = ev_general(df_loss_up, 'rock', 'ev_outcome_dual_depend_rock', 'p_opponent_up_loss_up', 'p_opponent_up_loss_down', 'p_opponent_up_loss_stay', 'opponent_previous_move')
    df_loss_up = ev_general(df_loss_up, 'paper', 'ev_outcome_dual_depend_paper', 'p_opponent_up_loss_up', 'p_opponent_up_loss_down', 'p_opponent_up_loss_stay', 'opponent_previous_move')
    df_loss_up = ev_general(df_loss_up, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_up_loss_up', 'p_opponent_up_loss_down', 'p_opponent_up_loss_stay', 'opponent_previous_move')

    df_loss_down = ev_general(df_loss_down, 'rock','ev_outcome_dual_depend_rock', 'p_opponent_down_loss_up', 'p_opponent_down_loss_down', 'p_opponent_down_loss_stay', 'opponent_previous_move')
    df_loss_down = ev_general(df_loss_down, 'paper','ev_outcome_dual_depend_paper', 'p_opponent_down_loss_up', 'p_opponent_down_loss_down', 'p_opponent_down_loss_stay', 'opponent_previous_move')
    df_loss_down = ev_general(df_loss_down, 'scissors','ev_outcome_dual_depend_scissors', 'p_opponent_down_loss_up', 'p_opponent_down_loss_down', 'p_opponent_down_loss_stay', 'opponent_previous_move')

    df_loss_stay = ev_general(df_loss_stay, 'rock', 'ev_outcome_dual_depend_rock', 'p_opponent_stay_loss_up', 'p_opponent_stay_loss_down', 'p_opponent_stay_loss_stay', 'opponent_previous_move')
    df_loss_stay = ev_general(df_loss_stay, 'paper', 'ev_outcome_dual_depend_paper', 'p_opponent_stay_loss_up', 'p_opponent_stay_loss_down', 'p_opponent_stay_loss_stay', 'opponent_previous_move')
    df_loss_stay = ev_general(df_loss_stay, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_stay_loss_up', 'p_opponent_stay_loss_down', 'p_opponent_stay_loss_stay', 'opponent_previous_move')

    df_tie_up = ev_general(df_tie_up, 'rock', 'ev_outcome_dual_depend_rock',  'p_opponent_up_tie_up', 'p_opponent_up_tie_down', 'p_opponent_up_tie_stay', 'opponent_previous_move')
    df_tie_up = ev_general(df_tie_up, 'paper', 'ev_outcome_dual_depend_paper',  'p_opponent_up_tie_up', 'p_opponent_up_tie_down', 'p_opponent_up_tie_stay', 'opponent_previous_move')
    df_tie_up = ev_general(df_tie_up, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_up_tie_up', 'p_opponent_up_tie_down', 'p_opponent_up_tie_stay', 'opponent_previous_move')

    df_tie_down = ev_general(df_tie_down, 'rock', 'ev_outcome_dual_depend_rock', 'p_opponent_down_tie_up', 'p_opponent_down_tie_down', 'p_opponent_down_tie_stay', 'opponent_previous_move')
    df_tie_down = ev_general(df_tie_down, 'paper', 'ev_outcome_dual_depend_paper', 'p_opponent_down_tie_up', 'p_opponent_down_tie_down', 'p_opponent_down_tie_stay', 'opponent_previous_move')
    df_tie_down = ev_general(df_tie_down, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_down_tie_up', 'p_opponent_down_tie_down', 'p_opponent_down_tie_stay', 'opponent_previous_move')

    df_tie_stay = ev_general(df_tie_stay, 'rock', 'ev_outcome_dual_depend_rock', 'p_opponent_stay_tie_up', 'p_opponent_stay_tie_down', 'p_opponent_stay_tie_stay', 'opponent_previous_move')
    df_tie_stay = ev_general(df_tie_stay, 'paper', 'ev_outcome_dual_depend_paper', 'p_opponent_stay_tie_up', 'p_opponent_stay_tie_down', 'p_opponent_stay_tie_stay', 'opponent_previous_move')
    df_tie_stay = ev_general(df_tie_stay, 'scissors', 'ev_outcome_dual_depend_scissors', 'p_opponent_stay_tie_up', 'p_opponent_stay_tie_down', 'p_opponent_stay_tie_stay', 'opponent_previous_move')

    #merging tables
    stacked_unordered = pd.concat([
        df_wins_up, df_wins_down, df_wins_stay,
        df_loss_up, df_loss_down, df_loss_stay,
        df_tie_up, df_tie_down, df_tie_stay
    ])
    stacked_ordered = stacked_unordered.sort_values(by=['key'], ascending=True)
    df_agent = df_agent.merge(stacked_ordered, 'outer')
    df_agent = df_agent.drop(['key'], axis=1)

    return df_agent



#############################
### Move choice functions ###
#############################

def softmax(x, beta = 1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)


def get_softmax_probabilities(df, columns):
    distribution = []
    # ev = df[['ev_agent_rock', 'ev_agent_paper', 'ev_agent_scissors']]
    ev = df[columns]
    for i in range(df.shape[0]):
        soft_max = softmax(ev.iloc[i], beta = 4).tolist() # add aggressive beta term to make max probability move more likely
        distribution.append(soft_max)
    dist = np.array(distribution)
    sofm = pd.DataFrame(dist, columns = ['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors'])

    return sofm


def pick_move(df, sofm):
    moves = np.array([])
    for i in range(df.shape[0]):
        move_choices = ['rock', 'paper', 'scissors']
        distribution = sofm.iloc[i].tolist()
        chosen_move = choices(move_choices, distribution)
        moves = np.append(moves, chosen_move)
    df = df.assign(agent_move = moves)

    return df


def assign_agent_outcomes(df):
    """
    Assign outcomes for the agent based on agent move choices.
    df should include only human rows, since agent outcomes are irrelevant for simulating bots
    """
    df.assign(agent_outcome = '')
    df['agent_outcome'] = df.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    return df

