"""
Library for general purpose functions related to data processing
"""

import pandas as pd
import numpy as np

DEFAULT_FILE = 'rps_v2_clean.csv'


def read_rps_data(fp = DEFAULT_FILE):
    """
    Read in the data! Yay!
    """
    return pd.read_csv(fp)


def add_col(df, col_names, value = 0):
    """
    assign a new column with values initialized as `value`
    col_names = list of new col names
    """
    for name in col_names:
        df[name] = value
    return df

def separate_df(df):
    """
    create a list of sub-datafames with unque game ids
    """
    experiments = [] 
    for game in df['game_id'].unique(): 
        e = df.loc[df['game_id'] == game]
        e = e.reset_index(drop=True)
        experiments.append(e)
    return experiments


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

def softmax(x, beta = 1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)


def get_softmax_probabilities(df, columns):
    distribution = []
    vals = df[columns]
    for i in range(df.shape[0]):
        soft_max = softmax(vals.iloc[i], beta = 4).tolist() # add aggressive beta term to make max probability move more likely
        distribution.append(soft_max)
    dist = np.array(distribution)
    sofm = pd.DataFrame(dist, columns = ['softmax_prob_rock', 'softmax_prob_paper', 'softmax_prob_scissors'])
    
    return sofm


def pick_move(df, sofm):
    moves = np.array([])
    for i in range(df.shape[0]):
        move_choices = ['rock', 'paper', 'scissors']
        distribution = sofm.iloc[i].tolist() # get ith [rock_prob,paper_prob,scissors_prob] from input df 
        # https://www.w3schools.com/python/ref_random_choices.asp
        # can also use other sample function
        chosen_move = random.choices(move_choices, distribution) 
        moves = np.append(moves, chosen_move)
    df = df.assign(agent_move = moves) # agent_move stores sampled moves
    return df

# from utils.py
OUTCOME_LOOKUP = np.transpose(
    pd.DataFrame.from_dict({
        'rock':     ['tie', 'lose', 'win'],
        'paper':    ['win', 'tie', 'lose'],
        'scissors': ['lose', 'win', 'tie']
    }, orient='index', columns=['rock', 'paper', 'scissors']))

def evaluate_outcome(player_move, opponent_move):
    """
    TODO check that both moves are in outcome lookup
    """
    OUTCOME_LOOKUP = np.transpose(
    pd.DataFrame.from_dict({
        'rock':     ['tie', 'lose', 'win'],
        'paper':    ['win', 'tie', 'lose'],
        'scissors': ['lose', 'win', 'tie']
    }, orient='index', columns=['rock', 'paper', 'scissors']))
    
    return OUTCOME_LOOKUP[player_move][opponent_move]


def assign_agent_outcomes(df):
    """
    Assign outcomes for the agent based on agent move choices.
    df should include only human rows, since agent outcomes are irrelevant for simulating bots
    """
    # df.assign(agent_outcome = '')
    df['agent_outcome'] = df.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    return df



# MODELING & VISUALIZATION
N_ROUNDS = 300
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

