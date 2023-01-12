#!/usr/bin/env python
# coding: utf-8

# # `model_wrapper.py`

# ## Initialize

# In[1]:


# Run dependencies
get_ipython().run_line_magic('run', './model_python_lib_utils.ipynb')
get_ipython().run_line_magic('run', './model_python_lib_event_counts.ipynb')
get_ipython().run_line_magic('run', './model_python_lib_decision_functions.ipynb')
get_ipython().run_line_magic('run', './python_lib_visualization.ipynb')


# ## Read data

# In[2]:


df = read_rps_data(os.path.join("data", DEFAULT_FILE))
df.head()


# ## Add bot predictors

# *Initialize columns*

# In[3]:


# TODO may want to move these to `utils.py` as globals so that functions in `event_counts.py` can reference them there
# Add columns corresponding to opponent move counts
supplementary_cols = [
    'opponent_move', 
    'previous_move', 'opponent_previous_move', 
    'previous_outcome', 'opponent_previous_outcome',
    'previous_transition', 'opponent_previous_transition',
    'current_transition',
    'opponent_prev2_transition'
]

event_counts = [
    # transition counts
    'up_transition_count', 'down_transition_count', 'stay_transition_count',
    'cournot_up_transition_count', 'cournot_down_transition_count', 'cournot_stay_transition_count',
    # outcome-dependent transition counts
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

opponent_counts = [
    # base move counts
    'opponent_rock_count', 'opponent_paper_count', 'opponent_scissors_count',
    # transition counts
    'opponent_up_transition_count', 'opponent_down_transition_count', 'opponent_stay_transition_count', 
    'opponent_cournot_up_transition_count', 'opponent_cournot_down_transition_count', 'opponent_cournot_stay_transition_count',
    # outcome-dependent transition counts   
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
]


df = add_col(df, supplementary_cols, value = '')
df = add_col(df, event_counts, value = 1)
df = add_col(df, opponent_counts, value = 1)

# df.head()



# *Populate new columns with counts*

# In[4]:


game_split = separate_df(df)
game_split = get_event_counts(df, game_split)
df = pd.concat(game_split).reset_index(drop = True)

df.head(8)


# ## Compute probabilities

# In[5]:


# TODO make these programmatic (esp. final set)

# Move baserate probabilities
df = add_prob(
    df, 
    'opponent_rock_count', 'opponent_paper_count', 'opponent_scissors_count', 
    'p_opponent_rock', 'p_opponent_paper', 'p_opponent_scissors'
)
# Transition probabilities
df = add_prob(
    df, 
    'opponent_up_transition_count', 'opponent_down_transition_count', 'opponent_stay_transition_count', 
    'p_opponent_transition_up', 'p_opponent_transition_down', 'p_opponent_transition_stay'
)
df = add_prob(
    df, 
    'opponent_cournot_up_transition_count', 'opponent_cournot_down_transition_count', 'opponent_cournot_stay_transition_count', 
    'p_opponent_cournot_transition_up', 'p_opponent_cournot_transition_down', 'p_opponent_cournot_transition_stay'
)
# Outcome-transition probabilities
df = add_prob(
    df,
    'opponent_win_up_count', 'opponent_win_down_count', 'opponent_win_stay_count',
    'p_opponent_win_up', 'p_opponent_win_down', 'p_opponent_win_stay'
)
df = add_prob(
    df,
    'opponent_loss_up_count', 'opponent_loss_down_count', 'opponent_loss_stay_count',
    'p_opponent_loss_up', 'p_opponent_loss_down', 'p_opponent_loss_stay'
)
df = add_prob(
    df,
    'opponent_tie_up_count', 'opponent_tie_down_count', 'opponent_tie_stay_count',
    'p_opponent_tie_up', 'p_opponent_tie_down', 'p_opponent_tie_stay'
)

# Dual transition outcome probabilities
df = add_prob(
    df,
    'opponent_up_win_up_count', 'opponent_up_win_down_count', 'opponent_up_win_stay_count',
    'p_opponent_up_win_up', 'p_opponent_up_win_down', 'p_opponent_up_win_stay'
)
df = add_prob(
    df,
    'opponent_up_loss_up_count', 'opponent_up_loss_down_count', 'opponent_up_loss_stay_count',
    'p_opponent_up_loss_up', 'p_opponent_up_loss_down', 'p_opponent_up_loss_stay'
)
df = add_prob(
    df,
    'opponent_up_tie_up_count', 'opponent_up_tie_down_count', 'opponent_up_tie_stay_count',
    'p_opponent_up_tie_up', 'p_opponent_up_tie_down', 'p_opponent_up_tie_stay'
)

df = add_prob(
    df,
    'opponent_down_win_up_count', 'opponent_down_win_down_count', 'opponent_down_win_stay_count',
    'p_opponent_down_win_up', 'p_opponent_down_win_down', 'p_opponent_down_win_stay'
)
df = add_prob(
    df,
    'opponent_down_loss_up_count', 'opponent_down_loss_down_count', 'opponent_down_loss_stay_count',
    'p_opponent_down_loss_up', 'p_opponent_down_loss_down', 'p_opponent_down_loss_stay'
)
df = add_prob(
    df,
    'opponent_down_tie_up_count', 'opponent_down_tie_down_count', 'opponent_down_tie_stay_count',
    'p_opponent_down_tie_up', 'p_opponent_down_tie_down', 'p_opponent_down_tie_stay'
)

df = add_prob(
    df,
    'opponent_stay_win_up_count', 'opponent_stay_win_down_count', 'opponent_stay_win_stay_count',
    'p_opponent_stay_win_up', 'p_opponent_stay_win_down', 'p_opponent_stay_win_stay'
)
df = add_prob(
    df,
    'opponent_stay_loss_up_count', 'opponent_stay_loss_down_count', 'opponent_stay_loss_stay_count',
    'p_opponent_stay_loss_up', 'p_opponent_stay_loss_down', 'p_opponent_stay_loss_stay'
)
df = add_prob(
    df,
    'opponent_stay_tie_up_count', 'opponent_stay_tie_down_count', 'opponent_stay_tie_stay_count',
    'p_opponent_stay_tie_up', 'p_opponent_stay_tie_down', 'p_opponent_stay_tie_stay'
)


# df.head(8)


# ## Calculate Expected Value

# *Filter bot rows, any rows without interpretable values for previous move, etc.*

# In[6]:


# Filter out non-nan, agent-only rows and add expected value calculations
df_agent = drop_bot_rows(df)


df_agent = df_agent.dropna(subset=['previous_move']) # TODO why is this necessary?
# NB: need this for cournot transition tracking
df_agent = df_agent[df_agent['previous_move'] != 'none']
df_agent = df_agent[df_agent['previous_move'] != '']
df_agent = df_agent[df_agent['opponent_previous_move'] != '']

# outcome-transition
df_agent = df_agent[df_agent['opponent_previous_transition'] != '']
df_agent = df_agent[df_agent['opponent_prev2_transition'] != '']


# df_agent.head(25)
# df_agent.shape


# *Compute expected value*

# In[7]:


# EV for move base rates
df_agent = ev_move_baserate(df_agent)

# EV for transitions
df_agent = ev_transitions(df_agent)
df_agent = ev_cournot(df_agent)

# EV for outcome-transitions
df_agent = ev_previous_outcome(df_agent)

# EV for dual outcome-transitions
df_agent = ev_previous_outcome_previous_transition(df_agent)


# df_agent.head(6)
# df_agent.shape


# In[8]:


# Test
# model7 = df_agent.copy()


# model7['ev_rock'] = model7['ev_outcome_dual_depend_rock']
# model7['ev_paper'] = model7['ev_outcome_dual_depend_paper']
# model7['ev_scissors'] = model7['ev_outcome_dual_depend_scissors']

# # Compute softmax distribution (1 min.)
# m7_softmax = get_softmax_probabilities(
#     model7, 
#     ['ev_rock', 'ev_paper', 'ev_scissors']
# )

# # Select agent move based on softmax computed above (1 min.)
# model7 = pick_move(model7, m7_softmax)

# # Evaluate outcome of agent move choices in simulation above
# model7 = assign_agent_outcomes(model7)


# In[9]:


# plot_summary_coarse = win_summary(groupby_f_data(model7, 'agent_outcome', 5), 'agent_outcome')
# plot_summary_fine = win_summary(groupby_f_data(model7, 'agent_outcome', 60), 'agent_outcome')
# plot_summary_fine = plot_summary_fine[plot_summary_fine['bin'] <= '50']



# In[10]:


# plot_win_rates(plot_summary_coarse)
# plot_win_rates(plot_summary_fine)


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=692c6dea-2203-4a62-9fb8-3ed6d8b73891' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
