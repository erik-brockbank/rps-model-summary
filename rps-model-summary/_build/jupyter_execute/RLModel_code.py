#!/usr/bin/env python
# coding: utf-8

# # Reference: RL Model Code

# This page is a reference for the code used to generate the figures in the previous results page.
# 
# This is included largely for reproducibility so that others can generate similar results; it also provides more fine-grained insight into the model implementation.

# ## Initialization

# In[1]:


# Run dependencies
get_ipython().run_line_magic('run', './RL_model_python_lib_utils.ipynb')
get_ipython().run_line_magic('run', './RL_model_python_lib_reward.ipynb')
get_ipython().run_line_magic('run', './RL_model_python_lib_decision_functions.ipynb')
get_ipython().run_line_magic('run', './python_lib_visualization.ipynb')


# In[ ]:


# Read data
df = read_rps_data(os.path.join("data", DEFAULT_FILE))

# Add opponent move column
separated = separate_df(df)
for e in separated:
    get_opponent_move(e)
df = pd.concat(separated)


# ## *Null Model*: Reward Learning from Move Baserates

# In[4]:


df_a = add_col(df, ['rock_reward', 'paper_reward','scissors_reward',], value =0)
separated = separate_df(df_a)
for e in separated:
    human_reward_move(e)
df_a = pd.concat(separated)


# In[5]:


soft_dist = get_softmax_probabilities(
    df_a, # df should be just human rows at this point, strip out nans etc. 
    ['rock_reward', 'paper_reward', 'scissors_reward']
)


# In[6]:


df_a = pick_move(df_a, soft_dist)


# In[7]:


df_a=df_a[df_a['is_bot']==0]
df_a=assign_agent_outcomes(df_a)


# In[8]:


f_a = groupby_f_data(df_a, 'agent_outcome', bins=60)
f_a = f_a[f_a['bin']<='50']
plot_win_rates(f_a[f_a['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## *Transition Model*: Reward Learning from Previous Moves

# In[9]:


separated = separate_df(df)
for e in separated:
    human_reward_past_cur_move(e)
df_b = pd.concat(separated)


# In[10]:


separated = separate_df(df_b)
df_result_b = pd.DataFrame()
for e in separated:
    e = get_softmax_probabilities_3b(e)
    e=pick_move_v2(e)
    e['agent_outcome'] = e.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    df_result_b=pd.concat([df_result_b,e],axis=0)


# In[11]:


f_b = groupby_f_data(df_result_b, 'agent_outcome', bins=60)
f_b = f_b[f_b['bin']<='50']
plot_win_rates(f_b[f_b['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## *Transition Model*: Reward Learning from Opponent Previous Moves

# In[12]:


# separate df into same game id
separated = separate_df(df)
for e in separated:
    human_reward_oppo_past_cur_move(e)
df_c = pd.concat(separated)


# In[13]:


separated = separate_df(df_c)
df_result_c = pd.DataFrame()
# align results from the generaed agent move and opponent move
for e in separated:
    e = get_softmax_probabilities_3c(e)
    e=pick_move_v2(e)
    e['agent_outcome'] = e.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    df_result_c=pd.concat([df_result_c,e],axis=0)


# In[14]:


f_c = groupby_f_data(df_result_c, 'agent_outcome', bins=60)
f_c = f_c[f_c['bin']<='50']
plot_win_rates(f_c[f_c['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## *Combined Transition Model*: Reward Learning from Self and Opponent Previous Moves

# In[17]:


df=df.replace('none',np.NaN)
separated = separate_df(df)
for e in separated:
    oppo_past_human_past_cur_move(e)
df_combine = pd.concat(separated)


# In[18]:


separated = separate_df(df_combine)
df_result_combined = pd.DataFrame()
for e in separated:
    e = get_softmax_probabilities_combined(e)
    e = pick_move_v2(e)
    e['agent_outcome'] = e.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    df_result_combined = pd.concat([df_result_combined,e],axis=0)


# In[19]:


f_combined = groupby_f_data(df_result_combined, 'agent_outcome', bins=60)
f_combined = f_combined[f_combined['bin']<='50']

plot_win_rates(f_combined[f_combined['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## *Disjunctive Transition Model*: Learning Separately from Self and Opponent Previous Moves

# In[ ]:


separated_agent_past = separate_df(df_b)
separated_oppo_past=separate_df(df_c)
df_result_mix = pd.DataFrame()
count=0
for i in range(len(separated_oppo_past)):
    e=get_softmax_probabilities_mix(separated_agent_past[i], separated_oppo_past[i])
    e=pick_move_v2(e)
    e['agent_outcome'] = e.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    df_result_mix=pd.concat([df_result_mix, e], axis=0)


# In[ ]:


f_mix = groupby_f_data(df_result_mix, 'agent_outcome', bins=60)
f_mix = f_mix[f_mix['bin']<='50']
plot_win_rates(f_mix[f_mix['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally

