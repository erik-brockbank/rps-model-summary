#!/usr/bin/env python
# coding: utf-8

# # Reference: RL Model Code

# ## Initialization

# In[1]:


# Run dependencies
get_ipython().run_line_magic('run', './RL_model_python_lib_utils.ipynb')
get_ipython().run_line_magic('run', './RL_model_python_lib_reward.ipynb')
get_ipython().run_line_magic('run', './RL_model_python_lib_decision_functions.ipynb')
get_ipython().run_line_magic('run', './RL_model_python_lib_visualization.ipynb')


# ## Model Fit

# In[2]:


df = read_rps_data(os.path.join("data", DEFAULT_FILE))


# In[3]:


# add opponent move column
separated = separate_df(df)
for e in separated:
    get_opponent_move(e)
df = pd.concat(separated)


# ### a) human_reward_move

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


plot_win_rates(win_summary(groupby_f_data(df_a,'agent_outcome',30),'agent_outcome'))


# In[ ]:





# In[ ]:





# ### 3b human_past_current_reward_move

# In[10]:


separated = separate_df(df)
for e in separated:
    human_reward_past_cur_move(e)
df_b = pd.concat(separated)


# In[21]:


separated = separate_df(df_b)
df_result_b = pd.DataFrame()
for e in separated:
    e = get_softmax_probabilities_3b(e)
    e=pick_move_3b(e)
    e['agent_outcome'] = e.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    df_result_b=pd.concat([df_result_b,e],axis=0)


# In[22]:


plot_win_rates(win_summary(groupby_f_data(df_result_b,'agent_outcome',30),'agent_outcome'))


# ### 3c opponent_past_human_current_reward_move

# In[27]:


# separate df into same game id
separated = separate_df(df)
for e in separated:
    human_reward_oppo_past_cur_move(e)
df_c = pd.concat(separated)


# In[25]:


separated = separate_df(df_c)
df_result_c = pd.DataFrame()
# align results from the generaed agent move and opponent move
for e in separated:
    e = get_softmax_probabilities_3c(e)
    e=pick_move_3c(e)
    e['agent_outcome'] = e.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    df_result_c=pd.concat([df_result_c,e],axis=0)


# In[26]:


plot_win_rates(win_summary(groupby_f_data(df_result_c,'agent_outcome',30),'agent_outcome'))


# ### 3d) opponent_past_human_past_current_move (mix)

# In[29]:


separated_agent_past = separate_df(df_b)
separated_oppo_past=separate_df(df_c)
df_result_mix = pd.DataFrame()
count=0
for i in range(len(separated_oppo_past)):
# for e_agent,e_oppo in separated_agent_past, separated_oppo_past:
    e=get_softmax_probabilities_mix(separated_agent_past[i], separated_oppo_past[i])
    e=pick_move_3d(e)
    e['agent_outcome'] = e.apply(lambda x: evaluate_outcome(x['agent_move'], x['opponent_move']), axis=1)
    df_result_mix=pd.concat([df_result_mix,e],axis=0)


# In[30]:


plot_win_rates(win_summary(groupby_f_data(df_result_mix,'agent_outcome',30),'agent_outcome'))


# In[ ]:




