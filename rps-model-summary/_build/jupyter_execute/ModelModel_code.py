#!/usr/bin/env python
# coding: utf-8

# # Reference: Model Code

# ## Initialization

# In[1]:


# Run dependencies
get_ipython().run_line_magic('run', './model_python_lib_utils.ipynb')
get_ipython().run_line_magic('run', './model_python_lib_event_counts.ipynb')
get_ipython().run_line_magic('run', './model_python_lib_decision_functions.ipynb')
get_ipython().run_line_magic('run', './python_lib_visualization.ipynb')

get_ipython().run_line_magic('run', './model_wrapper.ipynb')


# ## Model Fit

# In[2]:


# Read data
df = read_rps_data(os.path.join("data", DEFAULT_FILE))
df.head()


# ## Human performance: benchmark
# 

# In[8]:


# Plot human win rates
f_a = groupby_f_data(df_agent, 'player_outcome', bins=10)
# f_a = f_a[f_a['bin']<='50']
plot_win_rates(f_a[f_a['player_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## Null model: move base rates

# *Run model*

# In[4]:


model1 = df_agent.copy()

model1['ev_rock'] = model1['ev_move_baserate_rock']
model1['ev_paper'] = model1['ev_move_baserate_paper']
model1['ev_scissors'] = model1['ev_move_baserate_scissors']

# Compute softmax distribution (1 min.)
m1_softmax = get_softmax_probabilities(
    model1, 
    ['ev_rock', 'ev_paper', 'ev_scissors']
)

# Select agent move based on softmax computed above (1 min.)
model1 = pick_move(model1, m1_softmax)
# model1.head(25)

# Evaluate outcome of agent move choices in simulation above
model1 = assign_agent_outcomes(model1)
# model1.head(25)

# runtime: 1-2 mins?


# *Plot model results*

# In[10]:


# Plot agent win rates
f_b = groupby_f_data(model1, 'agent_outcome', bins=10)
# f_b = f_b[f_b['bin']<='50']
plot_win_rates(f_b[f_b['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## Transition model: bot transitions only

# *Run model*

# In[11]:


import time
start = time.time()

model2 = df_agent.copy()

model2['ev_rock'] = model2['ev_move_baserate_rock'] + model2['ev_transition_rock']
model2['ev_paper'] = model2['ev_move_baserate_paper'] + model2['ev_transition_paper']
model2['ev_scissors'] = model2['ev_move_baserate_scissors'] + model2['ev_transition_scissors']

# Compute softmax distribution (1 min.)
m2_softmax = get_softmax_probabilities(
    model2, 
    ['ev_rock', 'ev_paper', 'ev_scissors']
)

# Select agent move based on softmax computed above (1 min.)
model2 = pick_move(model2, m2_softmax)

# Evaluate outcome of agent move choices in simulation above
model2 = assign_agent_outcomes(model2)
# model2.head(25)

end = time.time()
print(end - start)

# runtime: about 1 min


# *Plot model results*

# In[14]:


# Plot agent win rates
f_c = groupby_f_data(model2, 'agent_outcome', bins = 10)
# f_c = f_c[f_c['bin']<='50']
plot_win_rates(f_c[f_c['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## Transition model: bot Cournot transitions only

# *Run model*

# In[15]:


start = time.time()

model3 = df_agent.copy()

model3['ev_rock'] = model3['ev_move_baserate_rock'] + model3['ev_cournot_transition_rock']
model3['ev_paper'] = model3['ev_move_baserate_paper'] + model3['ev_cournot_transition_paper']
model3['ev_scissors'] = model3['ev_move_baserate_scissors'] + model3['ev_cournot_transition_scissors']

# Compute softmax distribution (1 min.)
m3_softmax = get_softmax_probabilities(
    model3, 
    ['ev_rock', 'ev_paper', 'ev_scissors']
)

# Select agent move based on softmax computed above (1 min.)
model3 = pick_move(model3, m3_softmax)

# Evaluate outcome of agent move choices in simulation above
model3 = assign_agent_outcomes(model3)
# model3.head(25)

end = time.time()
print(end - start)


# *Plot model results*

# In[18]:


# Plot agent win rates
f_d = groupby_f_data(model3, 'agent_outcome', bins=10)
# f_d = f_d[f_d['bin']<='50']
plot_win_rates(f_d[f_d['agent_outcome']=='win'], img_name= 'model_botcournot_transition_only.png') # NB: add a filename argument to save the figure locally


# ## Transition model: bot transitions + Cournot transitions

# *Run model*

# In[19]:


start = time.time()

model4 = df_agent.copy()

model4['ev_rock'] = model4['ev_move_baserate_rock'] + model4['ev_transition_rock'] + model4['ev_cournot_transition_rock']
model4['ev_paper'] = model4['ev_move_baserate_paper'] + model4['ev_transition_paper'] + model4['ev_cournot_transition_paper']
model4['ev_scissors'] = model4['ev_move_baserate_scissors'] + model4['ev_transition_scissors']+ model4['ev_cournot_transition_scissors']

# Compute softmax distribution (1 min.)
m4_softmax = get_softmax_probabilities(
    model4, 
    ['ev_rock', 'ev_paper', 'ev_scissors']
)

# Select agent move based on softmax computed above (1 min.)
model4 = pick_move(model4, m4_softmax)

# Evaluate outcome of agent move choices in simulation above
model4 = assign_agent_outcomes(model4)
# model4.head(25)

end = time.time()
print(end- start)

# runtime: about 1 min


# *Plot model results*

# In[22]:


# Plot agent win rates
f_e = groupby_f_data(model4, 'agent_outcome', bins = 10)
# f_e = f_e[f_e['bin']<='50']
plot_win_rates(f_e[f_e['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## Outcome-transition model: outcome-transitions only

# *Run model*

# In[23]:


start = time.time()

model5 = df_agent.copy()

model5['ev_rock'] = model5['ev_outcome_transition_rock']
model5['ev_paper'] = model5['ev_outcome_transition_paper']
model5['ev_scissors'] = model5['ev_outcome_transition_scissors']

# Compute softmax distribution (1 min.)
m5_softmax = get_softmax_probabilities(
    model5, 
    ['ev_rock', 'ev_paper', 'ev_scissors']
)

# Select agent move based on softmax computed above (1 min.)
model5 = pick_move(model5, m5_softmax)

# Evaluate outcome of agent move choices in simulation above
model5 = assign_agent_outcomes(model5)

end = time.time()
print(end - start)

# runtime: about 1 min


# *Plot model results*

# In[31]:


# Plot agent win rates
f_f = groupby_f_data(model5, 'agent_outcome', bins=10)
# f_f = f_f[f_f['bin']<='50']
plot_win_rates(f_f[f_f['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# ## Dual-transition outcome model: complex strategy only

# *Run model*

# In[34]:


start = time.time()

model6 = df_agent.copy()


model6['ev_rock'] = model6['ev_outcome_dual_depend_rock']
model6['ev_paper'] = model6['ev_outcome_dual_depend_paper']
model6['ev_scissors'] = model6['ev_outcome_dual_depend_scissors']

# Compute softmax distribution (1 min.)
m6_softmax = get_softmax_probabilities(
    model6, 
    ['ev_rock', 'ev_paper', 'ev_scissors']
)

# Select agent move based on softmax computed above (1 min.)
model6 = pick_move(model6, m6_softmax)

# Evaluate outcome of agent move choices in simulation above
model6 = assign_agent_outcomes(model6)

end = time.time()
print(end - start)
# runtime: about 1 min


# *Plot model results*

# In[37]:


# Plot agent win rates
f_g = groupby_f_data(model6, 'agent_outcome', bins=10)
# f_g = f_g[f_g['bin']<='50']
plot_win_rates(f_g[f_g['agent_outcome']=='win']) # NB: add a filename argument to save the figure locally


# In[ ]:




