#!/usr/bin/env python
# coding: utf-8

# # Reference: Model Code

# ## Initialization

# In[1]:


# Run dependencies
get_ipython().run_line_magic('run', './model_python_lib_utils.ipynb')
get_ipython().run_line_magic('run', './model_python_lib_event_counts.ipynb')
get_ipython().run_line_magic('run', './model_python_lib_decision_functions.ipynb')
get_ipython().run_line_magic('run', './model_python_lib_visualization.ipynb')


# ## Model Fit

# In[2]:


# Read data
df = read_rps_data(os.path.join("data", DEFAULT_FILE))
df.head()


# In[3]:


# Fit model...

