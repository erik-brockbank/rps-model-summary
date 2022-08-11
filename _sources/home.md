# Modeling Adaptive Reasoning in 🪨 📜 ✂️

## Welcome!

This is the official home page for a multi-year project building ***models of how people learn from and adapt to their opponents*** when playing Rock, Paper, Scissors (or Roshambo). The bulk of the code was completed by UCSD students Betty Gong, MJ Mei, Annalea O'Halloran, and Alison Yu. The project was spearheaded by Erik Brockbank using data originally collected by Erik Brockbank and Edward Vul at the UCSD [Computational Cognition Lab](https://www.evullab.org/). Read on to learn more!


## Why this matters

Humans are excellent pattern learners. As children, our ability to learn language and social norms relies on detecting rich and nuanced structure in our environment. As adults, we rely on these abilities in everyday settings as well as for complex behaviors like testing scientific theories. One area where this is especially crucial is in **adversarial settings** where we're trying to *outwit* an opponent.

From sports and games to courtroom drama and international relations, people have a unique ability to adapt in adversarial interactions. Consider, for example, this interview with tennis great Andre Agassi, in which he describes realizing that his opponent Boris Becker would give away his serve direction by [sticking out his tongue](https://www.youtube.com/watch?v=57BMzCM6hQI) in a predictable way before serving. Or, this more recent article about the [strategic signalling](https://www.nytimes.com/2022/02/15/world/europe/us-russia-ukraine-war.html) between the US and Russia in Ukraine.

***What kind of reasoning forms the basis for recognizing patterns in adversarial interactions and what are the limits in this ability?***

To try and answer this, we explore people's behavior in a *much simpler setting* than elite tennis or international relations. We use the game of Rock, Paper, Scissors (RPS) to understand what kinds of behavioral patterns people can recognize and adapt to in order to outwit their opponent. Specifically, people play *hundreds of rounds* (usually 300) against another human or a bot. In this setting, RPS is a perfect *model system* for studying adversarial interactions. The rules are simple and people don't have a lot of prior expertise (this differentiates it from something like chess), so the best way to beat your opponent is to uncover patterns in their moves that allow you to predict their next move (for example, people will often play the same move again after winning...). In this project and elsewhere, we ask what kinds of patterns people can detect in their opponent's moves, what kinds of patterns they can revise in their own moves, and what kind of learning processes underly this behavior.

***On this site, we present one particular thread in this effort: a large-scale modeling project that compares several different computational models of how people adapt to stable patterns in an opponent's moves.***

The pages on this site present a summary of our results.

## Project Overview

The site is divided into the following sections:

🪨 **SUMMARY**: this section provides first a high-level summary of the approach and the findings on the [Overview](Overview.md) page, then a longer explanation of the background for this work on the [Introduction](Introduction.md) page and the data we use for the project on the [Data](Data.ipynb) page.

📜 **MODEL-BASED AGENT**: this section outlines key results for the first of two models we propose to explain how people learn from patterns in a RPS opponent's behavior. The [Model Overview](ModelModel.md) page describes how this model works and the [Model Results](ModelModel_results.md) page highlights the primary results comparing this model to human data. Finally, the [Reference: Model Code](ModelModel_code.ipynb) shows the actual code used to run the model in case you want to peek under the hood or try it at home.

✂️ **MODEL-FREE RL AGENT**: this section illustrates the second model we used to explore people's ability to adapt to their opponent. The **MODEL-BASED AGENT** is a *predictive* model with an explicit hypothesis space about opponent behavior, while the **MODEL-FREE RL AGENT**, as the name implies, takes a reinforcement learning approach that doesn't involve any kind of predictive inference about the opponent. This section is laid out just like the one above, with a [Model Overview](RLModel.md) page that explains how this model works, a subsequent [Model Results](RLModel_results.md) page demonstrating how well this model's behavior aligns with human performance, and a reference page for the underlying code on [Reference: RL Model Code](RLModel_code.ipynb).

🪨 **MODEL COMPARISON**: this section is where we fit parameter estimates for the two models above and compare how well they describe our human data. ***This section is still in progress! Check back for finalized results!***

📜 **DISCUSSION**: this section provides a high-level summary of the results in the previous section and describes planned future work. Take a look here if you want to get a sense of the broader impact or plans for this project moving forward.

✂️ **APPENDIX - CODE**: this section is where we've placed all the python libraries needed to run the model code in the sections above ([Reference: Model Code](ModelModel_code.ipynb) and [Reference: RL Model Code](RLModel_code.ipynb)). This is primarily here for full reproducibility purposes but shouldn't be something you need unless you're trying to run this yourself or are curious about how we actuall constructed these models. Each of the pages in this section is a python library that we've placed in jupyter notebook for easy viewing. There are four libraries for each of the two models:
1. a utils library with general purpose functions needed for each model ([model_utils.py](model_python_lib_utils.ipynb) and [RL_model_utils.py](RL_model_python_lib_utils.ipynb))
2. a library for tracking the prior events relevant to each model's decision function ([model_event_counts.py](model_python_lib_event_counts.ipynb) and [RL_model_reward.py](RL_model_python_lib_reward.ipynb))
3.  a decision function library with the functions each model uses to choose a move ([model_decision_functions.py](model_python_lib_decision_functions.ipynb) and [RL_model_decision_functions.py](RL_model_python_lib_decision_functions.ipynb))
4. a visualization library for generating the graphs used to evaluate each model's performance ([model_visualization.py](model_python_lib_visualization.ipynb) and [RL_model_visualization.py](RL_model_python_lib_visualization.ipynb)).

Together, this represents a complete overview of the code and results for these models. The *source-of-truth* repo for the model-based agent is [here](https://github.com/erik-brockbank/rps-agent-model) and the repo for the RL agent is [here](https://github.com/erik-brockbank/rps-rl-model).

Feel free to contact [Erik Brockbank](http://www.erikbrockbank.com/) with any questions about this work!

[//]: # "TODO add videos of kids/adults playing RPS (can you embed media in markdown?)"
