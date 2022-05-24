# Overview

Overview of the model (equations, decision policy, etc.)

This model-based agent uses a **<ins>Statistical Learning Approach</ins>** to keeps track of patterns in the opponent’s moves to try and predict the opponent’s next move.

> **Simple Explanation of how the Statistical Learning Agent works:**  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Track whether or not the opponent has a bias towards playing rock, paper or scissors
> 1. Track the cumulative counts of rock, paper and scissors that the bot has made
> 2. Use the cumulative counts as the basis for predicting the next probable move the bot will make
> 3. Choose the right move corresponding the predicted move
>

## Model-based Agent Decision-making Processing
---
