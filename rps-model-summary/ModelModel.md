# Overview

Overview of the model (equations, decision policy, etc.)

This model-based agent uses a **<ins>Statistical Learning Approach</ins>** to keeps track of patterns in the opponent’s moves to try and predict the opponent’s next move.

> **Simple Explanation of how the Statistical Learning Agent works:**  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Track whether or not the opponent has a bias towards playing rock, paper or scissors
> 1. Track the cumulative counts of rock, paper and scissors that the bot has made
> 2. Use the cumulative counts as the basis for predicting the next probable move the bot will make
> 3. Choose the right move corresponding the predicted move
>

## Details(Equations) in Model-based Agent Decision-making Process
---
To choose the final move, we utilized ***expected values*** and ***softmax function*** in our model-based agent.

#### <ins>**Expected Values:**<ins>
We calculated corresponding expected values for each move (rock, paper, and scissor) using the probabilities of the bot playing each move based on the previous cumulative counts.
$$
  E[X] = \sum x_ip(x_i)
$$

$x_i$ = the values that X takes   
$p(x_i)$ = the probability that X takes the value $x_i$  



