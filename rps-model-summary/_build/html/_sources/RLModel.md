# Model Overview
<!--
Overview of the model (equations, decision policy, etc.)

This model adopts the notion of **<ins>reinforcement learning</ins>** as the agent determines its next move by scrutinizing its previous winning patterns. The reinforcement learning model is fundamentally different than the model based model as its next moves are no longer susceptible to his opponent's counterparts. Therefore, this model is as known as "model-free". The implementation of the reinforcement model incorporates is summarized as below.


> **Simple Explanation of how the Reinforcement Learning Agent works:**
> Essentially, we kept tracking the agent's prior moves and calculated their respective reward rates through the soft-max probability function. In order to facilitate the tracking process, we generated new columns that store the human player's rewards associated with different scenarios. These scenarios include the rewards of human player's each move, combinations of the human previous move and the human's current move, opponent's past move and the human's current move, opponent past move, human past move, and human current move reward. The counts of the accumulative rewards have been updated under the rules of win-3pts, tie-0pt,and loss--1pt. After the reward counts have been finalized, the agent will simulate playing against each bot by converting the rewards to a probability distribution through the soft-max function. Finally, the visualizations of the simulation of agent outcomes were generated.


## Details(Equations) in Model-free Agent Decision-making Process
---
To choose the final move, we utilized ***expected values*** and ***softmax function*** in our model-free agent.

#### <ins>**Expected Values:**<ins>
We calculated corresponding expected values for each move (rock, paper, and scissor) using the probabilities of the agent playing each move based on the reward column information from the previous round.

$$
  E[X] = \sum x_ip(x_i)
$$

$x_i$ = the values that X takes
$p(x_i)$ = the probability that X takes the value $x_i$ -->
