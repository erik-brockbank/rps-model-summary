# Overview


High level description of:
a) the data
b) our models
c) our results
d) our conclusions


## Project Goals

Through interaction with others and the environment, people make countless of decisions in various different settings every single day. To achieve a certain desired goal, people often need to make successive observations of a sequential process before making a final decision. We are curious about the model behind the process of how people detect the sequential patterns through interactions despite all the uncertainty and complexity. To achieve this goal, we will be implementing two types of model mimicking human reasoning based on data of people playing rock, paper, scissors with bot from a prior study.


## Data

- Human participants (217 people total) were paired with bot to play rock, paper, scissors
- 300 rounds of rock, paper, scissors per human/bot pairing
- The bot that participants are playing against would use one of the seven strategies shown below
- For each round the moves, outcomes and view time were recorded

<p align = "center">
<img src="img/strategies.png" width = "500px">
</p>


*"+" sign means that the bot transitions to a winning move based on the chosen reference move from last round*  
*"-" sign means that the bot transitions to a losing move based on the chosen reference move from last round*  
*"0" means that the bot plays the exact chosen reference move from last round*  