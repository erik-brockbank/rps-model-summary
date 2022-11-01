# Model Overview

***At a high-level, the RL-based agent simulates what would happen if (human) players were trying to determine their next optimal move on the basis of particular rewards associated with different past events (patterns in agent's previous moves).***

In this way, the RL-based agent represents a kind of *idealized* model of human behavior that serves as a basis for evaluating what people are *actually* doing.

In order to simulate human play against the different bot opponents, the RL-based agent has two high-level functions (we spell these out in more detail below):

1. **Track reward patterns**: First, the agent maintains a count of *rewards* that it uses to try and predict its next move. To demonstrate, the simplest version is to straightly add up the rewards (3 for 'Win', 0 for 'tie', -1 for 'lose) that the agent have from the previous rounds. In each round, these rewards allow for a (very rough) prediction that the agent will choose whichever move that generated a win on the previous round.

2. **Choose an optimal move**: The RL-based choose the next move based on whichever move accumulates the highest rewards. The higher the reward is, the more likely the agent will choose this move. The agent use certain reward patterns we will define later to make its own move choice. In one word, tt samples a move each round probabilistically based on the reward of moves from the past.

***How well does an RL-based model using the steps above capture human patterns of learning against the different bot opponents?***

To answer this question, we implement different versions of the model and compare them to the human behavior in our experiment data. Each version of the model has the same *decision function* from step 2 above but differs in the underlying information that it tracks about different combination of reward counts (step 1). In this way, we simulate different levels of *reward tracking* that people might be engaging in when playing against algorithmic opponents.

# Model Details

Below, we describe in more detail the two central aspects of the model-based agent's behavior outlined above.

*The content in this section is meant to provide enough detail for readers to inspect the underlying code or think carefully about the [Model Results](RLModel_results.md) on the next page.*


## Tracking behavior associated rewards

For an apples-to-apples comparison of model behavior to our experimental results, we run a version of the model alongside each human participant. Thus, model results are based on 300 simulated move choices in each of the games that human participants played; the agent's decisions in these games is based on reward counts of various prior moves in each game, to ensure that the agent essentially *shared the same experience* as our human players.

The basis for the RL-based agent's move choices is an ongoing count of patterns in previous moves. To compute this, we add columns to our existing data frame that maintain the relevant event counts from earlier rounds in each row. For example, the simplest version of the agent adds columns for `rock_reward`, `paper_reward`, and `scissors_reward`. Next, we iterate through each round of human play and update the reward based on the outcome the agent obtains by playing *Rock*, *Paper*, and *Scissors*. Since for this null model, we merely accumulate rewards of each single move, and the model favors whichever move (`Rock`, `Paper`, `Scissors`) wins the most.

In fact, the patterns that the RL-based agent tracks can be arbitrarily complex. In our [Model Results](RLModel_results.md) on the next page, we evaluate a number of additional models that track different combinations of previous moves using the same process described above. These patterns fall into several distinct categories, described below in increasing complexity:

- **Self-previous move reward**: This RL-based agent tracks the reward pattern of its own *previous move* and current move. This reward pattern allows agent to choose an optimal move given its previous move. In another word, if the agent play `Rock` last time, the agent is going to think which move gives me the highest reward after my `Rock` move. How to achieve this mindset? The model increments the current move's reward based on the current outcome, ie. *win* (3 $points$), *tie* (0 $point$), and *loss* (-1 $point$). (For a reminder of how *reward counts* are formalized in rock, paper, scissors, see the latter [RL_model_reward](RL_model_reward.ipynb) page). This pattern is well-aligned to the opponent-transition bot strategy; but *how* quickly the agent learns to choose their moves, how well it performs against the *other* bots, and whether this closely captures *human* learning against the bot opponents are all open questions (see [Model Results](RLModel_results.md)).

- **Opponent-previous move rewards**: We also implement a version of the RL-based agent that tracks the *opponent's previous move* rewards. This reward pattern allows agent to choose an optimal move given its opponent's previous move. In another word, if the agent play `Paper` last time, the agent is going to think which move gives me the highest reward after my opponent plays `Paper`. How to know the agent's rewards? For example, if the agent wins by playing `Rock` this time, the model increments the `Paper_Rock_reward` by 3; if the agent loses by play `Rock`, the model decrement the `Paper_Rock_reward` by 1. This pattern sounds like bing aligned to the self-transition bot strategy. It allows us to explore how well such reward pattern performs in simulated rounds against all the bot opponents, and whether this aligns with human learning. (see [Model Results](RLModel_results.md))

- **Disjunctive agent-opponent previous move rewards**: In addition to simple *self-* and *opponent-* previous move rewards, this RL-based agent tracks the disjunction of the opponent's past move and human's current move by picking the total higher-rewarded pattern first. To be clarified, the agent keeps both self-previous reward distribution and opponent-previous reward distribution in mind, and when it is going to choose a next move, it decides one favored pattern and pick a favored move from it. How does this selection works? For example, the bot plays `Rock` previously and the agent plays `Paper` previously. Now the agent perceives `Paper_{R, P, S}_rewards` as self-previous reward pattern (3 values), and `Rock_{R, P, S}_rewards` as opponent_previous reward pattern (3 values). Next, the agent sums up each pattern's reward, and choose the higher total reward as the favored pattern. Then following the same procedure as before, the agent will pick a higher-rewarded move from the pattern. Using this strategy to pick move should perform better, because the agent is able to extract the winning information from one of the strategy.

- **Combined agent-opponent previous move rewards**: Finally, the most complex version of the model-based agent tracks the ongoing combined reward count of the opponent past move and human past move. This represents a further increase in complexity from the disjunctive version which only considers one-side information. As such, this level of reward count tracking requires 27 columns to maintain state from one round to the next. Given an example, this patter contains information as `{opponent_previous_move}_{human_previous_move}_{R, P, S}_reward`. While it is unlikely that human explicitly track previous moves on this level of complexity, this allows us to model ideal learning conditions for the most complex RL-agent and determine how such an agent performs by adopting simpler reward counting mechanisms.

## Choosing a move

Each instantiation of the RL-based agent tracks a particular *reward count* or previous moves that it uses as the basis for choosing its optimal next move in a given round. *But how does the agent choose its own move on the basis of this information?*

**Rather than being like the Model-based agent to calculate expected values of the move combination, the RL-based agent probabilistically choosing the move have the highest reward directly.** What does this mean?

The RL-based agent converts the corresponding `Rock, Paper, and Scissors` reward columns to a probability distribution through a calculation of its *softmax probability* and sample a move from the distribution.

**Sample a move using *softmax***: The RL-based agent has tracked rewards based on patterns we defined in different models. Given an example, `rock_rock_reward` in `agent_previous_move` model means human previous move is rock and current move is rock, and this column is from which the softmax probability of `Rock` comes. Knowing the prior move conditions, the agent grabs `{Prior_condition}_{Rock or Paper or Scissors}_reward` columns, and passes the values to the softmax probability calculation. The softmax formula is:

$$
  P(M_a) = \dfrac{e^{\beta E[M_a]}}{\sum_{M_a \in \{R, P, S\}} e^{\beta E[M_a]}}
$$

Then it returns a probability distribution to sample `Rock`, `Paper` and `Scissors` in the current round.

Finally, the RL-model samples a move from the softmax probability distribution. In summary, the move with a higher rewards in a given round are more likely to be chosen. If the agent is successfully able to track the reward of the previous moves, it will have a high probability of choosing the move that *beats* the opponent's next move.

***But how well does the RL-based agent, choosing its moves in simulated rounds through the process described above, perform against the bot opponents from our experiments?***

In the next page ([Model Results](RLModel_results.md)), we test this model's performance in the context of different reward tracking pattern.
