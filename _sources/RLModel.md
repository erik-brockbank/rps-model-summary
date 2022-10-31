# Model Overview

***At a high-level, the RL-based agent simulates what would happen if (human) players were trying to determine their next optimal move on the basis of particular rewards associated with different past events (patterns in agent's previous moves).***

In this way, the RL-based agent represents a kind of *idealized* model of human behavior that serves as a basis for evaluating what people are *actually* doing.

In order to simulate human play against the different bot opponents, the RL-based agent has two high-level functions (we spell these out in more detail below):

1. **Track reward patterns**: First, the agent maintains a count of *events* that it uses to try and predict the the reward of its next move. To illustrate, the simplest version of this is an ongoing count of how many times the agent has won by playing *Rock*, *Paper*, and *Scissors* in previous rounds. In each round, these counts allow for a (very rough) prediction that the agent will choose whichever move that generated a win on the previous round.

2. **Choose an optimal move**: The behavior tracking above allows the agent to generate predictions in each round about its most logical next move based on the reward counts observed in the previous rounds. The RL-based agent must then use these patterns to make its own move choice. It samples a move each round probabilistically based on how well each possible move performed in the past would beat the move it thinks its opponent will make.

***How well does an RL-based model using the steps above capture human patterns of learning against the different bot opponents?***

To answer this question, we implement different versions of the model and compare them to the human behavior in our experiment data. Each version of the model has the same *decision function* from step 2 above but differs in the underlying information that it tracks about different combination of reward counts (step 1). In this way, we simulate different levels of *reward tracking* that people might be engaging in when playing against algorithmic opponents.

# Model Details

Below, we describe in more detail the two central aspects of the model-based agent's behavior outlined above.

*The content in this section is meant to provide enough detail for readers to inspect the underlying code or think carefully about the [Model Results](RLModel_results.md) on the next page.*


## Tracking behavior associated rewards

For an apples-to-apples comparison of model behavior to our experimental results, we run a version of the model alongside each human participant. Thus, model results are based on 300 simulated move choices in each of the games that human participants played; the agent's decisions in these games is based on reward counts of various prior moves in each game, to ensure that the agent essentially *shared the same experience* as our human players.

The basis for the RL-based agent's move choices is an ongoing count of patterns in previous moves. To compute this, we add columns to our existing data frame that maintain the relevant event counts from earlier rounds in each row. For example, the simplest version of the agent adds columns for `rock_reward`, `paper_reward`, and `scissors_reward`. Next, we iterate through each round of human play and update the values in each column based on the cumulative number of times that human played *Rock*, *Paper*, and *Scissors*. The values across these columns in a given row of our dataframe represent an idealized (but very simple) model of human behavior that can be used to make (rough) the most optimal choice for the next round.

While the agent's count of each previous move constitutes an overly simplistic basis for the optimal next move against the bot opponents, the patterns that the RL-based agent tracks can be arbitrarily complex. In our [Model Results](RLModel_results.md) on the next page, we evaluate a number of additional models that track different combinations of previous moves using the same process described above. These patterns fall into several distinct categories, described below in increasing complexity:

- **Self-previous move reward**: The RL-based agent tracks its own *previous reward* counts by adding columns which tally the reward of *win* ($3 points$), *tie* ($0 point$), and *loss* ($-1 point$) reward points that the agent itself has made from one round to the next (for a reminder of how *reward counts* are formalized in rock, paper, scissors, see the latter [RL_model_reward](RL_model_reward.ipynb) page). This count is obviously well-aligned to the self-previous move reward agent; but *how* quickly it learns to choose their moves, how well it performs against the *other* bots, and whether this closely captures *human* learning against the bot opponents are all open questions (see [Model Results](RLModel_results.md)).

- **Opponent-previous move rewards**: We also implement a version of the RL-based agent that tracks the opponent's previous move rewards. This requires a cumulative tally in each round of the opponent's *win* ($3 points$), *tie* ($0 point$), and *loss* ($-1 point$) reward points. This state information is optimal against the opponent-transition bots; as with the above, this allows us to explore how well such tracking performs in simulated rounds against all the bot opponents and whether this aligns with human learning.

- **Disjunctive agent-opponent previous move rewards**: In addition to simple *self-* and *opponent-* previous move rewards, the RL-based agent tracks the disjunction of the opponent's past move and human's current move *contingent on prior round outcomes*. Rather than merely tallying the reward of *win* ($3 points$), *tie* ($0 point$), and *loss* ($-1 point$) reward points , these counts are updated *for each possible previous outcome* (*win*, *tie*, and *loss*). Maintaining and updating these counts therefore requires nine additional columns in the data instead of three as with the above. Using this tally for RL-based agent decision making ought to perform better since it takes both the previous moves of human and opponents into consideration.

- **Combined agent-opponent previous move rewards**: Finally, the most complex version of the model-based agent tracks the ongoing combined reward count of the opponent past move and human past move. This represents a further increase in complexity from the above version which only tracks the reward counts of the disjunction of the opponent's past move and human's current move. As such, this level of reward count tracking requires 27 columns to maintain state from one round to the next. While it is unlikely that people playing bot opponents would explicitly track previous moves on this level of complexity, this allows us to model ideal learning conditions for the most complex RL-agent and determine how such an agent performs by adopting simpler reward counting mechanisms.

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
