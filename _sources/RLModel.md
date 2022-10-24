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

The basis for the RL-based agent's move choices is an ongoing count of patterns in previous moves. To compute this, we add columns to our existing data frame that maintain the relevant event counts from earlier rounds in each row. For example, the simplest version of the agent adds columns for `rock_count`, `paper_count`, and `scissors_count`. Next, we iterate through each round of human play and update the values in each column based on the cumulative number of times that human played *Rock*, *Paper*, and *Scissors*. The values across these columns in a given row of our dataframe represent an idealized (but very simple) model of human behavior that can be used to make (rough) the most optimal choice for the next round.

While the agent's count of each previous move constitutes an overly simplistic basis for the optimal next move against the bot opponents, the patterns that the RL-based agent tracks can be arbitrarily complex. In our [Model Results](RLModel_results.md) on the next page, we evaluate a number of additional models that track different combinations of previous moves using the same process described above. These patterns fall into several distinct categories, described below in increasing complexity:

- **Self-previous move reward**: The RL-based agent tracks its own *previous reward* counts by adding columns which tally the cumulative number of *win* ($3 points$), *tie* ($0 point$), and *loss* ($-1 point$) reward points that the agent itself has made from one round to the next (for a reminder of how *reward counts* are formalized in rock, paper, scissors, see the latter [RL_model_reward](RL_model_reward.ipynb) page). This count is obviously well-aligned to the self-previous move reward agent; but *how* quickly it learns to choose their moves, how well it performs against the *other* bots, and whether this closely captures *human* learning against the bot opponents are all open questions (see [Model Results](RLModel_results.md)).

- **Opponent-previous move rewards**: We also implement a version of the RL-based agent that tracks the opponent's previous move rewards. This requires a cumulative tally in each round of the opponent's *win* ($3 points$), *tie* ($0 point$), and *loss* ($-1 point$) reward points. This state information is optimal against the opponent-transition bots; as with the above, this allows us to explore how well such tracking performs in simulated rounds against all the bot opponents and whether this aligns with human learning.

- **Disjunctive agent-opponent previous move rewards**: In addition to simple *self-* and *opponent-* previous move rewards, the RL-based agent tracks the disjunction of the opponent's past move and human's current move *contingent on prior round outcomes*. Rather than merely tallying the count of *win* ($3 points$), *tie* ($0 point$), and *loss* ($-1 point$) reward points , these counts are updated *for each possible previous outcome* (*win*, *tie*, and *loss*). Maintaining and updating these counts therefore requires nine additional columns in the data instead of three as with the above. Using this tally for RL-based agent decision making ought to perform better since it takes both the previous moves of human and opponents into consideration. 

- **Combined agent-opponent previous move rewards**: Finally, the most complex version of the model-based agent tracks the ongoing combined reward count of the opponent past move and human past move. This represents a further increase in complexity from the above version which only tracks the reward counts of the disjunction of the opponent's past move and human's current move. As such, this level of reward count tracking requires 27 columns to maintain state from one round to the next. While it is unlikely that people playing bot opponents would explicitly track previous moves on this level of complexity, this allows us to model ideal learning conditions for the most complex RL-agent and determine how such an agent performs by adopting simpler reward counting mechnisms. 

## Choosing a move

Each instantiation of the RL-based agent tracks a particular *reward count* or previous moves that it uses as the basis for choosing its optimal next move in a given round. *But how does the agent choose its own move on the basis of this information?*

**The RL-based agent has a *decision policy* of probabilistically choosing the move that will perform best based on the move that has the highest probability of winning in previous rounds.** What does this mean?

There are two primary steps the RL-based agent undertakes to select a move: and *expected value calculation* and a *softmax move sampling* process.

**Expected value calculation**: First, the agent calculates the *expected value* of each possible move it could take (*Rock*, *Paper*, or *Scissors*) in the next round. The expected value of a move is the sum of all possible possible *outcomes* from playing that move weighted by the probability of those outcomes. For example, the expected value of the agent move choice $M_a$ of *Rock* ($M_a = `R'$) can be written as:

$$
  E[M_a = `R'] = \sum_{M_o \in \{`R', `P', `S'\}} U(M_a, M_o)P(M_o)
$$

In the above formulation, $M_o$ is the *opponent's potential move choice* ($`R'$, $`P'$, or $`S'$). $U$ is the *reward that the agent receives* for the combination of its own move choice $M_a$ and its opponent's move choice $M_o$. In the example above, this would be the points the agent receives for each possible opponent move choice $M_o \in \{`R', `P', `S'\}$ when the agent plays $M_a = `R'$: 3 points for a win (if $M_o=`S'$), 0 points for a tie ($M_o=`R'$), -1 points for a loss. Finally, $P(M_o)$ is the probabily assigned to a particular opponent move choice $M_o$. The probability of each possible opponent move choice is precisely what the agent estimates in the previous section through its opponent tracking!

**Sample a move using *softmax***: The RL-based agent estimates an expected value (*EV*) for each possible move it could play using the process outlined above, based on the probabilities it assigns to its opponent's moves. The agent's task is to choose the move that has the highest expected value. However, rather than merely choosing the move with the highest expected value each round, the agent chooses its move probabilistically *in proportion to the expected value of each possible move*. In other words, if one possible move has a *much higher* expected value than the others, then the agent should strongly favor this move; for example, if the agent believes the opponent is *all but guaranteed to play Scissors*, then *Rock* will have a dramatically larger EV and the agent should correspondingly favor *Rock* strongly. If, however, all of the candidate moves are equally good (as would be the case if the agent believes its opponent is *equally likely* to play *Rock*, *Paper*, or *Scissors*), then it should assign them all roughly equal probabilities when choosing its own move. In order for the model-based agent to choose its moves *in proportion to their relative EVs*, we map the expected value of each possible move calculated above to a probability of its being chosen using the *softmax* function:

$$
  P(M_a) = \dfrac{e^{\beta E[M_a]}}{\sum_{M_a' \in \{`R', `P', `S'\}} e^{\beta E[M_a']}}
$$

In the above, the probability of the agent choosing a move $P(M_a)$ is $e$ raised to the expected value of that move $E[M_a]$ as described above, divided by the sum of $e$ raised to the expected value of *all possible moves* $M_a'$. The $\beta$ term in the numerator and denominator is a common parameter used to scale *how much the agent should favor the highest expected value move*. In this version of the model, we simply set this to 1. However, our [Model Comparison](ModelComparison.md) estimates a *fitted $\beta$ parameter* for each participant as a way of estimating how well the RL-based agent describes human move decisions.

Once the RL-based agent has transformed the expected value of each possible move into a probability distribution over those moves using the softmax function, it samples a move for that round according to its softmax probability. Thus, moves with a higher expected value in a given round are more likely to be chosen. More generally, if the agent is successfully able to track the reward counts of the previous moves, then it will have a high probability of choosing the move that *beats* the opponent's next move.

***But how well does the RL-based agent, choosing its moves in simulated rounds through the process described above, perform against the bot opponents from our experiments?***

In the next page ([Model Results](RLModel_results.md)), we test this model's performance in the context of different reward counts tracking.
