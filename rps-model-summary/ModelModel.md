# Model Overview

***At a high-level, the Model-based Agent simulates what would happen if human players were trying to predict their bot opponent's next move on the basis of particular patterns in their previous moves.***

In this way, the Model-based Agent represents a kind of *idealized* model of human behavior that serves as a basis for evaluating what people are *actually* doing.

In order to simulate human play against the different bot opponents, the Model-based Agent has two high-level functions (we spell these out in more detail below):

1. **Track behavior patterns**: First, the agent maintains a count of *events* that it uses to try and predict the opponent's next move. To illustrate, the simplest version of this is an ongoing count of how many times the opponent played `Rock`, `Paper`, and `Scissors` in previous rounds. In each round, these counts allow for a (very rough) prediction that an opponent will choose whichever move was most likely across the previous rounds.

2. **Choose an optimal move**: The behavior tracking above allows the agent to generate predictions in each round about the opponent's most likely move based on patterns observed in the previous rounds. The Model-based Agent must then use these predictions to make its own move choice. It samples a move each round probabilistically based on how well each possible move will perform against the move it thinks its opponent will make.

***How well does the Model-based Agent described above capture human patterns of learning against the different bot opponents?***

To answer this question, we implement different versions of the model and compare them to the human behavior in our experiment data. Each version of the model has the same *decision function* from step 2 above but differs in the underlying information that it tracks about its opponent (step 1). In this way, we simulate different levels of *opponent modeling* that people might be engaging in when playing against algorithmic opponents.


# Model Details

Below, we describe in more detail the two central aspects of the Model-based Agent's behavior.

*The content in this section is meant to provide enough detail for readers to inspect the underlying code or think carefully about the [Model Results](ModelModel_results.md) on the next page.*

For an apples-to-apples comparison of model behavior to our experimental results, each variant of the model simulates all of our human participants in the original experiment. Thus, model results are based on 300 agent-supplied move choices in each of the games that human participants played; the agent's decisions in these games are based on the bot opponent's prior moves in each game, to ensure that the agent *shares the same experience* as our human players.

## Tracking opponent behavior

The basis for the Model-based Agent's move choices is an ongoing count of patterns in the opponent's previous moves. To compute this, we add columns to our existing data frame that maintain the relevant event counts from earlier bot decisions. For example, the simplest version of the agent adds columns for `opponent_rock_count`, `opponent_paper_count`, and `opponent_scissors_count` to track the bot opponent's move counts. We populate them by iterating through each round of human play and updating the values in each column based on the cumulative number of times that human's opponent displayed the given pattern, for example playing `Rock`, `Paper`, and `Scissors`. The values across these columns in a given row of our dataframe represent an idealized (and very simple) model of opponent behavior that can be used to make predictions about the opponent's next move.

While the opponent's cumulative count of each move constitutes an overly simplistic basis for prediction against the bot opponents, the patterns that the Model-based Agent tracks can be arbitrarily complex. In our [Model Results](ModelModel_results.md) on the next page, we evaluate a number of additional models that track different patterns in the bot opponent moves using the same process described above. These patterns fall into several distinct categories, described below in increasing complexity:

- **Self-transitions**: The Model-based Agent tracks the opponent's *self-transition* counts by adding columns which tally the cumulative number of *up* ($+$), *down* ($-$), and *stay* ($0$) self-transitions that the opponent has made from one round to the next (for a reminder of how *transitions* are formalized in RPS, see the previous [Data](Data.ipynb) page). This count is obviously well-aligned to the self-transition bot opponents; but *how* quickly it learns to predict their moves, how well it performs against the *other* bots, and whether this closely captures *human* learning against the bot opponents are all open questions (see [Model Results](ModelModel_results.md)).

- **Opponent-transitions**: We also implement a version of the Model-based Agent that tracks each bot's *opponent-transition* counts. This requires a cumulative tally in each round of the opponent's *up* ($+$), *down* ($-$), and *stay* ($0$) opponent-transitions. This state information is optimal against the opponent-transition bots; as with the above, this allows us to explore how well such tracking performs in simulated rounds against *all* the bot opponents and whether this aligns with human learning.

- **Outcome-based transitions**: In addition to simple *self-* and *opponent-* transitions, the Model-based Agent tracks transitions *contingent on prior round outcomes*. Rather than merely tallying the count of *up* ($+$), *down* ($-$), and *stay* ($0$) transitions, these counts are updated *for each possible previous outcome* (*win*, *tie*, and *loss*). Maintaining and updating these counts therefore requires nine additional columns in the data (counts of *win-up-transition*, *win-down-transition*, *win-stay-transition*, etc.). Using this tally for Model-based Agent decision making ought to perform optimally against the outcome-dependent transition bots (i.e., those using *win-stay, lose-shift* strategies), *as well as* those using simple self-transition and opponent-transition strategies (these are essentially a subset of the outcome-transition strategies).

- **Transitions based on the previous outcome and previous transition**: Finally, the most complex version of the Model-based Agent tracks the ongoing count of the opponent's self-transitions contingent on each combination of a) the previous round outcome (as above) *and* b) the opponent's *previous* transition. This represents a further increase in complexity from the above version which only tracks potential patterns in the transitions that follow each outcome. As such, this level of opponent behavior tracking requires 27 columns to maintain state from one round to the next. While it is unlikely that people playing bot opponents would explicitly track patterns at this level of complexity, this allows us to model ideal learning conditions for the most complex bot opponent and determine how such an agent performs against simpler opponents such as the outcome-based transition bots.


**In summary, the Model-based Agent's *opponent tracking* process involves updating cumulative counts of prior events (such as the opponent's moves or transitions) that might allow it to better predict its opponent's next move in a given round.** We implement several different *classes* of opponent tracking which correspond to distinct and increasingly complex sequential patterns in opponent behavior that the agent uses to select a move.

*In the next section, we walk through how the Model-based Agent uses this information about its opponent's prior actions to choose its own move in simulated rounds against the bot opponents.*


## Choosing a move

Each instantiation of the Model-based Agent tracks a particular *sequential dependency* or pattern in its opponent moves that it uses as the basis for predicting its opponent's next move in a given round. *But how does the agent choose its own move on the basis of this information?*

**The Model-based Agent has a *decision policy* of probabilistically choosing the move that will perform best against the move it expects from its opponent each round.** What does this mean?

There are two primary steps the Model-based Agent undertakes to select a move: an *expected value calculation* and a *softmax move sampling* process.

**Expected value calculation**: First, the agent calculates the *expected value* of each possible move it could select (`Rock`, `Paper`, or `Scissors`) in the next round. The expected value of a move is the sum of all possible *outcomes* from playing that move weighted by the probability of those outcomes. For example, the expected value of the agent move choice $M_a$ of `Rock` ($M_a = R$) can be written as:

$$
  E[M_a = R] = \sum_{M_o \in \{R, P, S\}} U(M_a, M_o)P(M_o)
$$

In the above formulation, $M_o$ is the *opponent's potential move choice* ($R$, $P$, or $S$). $U$ is the *reward that the agent receives* for the combination of its own move choice $M_a$ and its opponent's move choice $M_o$. In the example above, this would be the points the agent receives for each possible opponent move choice $M_o \in \{R, P, S\}$ when the agent plays $M_a = R$: 3 points for a win (if $M_o=S$), 0 points for a tie ($M_o=R$), -1 points for a loss ($M_o=P$). Finally, $P(M_o)$ is the probabily assigned to a particular opponent move choice $M_o$. The probability of each possible opponent move choice is precisely what the agent estimates in the previous section through its opponent tracking!

**Sample a move based on its *softmax probability***: The Model-based Agent estimates an expected value (*EV*) for each possible move it could play using the process outlined above, based on the probabilities it assigns to its opponent's moves. However, rather than merely choosing the move with the highest EV each round, the agent chooses its move probabilistically *in proportion to the expected value of each possible move*. In other words, if one possible move has a *much higher* expected value than the others, then the agent should strongly favor this move; for example, if the agent believes the opponent is *all but guaranteed to play `Scissors`*, then `Rock` will have a dramatically larger EV and the agent should correspondingly favor `Rock` strongly. If, however, all of the candidate moves are equally good (as would be the case if the agent believes its opponent is *equally likely* to play `Rock`, `Paper`, or `Scissors`), then it should assign them all roughly equal probabilities when choosing its own move. In order for the Model-based Agent to choose its moves *in proportion to their relative EVs*, we map the expected value of each possible move calculated above to a probability of its being chosen using the *softmax* function:

$$
  P(M_a) = \dfrac{e^{\beta E[M_a]}}{\sum_{M_{a'} \in \{R, P, S\}} e^{\beta E[M_{a'}]}}
$$

In the above, the probability of the agent choosing a move $P(M_a)$ is a function of the expected value of that move $E[M_a]$, scaled by the expected value of *all possible moves* $M_{a'}$. The $\beta$ term in the numerator and denominator is a common parameter used to scale *how much the agent should favor the highest expected value move*. In this version of the model, we simply set this to 1. However, in another line of work we estimate a *fitted $\beta$ parameter* for each participant as a way of estimating how well the Model-based Agent describes human move decisions.

Once the Model-based Agent has transformed the expected value of each possible move into a probability distribution over those moves using the softmax function, it samples a move for that round according to its softmax probability. Thus, moves with a higher expected value in a given round are more likely to be chosen. More generally, if the agent is successfully able to predict its opponent's *next* move using the sequential pattern it tracks in its opponent's previous moves, then it will have a high probability of choosing the move that *beats* the predicted move.

***But how well does the Model-based Agent, choosing its moves in simulated rounds through the process described above, perform against the bot opponents from our experiments?***

In the next page ([Model Results](ModelModel_results.md)), we test this model's ability to adapt to each of the bot opponents in our experiment based on different levels of opponent behavior tracking.
