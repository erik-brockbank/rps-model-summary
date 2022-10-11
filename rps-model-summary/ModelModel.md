# Model Overview

***At a high-level, the model-based agent simulates what would happen if (human) players were trying to predict their (bot) opponent's next move on the basis of particular patterns in their previous moves.***

In this way, the model-based agent represents a kind of *idealized* model of human behavior that serves as a basis for evaluating what people are *actually* doing.

In order to simulate human play against the different bot opponents, the model-based agent has two high-level functions (we spell these out in more detail below):

1. **Track behavior patterns**: First, the agent maintains a count of *events* that it uses to try and predict the opponent's next move. To illustrate, the simplest version of this is an ongoing count of how many times the opponent played *Rock*, *Paper*, and *Scissors* in previous rounds. In each round, these counts allow for a (very rough) prediction that an opponent will choose whichever move was most likely across the previous rounds.

2. **Choose an optimal move**: The behavior tracking above allows the agent to generate predictions in each round about the opponent's most likely move based on patterns observed in the previous rounds. The model-based agent must then use these predictions to make its own move choice.

***How well does an agent-based model using the steps above capture human patterns of learning against the different bot opponents?***

To answer this question, we implement different versions of the model and compare them to the human behavior in our experiment data. Each version of the model has the same *decision function* from step 2 above but differs in the underlying information that it tracks about its opponent (step 1). In this way, we simulate different levels of *opponent tracking* that people might be engaging in when playing against algorithmic opponents.


# Model Details

Below, we describe in more detail the two central aspects of the model-based agent's behavior outlined above.

*The content in this section is meant to provide enough detail for readers to inspect the underlying code or think carefully about the [Model Results](ModelModle_results.md) on the next page.*


## Tracking opponent behavior

For an apples-to-apples comparison of model behavior to our experimental results, we run a version of the model alongside each human participant. Thus, model results are based on 300 simulated move choices in each of the games that human participants played; the agent's decisions in these games is based on the bot opponent's prior moves in each game, to ensure that the agent essentially *shared the same experience* as our human players.

The basis for the model-based agent's move choices is an ongoing count of patterns in the opponent's previous moves. To compute this, we add columns to our existing data frame that maintain the relevant event counts from earlier rounds in each row. For example, the simplest version of the agent adds columns for `opponent_rock_count`, `opponent_paper_count`, and `opponent_scissors_count`. Next, we iterate through each round of human play and update the values in each column based on the cumulative number of times that human's opponent played *Rock*, *Paper*, and *Scissors*. The values across these columns in a given row of our dataframe represent an idealized (but very simple) model of opponent behavior that can be used to make (rough) predictions about the opponent's next move.

While the opponent's cumulative count of each move constitutes an overly simplistic basis for prediction against the bot opponents, the patterns that the model-based agent tracks can be arbitrarily complex. In our [Model Results](ModelModel_results.md) on the next page, we evaluate a number of additional models that track different patterns in the bot opponent moves using the same process described above. These patterns fall into several distinct categories, described below in increasing complexity:

- **Self-transitions**: The model-based agent tracks the opponent's *self-transition* counts by adding columns which tally the cumulative number of *up* ($+$), *down* ($-$), and *stay* ($0$) self-transitions that the opponent has made from one round to the next (for a reminder of how *transitions* are formalized in rock, paper, scissors, see the previous [Data](Data.ipynb) page). This count is obviously well-aligned to the self-transition bot opponents; but *how* quickly it learns to predict their moves, how well it performs against the *other* bots, and whether this closely captures *human* learning against the bot opponents are all open questions (see [Model Results](ModelModel_results.md)).

- **Opponent-transitions**: We also implement a version of the model-based agent that tracks the opponent's *opponent-transition* counts. This requires a cumulative tally in each round of the opponent's *up* ($+$), *down* ($-$), and *stay* ($0$) opponent-transitions. This state information is optimal against the opponent-transition bots; as with the above, this allows us to explore how well such tracking performs in simulated rounds against all the bot opponents and whether this aligns with human learning.

- **Outcome-based transitions**: In addition to simple *self-* and *opponent-* transitions, the model-based agent tracks transitions *contingent on prior round outcomes*. Rather than merely tallying the count of *up* ($+$), *down* ($-$), and *stay* ($0$) transitions, these counts are updated *for each possible previous outcome* (*win*, *tie*, and *loss*). Maintaining and updating these counts therefore requires nine additional columns in the data instead of three as with the above. Using this tally for model-based agent decision making ought to perform optimally against the outcome-dependent transition bots (i.e., those using *win-stay, lose-shift* strategies), *as well as* those using simple self-transition and opponent-transition strategies (these are essentially a subset of the outcome-transition strategies).

- **Transitions based on the previous outcome and previous transition**: Finally, the most complex version of the model-based agent tracks the ongoing count of the opponent's self-transitions contingent on each combination of a) the previous round outcome (as above) *and* b) the opponent's *previous* transition. This represents a further increase in complexity from the above version which only tracks potential patterns in the transitions that follow each outcome. As such, this level of opponent behavior tracking requires 27 columns to maintain state from one round to the next. While it is unlikely that people playing bot opponents would explicitly track patterns at this level of complexity, this allows us to model ideal learning conditions for the most complex bot opponent and determine how such an agent performs against simpler opponents such as the outcome-based transition bots.


**In summary, the model-based agent's *opponent tracking* process involves updating cumulative counts of prior events (such as the opponent's moves or transitions) that might allow it to better predict its opponent's next move in a given round.** We implement several different *classes* of opponent tracking which correspond to distinct and increasingly complex sequential patterns in opponent behavior that the agent uses to select a move.

*In the next section, we walk through how the model-based agent uses this information about its opponent's prior actions to choose its own move in simulated rounds against the bot opponents.*


## Choosing a move

BLAH

1. Compute expected value of each move using probabilities above

2. Sample from Softmax over EV of each move






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
