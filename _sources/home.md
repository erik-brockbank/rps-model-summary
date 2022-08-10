# Modeling Adaptive Reasoning in 🪨 📜 ✂️

## Welcome!

This is the official home page for a multi-year project building ***models of how people learn from and adapt to their opponents*** when playing Rock, Paper, Scissors (or Roshambo). The bulk of the code was completed by UCSD students Betty Gong, MJ Mei, Annalea O'Halloran, and Alison Yu. The project was spearheaded by Erik Brockbank using data originally collected by Erik Brockbank and Edward Vul at the UCSD [Computational Cognition Lab](https://www.evullab.org/). Read on to learn more!


## Why this matters

Humans are excellent pattern learners. As children, our ability to learn language and social norms relies on detecting rich and nuanced structure in our environment. As adults, we rely on these abilities in everyday settings as well as for complex behaviors like testing scientific theories. One area where this is especially crucial is in **adversarial settings** where we're trying to *outwit* an opponent.

From sports and games to courtroom drama and international relations, people have a unique ability to adapt in adversarial interactions. Consider, for example, this interview with tennis great Andre Agassi, in which he describes realizing that his opponent Boris Becker would give away his serve direction by [sticking out his tongue](https://www.youtube.com/watch?v=57BMzCM6hQI) in a particular direction. Or, this more recent article about the [strategic signalling](https://www.nytimes.com/2022/02/15/world/europe/us-russia-ukraine-war.html) between the US and Russia in Ukraine.

***What kind of reasoning forms the basis for recognizing patterns in adversarial interactions and what are the limits in this ability?***

To try and answer this, we explore people's behavior in a *much simpler setting* than elite tennis or international relations. We use the game of Rock, Paper, Scissors (RPS) to understand what kinds of behavioral patterns people can recognize and adapt to in order to outwit their opponent. Specifically, people play *hundreds of rounds* (usually 300) against another human or a bot. In this setting, RPS is a perfect *model system* for studying adversarial interactions. The rules are simple and people don't have a lot of prior expertise (this differentiates it from something like chess), so the best way to beat your opponent is to uncover patterns in their moves that allow you to predict their next move (for example, people will often play the same move again after winning...). In this project and elsewhere, we ask what kinds of patterns people can detect in their opponent's moves, what kinds of patterns they can revise in their own moves, and what kind of learning processes underly this behavior.

***On this site, we present one particular thread in this effort: a large-scale modeling project that compares several different computational models of how people adapt to stable patterns in an opponent's moves.***

The pages on this site present a summary of our results.

## Project Overview

The site is divided into the following sections:

1. The **SUMMARY** section provides first a high-level summary of the approach and the findings on the [Overview](Overview.md) page, then a longer explanation of the background for this work on the [Introduction](Introduction.md) page and the data we use for the project on the [Data](Data.ipynb) page.

[//]: # "TODO add videos of kids/adults playing RPS (can you embed media in markdown?)"
