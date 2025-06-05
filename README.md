This code is intended to analyze data from the files

board_results.csv
HandID,ContractLevel,TrumpSuit,Doubled,Direction,TricksTaken

hands.csv
HandID,NS,NH,ND,NC,ES,EH,ED,EC,SS,SH,SD,SC,WS,WH,WD,WC

This data was harvested and cleaned from ACBL Live Clubs results; board_results.csv has about 100 million rows, while hands.csv has about 9 million rows.

For a brief description of bridge to give context to this project, all 52 cards in a standard deck are dealt, 13 to each player. The hands are denoted by the cardinal direction that the player is sitting at the table, North, East, South, or West (typically denoted N, E, S, W). These hands can be found in hands.csv with a unique HandID
Each HandID is then played by multiple distinct groups of four people; the results of their bidding and play are recorded in board_results.csv
ContractLevel denotes the number of tricks that the players "promised" to take in the bidding
TrumpSuit denotes the suit determined to be trump (the most powerful suit) during the bidding.
If the contract is doubled (the opponents believe that the contract was doomed to fail), that is denoted in Doubled
The direction of the contract (whether the NS team or the EW team won the contract) is denoted in Direction
The number of tricks that the contract actually took is in TricksTaken

A very big question in bridge is given the 13 cards in a single hand, can one come up with a simple way of valuing the hand (assigning the number of tricks you expect your hand to be able to take). The simplicity is important, as all calculations are typically done in the players head, so advanced arithmetic or 
complicated functions are impractical for the average player to use.

As a very brief index of the files:
main.py - contains the lines to be run; this has most of the utility code to make sure that other files remain clean and that I don't accidentally re-run the creation of a large file
hand_valuation.py - this represents different ways of valuing a hand and contains the interface that should be used for a valuation technique for other files. This also creates HandFeature files, which track the features used by a hand valuation system.
mean_squared_error.py - this computes the mean_squared_error of a particular HandValuation system
regression.py - this allows for the construction of a hand valuation system using a regression on a HandFeature file
selector.py - this includes functionality to determine which contracts should be included in a regression or error calculation. By default Club and Diamond contracts are excluded because standard bidding systems cause them to be inconsistent with a standard valuation system's goals
bundler.py - this is an auxiliary class that groups similar contracts (by default this is only by trump suit)
variance_calculator.py - this is a computationally heavy class that estimates the variance of a bundle of contracts; by default this is all contracts with the same trump suit for the same hand. Note that this is important in order to weight bundles correctly, as the true variance of bundles should differ across hands and contracts. Using sample variance would be inaccurate for bundles with small sample size, so empirical bayesian techniques are used instead. This uses pre-generated monte carlo models (they are large and thus not included here, and the generating code is not yet cleaned and nicely documented. Adding the generating code is on my TODO list)
data_visualizer.py - this is a class for many of the preliminary visualizations that I used in determining models for the variance_calculator and regressions
