# Machine-Learning-Final-Project----Mattingly-Michael
This is the final project for CMPSCI4200.  

## Data Discussion
[The data being used for this project comes from fivethirtyeight.com](https://github.com/fivethirtyeight/data/tree/master/nhl-forecasts)

In order to draw reasonable conclusions from the data, some assumptions need to be made.
First, the data needs to be restricted to complete seasons with each current team equally represented.
For this the data is filtered to the NHL seasons 2016-2021
A future consideration will be excluding the 2021 season due to the COVID-19 pandemic since games were not played in all NHL stadiums.

## Now let's look at a correlation heat map to see the overall correlation between variables in the dataset.
![](./CorrHeatMap.png)

### Here is a distribution of our proposed target variable: home_team_score
![](./HomeTeamScoreHistogram.png)

We see home_team_score seems to follow a normal distribution which satisfies a requrirement for regression and suggests no transformation is needed.

### Home team score varies by team from 2016-2021 
![](./HomeTeamScoreVariabilityByTeam.png)

The vertical bar chart illustrates variability in the goals per game scored by team.

### The correlation heat map shows some correlation between home_team_pregame_rating and the home_team_score but let's look closer...
If we look at the mean home_team_pregame_rating with the mean home_team_score in the data we see a clear direct linear relationship

![](./AggTeamRatingbyYearMeanGoalsPerGame.png)

We also note a simple non-aggregated view of away team pre game rating is indirectly related with home team score.

![](./CorrPlotAwayRatingHomeGoals.png)

### Home team score also seems to decrease slightly as the probability of Overtime increases

![](./CorrPlotOverTimePrHomeGoals.png)

### Home team score also seems to decrease slightly during a playoff game

![](./CorrPlotPlayoffGameHomeGoals.png)

### Prediction strategy
Based on the initial analysis we can see from the histogram of home_team_score that the target variable (home_team_score) appears to be normally distributed.
We will apply regression to predict the number of goals the home team will score based on home_team_pregame_rating, away_team_pregame_rating, 
who the home team is, who the opponent is, if the game is a playoff game, and if overtime is probable. 

The data will be scaled given the order of magnitude difference between home and away team ratings and other variables.
Character variables for home and away teams will be recoded using dummy or one hot encoding.

Our strategy will be split the data randomly between train and test to ensure we do overfit the mddel.

Finally, additional predictive features to explore (time permitting) will be moving averages and lag variables for goals per game and prior game goals scored.
