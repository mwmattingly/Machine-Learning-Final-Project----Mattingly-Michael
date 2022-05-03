# Machine-Learning-Final-Project----Mattingly-Michael
This is the final project for CMPSCI4200.  

# Part 1

## Data Discussion
### Here is the link to the Data
[The data being used for this project comes from fivethirtyeight.com](https://github.com/fivethirtyeight/data/tree/master/nhl-forecasts)

In order to draw reasonable conclusions from the data, some assumptions need to be made.
First, the data needs to be restricted to complete seasons with each current team equally represented.
For this the data is filtered to the NHL seasons 2016-2021
A future consideration will be excluding the 2021 season due to the COVID-19 pandemic since games were not played in all NHL stadiums.

## Now let's look at a correlation heat map to see the overall correlation between variables in the dataset.
![](./IMAGES/CorrHeatMap.png)

### Here is a distribution of our proposed target variable: home_team_score
![](./IMAGES/HomeTeamScoreHistogram.png)

We see home_team_score seems to follow a normal distribution which satisfies a requrirement for regression and suggests no transformation is needed.

### Home team score varies by team from 2016-2021 
![](./IMAGES/HomeTeamScoreVariabilityByTeam.png)

The vertical bar chart illustrates variability in the goals per game scored by team.

### The correlation heat map shows some correlation between home_team_pregame_rating and the home_team_score but let's look closer...
If we look at the mean home_team_pregame_rating with the mean home_team_score in the data we see a clear direct linear relationship

![](./IMAGES/AggTeamRatingbyYearMeanGoalsPerGame.png)

We also note a simple non-aggregated view of away team pre game rating is indirectly related with home team score.

![](./IMAGES/CorrPlotAwayRatingHomeGoals.png)

### Home team score also seems to decrease slightly as the probability of Overtime increases

![](./IMAGES/CorrPlotOverTimePrHomeGoals.png)

### Home team score also seems to decrease slightly during a playoff game

![](./IMAGES/CorrPlotPlayoffGameHomeGoals.png)

### Prediction strategy
Based on the initial analysis we can see from the histogram of home_team_score that the target variable (home_team_score) appears to be normally distributed.
We will apply regression to predict the number of goals the home team will score based on home_team_pregame_rating, away_team_pregame_rating, 
who the home team is, who the opponent is, if the game is a playoff game, and if overtime is probable. 

The data will be scaled given the order of magnitude difference between home and away team ratings and other variables.
Character variables for home and away teams will be recoded using dummy or one hot encoding.

Our strategy will be split the data randomly between train and test to ensure we do overfit the mddel.

Finally, additional predictive features to explore (time permitting) will be moving averages and lag variables for goals per game and prior game goals scored.

# Part 2 Regression Problem

## 1 Distribution of terget values

## The target variable is home_team_score or goals scored by the home team for a given game
## The target is normally distributed, has a mean of X and is range-bounded between Y and Z.

## 2 There are no NA or missing values

## 3 Decided to keep overtime and team abbreviation 

## 4 Using dummy coding since the values are ordinal not hierarchical

## 5 Aggregation columns

## 6 Kept overtime and team abbreviation due to...

## 7 Split into x_data and y_data

## 8 Used scaling normalization...

## 9 Used a 90/10 split (90% train, 10% test)

## 2 Training

## 1 Using regression, import linear and logistic regression

## 2 Fit/train models

## 3 Post Model Analysis

## 1 Use model.score on test to get idea of accuracy

## 2 Here is a comparison of accuracy between models

## 3 Change the split between test and train to 80/20, 70/30, 60/40, 50/50,...

## 4 Use residuals library to plot residuals

## 5 Classification 

## 6 Which model is better 



