# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 08:47:44 2022

@author: Michael
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn import preprocessing
import random

"Import csv file"
df = pd.read_csv('C:/Users/Michael/Downloads/nhl_elo.csv')

" Look at data types"
print(df.dtypes)

" Get basic information and description of dataset"
df.info()
df.describe()

" Make a copy of the data set"
cdf = df.copy()

" Filter out seasons prior to 2016 season to ensure all active teams are included" 
cdf = cdf[(cdf['season'] >= 2016) & (cdf['season'] <= 2021)]

" Create Correlation heatMap"
sns.heatmap(cdf.corr())

" Examime the distribution of home_team_score "
cdf.hist(column='home_team_score')

" Check for variability in home team score by team"
cdf['MeanHomeTeamScore'] = cdf.groupby(['home_team_abbr'])['home_team_score'].transform('mean')
cdf = cdf.sort_values(by=['MeanHomeTeamScore'], ascending=True)

plt.figure(figsize=(10,6))        
sns.barplot(x = 'home_team_abbr',
            y = 'MeanHomeTeamScore',
            data = cdf)
plt.xlabel("Home Team", size=15)
plt.ylabel("Average Score per Game 2016-2021", size=15)
plt.title("Variability of goals per game at home by NHL Team: 2016-2021", size=18)
plt.tight_layout()
plt.show()

" Examine the aggregate relationship between average home_team_score and team rating by team"
cdf['HomeTeamRatingByYear'] = cdf.groupby(['season','home_team_abbr'])['home_team_pregame_rating'].transform('mean')
sns.regplot(x=cdf["HomeTeamRatingByYear"], y=cdf["HomeTeamScoreByYear"])

cdf['AwayTeamRatingByYear'] = cdf.groupby(['season','home_team_abbr'])['away_team_pregame_rating'].transform('mean')
sns.regplot(x=cdf["AwayTeamRatingByYear"], y=cdf["HomeTeamScoreByYear"])

" Examine relationship between home team pre-game rating and home team score"
sns.regplot(x=cdf["home_team_pregame_rating"], y=cdf["home_team_score"])

" Examime the aggregate relationship between average home_team_score by team"
cdf['HomeTeamScoreByYear'] = cdf.groupby(['season','home_team_abbr'])['home_team_score'].transform('mean')

sns.lineplot('season', 'HomeTeamScoreByYear', ci=None, 
             hue='home_team_abbr', data=cdf)

" Examine correlcation plots"
" Examime correlation between home pre game team rating and home goals"
sns.regplot(x=cdf["home_team_pregame_rating"], y=cdf["home_team_score"])

" Examime correlation between away pre game team rating and home goals"
sns.regplot(x=cdf["away_team_pregame_rating"], y=cdf["home_team_score"])

" Examime correlation between home team win pr and home goals"
sns.regplot(x=cdf["home_team_winprob"], y=cdf["home_team_score"])

" Examime correlation between home team expected points and home goals"
sns.regplot(x=cdf["home_team_expected_points"], y=cdf["home_team_score"])

" Examime correlation between away team win pr and home goals"
sns.regplot(x=cdf["away_team_winprob"], y=cdf["home_team_score"])

" Examime correlation between playoff game and home goals"
sns.regplot(x=cdf["playoff"], y=cdf["home_team_score"])

" Examime correlation between neutral location and home goals"
sns.regplot(x=cdf["neutral"], y=cdf["home_team_score"])

" Examime correlation between overtime probabilty and home goals"
sns.regplot(x=cdf["overtime_prob"], y=cdf["home_team_score"])

" Create and examine pair plots"
plot =  sns.pairplot(cdf)
plot = sns.pairplot(cdf, hue="home_team_abbr")
plot = plot.map_upper(plt.scatter)
plot = plot.map_lower(sns.kdeplot)