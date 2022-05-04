# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 08:46:15 2022

@author: Michael
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn import preprocessing
import random
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

from sklearn.tree import DecisionTreeClassifier
import matplotlib.image as pltimg

from sklearn.model_selection import train_test_split
from sklearn import tree
from dtreeviz.trees import dtreeviz

from yellowbrick.regressor import ResidualsPlot
from sklearn.pipeline import Pipeline

"Import csv file"
df = pd.read_csv('C:/Users/Michael/Downloads/nhl_elo.csv')

" Make a copy of the data set"
cdf = df.copy()

" Filter out seasons prior to 2016 season to ensure all active teams are included" 
cdf = cdf[(cdf['season'] >= 2016) & (cdf['season'] <= 2021)]

" Part 1 Data Preparation"
" Since we are looking at regression, here is a plot of the target values"
" 1. Examime the distribution of home_team_score "
cdf.hist(column='home_team_postgame_rating')

" 1a. Describe the target variable"
cdf['home_team_score'].describe()
" 7,515 rows, Average 3.02, Min=0, Max=10"

" 2. Check for NA values"
cdf.isnull().any()
cdf.info()

" Drop game_importance_rating and game_overall_rating since NaN for all obs"
cdf=cdf.drop(['game_importance_rating','game_overall_rating'],axis=1)
cdf.isnull().any()

" Confirm target variable has no null values"
cdf['home_team_score'].isnull().values.any()

cdf.isnull().any()
cdf.info()
cdf.describe()


" 3. Decide which non-numeric columns should be kept"

" Keep ot (overtime indicator) but fill null values"
cdf['ot'] = cdf['ot'].fillna('noOT')
cdf.isnull().any()
cdf.info()
cdf.describe()

" Explore keep home and away abbr"
home_team_tab = pd.crosstab(index=cdf["home_team_abbr"],  columns="count")  
away_team_tab = pd.crosstab(index=cdf["away_team_abbr"],  columns="count")        

" Home and Away abbreviations are NOT hierarchtical i.e. STL > CHI > PIT"
" So we should avoid label encoding if we want to use them in the model"
" Lets apply get_dummies"
cdf = pd.get_dummies(cdf, columns=['home_team_abbr'])
cdf = pd.get_dummies(cdf, columns=['away_team_abbr'])
cdf = pd.get_dummies(cdf, columns=['ot'])

cdf.info()
cdf.describe()

" Check for multicolinearity using variance of inflation (VIF)"
def calculate_vif(cdf):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = cdf.columns
    for i in range(0, x_var_names.shape[0]):
        y = cdf[x_var_names[i]]
        x = cdf[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)

X=cdf.drop(['home_team_score','date','status','home_team','away_team'],axis=1)
X.info()
X.describe()
X.isnull().any()
my_VIF= calculate_vif(X)

" Evidence of multi-colinearity between home/away ratings and home/away abbreviations since VIF >> 5"
" Exclude columns causing multicolinearity"
cdf = cdf[['season', 'home_team','date','playoff','neutral','home_team_pregame_rating',
           'away_team_pregame_rating','home_team_winprob','overtime_prob','home_team_score',
           'away_team_score','ot_2OT','ot_3OT','ot_5OT','ot_OT','ot_SO',
           'home_team_postgame_rating','away_team_postgame_rating']]

" Check for multicolinearity again"
X=cdf.drop(['home_team_score','season','home_team','date','home_team_winprob','away_team_pregame_rating','overtime_prob'],axis=1)
calculate_vif(X)

" All Vif < 5. OK to proceed"

"5 Create atleast 3 aggregation columns"
" Rolling average home_team_score, Rolling sd home_team_score, Rolling home_team_rating"
" Sort the dataframe"
cdf = cdf.sort_values(by = ['home_team', 'date'], ascending = [True, True], na_position = 'first')

cdf['home_score_ma'] = cdf.groupby('home_team')['home_team_score'].shift(1).rolling(3).mean().transform(lambda x: x.fillna(x.mean()))
cdf['home_score_sd'] = cdf.groupby('home_team')['home_team_score'].shift(1).rolling(3).std().transform(lambda x: x.fillna(x.mean()))
cdf['home_rating_ma'] = cdf.groupby('home_team')['home_team_pregame_rating'].shift(1).rolling(3).mean().transform(lambda x: x.fillna(x.mean()))
cdf['ot_ma'] = cdf.groupby('home_team')['ot_OT'].shift(1).rolling(3).mean().transform(lambda x: x.fillna(x.mean()))

" Attempted to keep home and away abbreviation since the target variable varied "
" by home and away team "
" Ultimately dropped dropped due to multicolinearity home and away pregame rating"
" Kept ot, dummy coded, and cacluated lag count of ot games in last 3 games "
" Because consecutive over time games take up a lot of energy and can reduce effort "

" 7 Split data into features (x_data) and labels (y_data)"

x_data = cdf[['playoff','neutral',
              'away_team_pregame_rating','home_team_pregame_rating',
              'away_team_postgame_rating','home_team_postgame_rating',             
              'home_team_winprob','overtime_prob','away_team_score',
              'ot_2OT','ot_3OT','ot_5OT','ot_OT','ot_SO',
              'home_score_ma','home_score_sd','ot_ma']]

y_data= cdf[['home_team_score']]

" Create a function thats take 2 paramters"
" 1 for test_size to split between test and train "
" 1 for regression model algorithm"
" Use a Standard Scaler"
" Return the model accuracy for the given split and model"

def buildModel(split, mod):
    " Split Test and Train"
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=split, random_state=1)
    model_scaled = Pipeline([
        ('scale',StandardScaler()),
        ('model',mod)
        ])
    model_scaled.fit(X_train, y_train)
    return model_scaled.score(X_test, y_test)

buildModel(.1, linear_model.LinearRegression())
"0.8467031360948989"
buildModel(.2, linear_model.LinearRegression())
"0.8607773730860806"
buildModel(.3, linear_model.LinearRegression())
"0.8598425726838617"
buildModel(.4, linear_model.LinearRegression())
"-295079972272.5723"
buildModel(.5, linear_model.LinearRegression())
"-689029288942.7584"
buildModel(.6, linear_model.LinearRegression())
"-26418758.13262252"
buildModel(.7, linear_model.LinearRegression())
"-2196783943.6683693"
buildModel(.8, linear_model.LinearRegression())
"-4146116952642.7515"
buildModel(.9, linear_model.LinearRegression())
"-16286578993160.787"

buildModel(.1, linear_model.SGDRegressor())
"-12.3841771768543"
buildModel(.2, linear_model.SGDRegressor())
"-58.14727752252292"
buildModel(.3, linear_model.SGDRegressor())
"-29.01869981947882"
buildModel(.4, linear_model.SGDRegressor())
"-92869.82887498417"
buildModel(.5, linear_model.SGDRegressor())
"-10289.538423098253"
buildModel(.6, linear_model.SGDRegressor())
"-47.94843885111174"
buildModel(.7, linear_model.SGDRegressor())
"-0.14641284316459813"
buildModel(.8, linear_model.SGDRegressor())
"0.7321046975604248"
buildModel(.9, linear_model.SGDRegressor())
" 0.66729536"

" 8 using a scaling/normalization library to normalize data"
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.3, random_state=1)

model_scaled = Pipeline([
       ('scale',StandardScaler()),
       ('model',linear_model.LinearRegression())
       ])
model_scaled.fit(X_train, y_train)
model_scaled.score(X_test, y_test)
model_scaled.get_params()

" Print a comparison of the two models’ accuracy "
print("Accuracy of linear regression")
buildModel(.3, linear_model.LinearRegression())   
print("Accuracy of SGDRegressor")
buildModel(.8, linear_model.SGDRegressor())

" Store this data and plot the relationship between accuracy and percent of training data. "

percent_test = [.1, .2, .3]
accuracy = [0.847, 0.861, 0.860]
plt.plot(percent_test, accuracy )
plt.ylabel("Linear Regression Accuracy")
plt.xlabel("% Test Sample")
plt.title("Model accuracy vs. %Test")
plt

" Using a residuals library of your choice, plot the residuals from the linear regression "
" model. If you chose a classification problem, this plot will look slightly strange. "
" a. Write a few sentences analyzing this plot."

" Visualize Residuals"
visual = ResidualsPlot(model_scaled)
visual.fit(X_train,y_train)
visual.score(X_test,y_test)
visual.poof

"  Make an argument for which of the two models for the problem type you chose works "
" better in this case."
" The linear regression with 30% Test provided the highest accuracy"
" The SGG regressor achieved an accuracy of 73%"

" In this case using linear regression outperformed SGD regressor in the test data "
" Based on accuracy in the test, we would recommend linear regression to predict home team score "





