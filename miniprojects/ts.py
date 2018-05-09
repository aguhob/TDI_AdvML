
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 144


# In[ ]:


import numpy as np
import pandas as pd
import datetime as dt
import gzip
import grader


# # Time Series Data: Predict Temperature
# Time series prediction presents its own challenges which are different from machine-learning problems.  As with many other classes of problems, there are a number of common features in these predictions.
# 
# ## A note on scoring
# It **is** possible to score >1 on these questions. This indicates that you've beaten our reference model - we compare our model's score on a test set to your score on a test set. See how high you can go!
# 
# ## Fetch the data:

# In[ ]:


get_ipython().system(u"aws s3 sync s3://dataincubator-course/mldata/ . --exclude '*' --include 'train.txt.gz'")


# The columns of the data correspond to the
#   - year
#   - month
#   - day
#   - hour
#   - temp
#   - dew_temp
#   - pressure
#   - wind_angle
#   - wind_speed
#   - sky_code
#   - rain_hour
#   - rain_6hour
#   - city
# 
# This function will read the data from a file handle into a Pandas DataFrame.  Feel free to use it, or to write your own version to load it in the format you desire.

# In[ ]:


def load_stream(stream):
    return pd.read_table(stream, sep=' *', engine='python',
                         names=['year', 'month', 'day', 'hour', 'temp',
                                'dew_temp', 'pressure', 'wind_angle', 
                                'wind_speed', 'sky_code', 'rain_hour',
                                'rain_6hour', 'city'])


# In[ ]:


df = load_stream(gzip.open('train.txt.gz', 'r'))


# The temperature is reported in tenths of a degree Celcius.  However, not all the values are valid.  Examine the data, and remove the invalid rows.

# In[ ]:


df = ...


# We will focus on using the temporal elements to predict the temperature.
# 
# ## Per city model
# 
# It makes sense for each city to have it's own model.  Build a "groupby" estimator that takes an estimator factory as an argument and builds the resulting "groupby" estimator on each city.  That is, `fit` should create and fit a model per city, while the `predict` method should look up the corresponding model and perform a predict on each.  An estimator factory is something that returns an estimator each time it is called.  It could be a function or a class.

# In[ ]:


from sklearn import base

class GroupbyEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, column, estimator_factory):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators
    
    def fit(self, X, y):
        # Create an estimator and fit it with the portion in each group
        return self

    def predict(self, X):
        # Call the appropriate predict method for each row of X
        return ...


# # Questions
# 
# For each question, build a model to predict the temperature in a given city at a given time.  You will be given a list of records, each a string in the same format as the lines in the training file.  Return a list of predicted temperatures, one for each incoming record.  (As you can imagine, the temperature values will be stripped out in the actual text records.)
# 
# ## month_hour_model
# Seasonal features are nice because they are relatively safe to extrapolate into the future. There are two ways to handle seasonality.  
# 
# The simplest (and perhaps most robust) is to have a set of indicator variables. That is, make the assumption that the temperature at any given time is a function of only the month of the year and the hour of the day, and use that to predict the temperature value.
# 
# **Question**: Should month be a continuous or categorical variable?  (Recall that [one-hot encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) is useful to deal with categorical variables.)

# In[ ]:


def season_factory():
    return ... # A single estimator or a pipeline

season_model = GroupbyEstimator('city', season_factory).fit(df, df['temp'])


# You will need to write a function that makes predictions from a list of strings.  You can either create a pipeline with a transformer and the `season_model`, or you can write a helper function to convert the lines to the format you expect.

# In[ ]:


grader.score('ts__month_hour_model', lambda x: [0] * len(x))


# ## fourier_model
# Since we know that temperature is roughly sinusoidal, we know that a reasonable model might be
# 
# $$ y_t = y_0 \sin\left(2\pi\frac{t - t_0}{T}\right) + \epsilon $$
# 
# where $k$ and $t_0$ are parameters to be learned and $T$ is one year for seasonal variation.  While this is linear in $y_0$, it is not linear in $t_0$. However, we know from Fourier analysis, that the above is
# equivalent to
# 
# $$ y_t = A \sin\left(2\pi\frac{t}{T}\right) + B \cos\left(2\pi\frac{t}{T}\right) + \epsilon $$
# 
# which is linear in $A$ and $B$.
# 
# Create a model containing sinusoidal terms on one or more time scales, and fit it to the data using a linear regression.

# In[ ]:


grader.score('ts__fourier_model', lambda x: [0] * len(x))


# *Copyright &copy; 2016 The Data Incubator.  All rights reserved.*
