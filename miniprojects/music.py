
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 144


# In[ ]:


import grader


# # Classifing Music by Genre
# 
# Music offers an extremely rich and interesting playing field. The objective of this miniproject is to develop models that are able to recognize the genre of a musical piece, first from pre-computed features and then working from the raw waveform. This is a typical example of a classification problem on time series data.
# 
# Each piece has been classified to belong to one of the following genres:
# - electronic
# - folkcountry
# - jazz
# - raphiphop
# - rock
# 
# The model will be assessed based on the accuracy score of your classifier.  There is a reference solution.  The reference solution has a score of 1. *(Note that this doesn't mean that the accuracy of the reference solution is 1)*. Keeping this in mind...
# 
# ## A note on scoring
# It **is** possible to score >1 on these questions. This indicates that you've beaten our reference model - we compare our model's score on a test set to your score on a test set. See how high you can go!
# 

# # Questions
# 
# 
# ## Question 1: All Features Model
# Download a set of pre-computed features from Amazon S3:

# In[ ]:


get_ipython().system(u"aws s3 sync s3://dataincubator-course/mldata/ . --exclude '*' --include 'df_train_anon.csv'")


# This file contains 549 pre-computed features for the training set. The last column contains the genre.
# 
# Build a model to generate predictions from this feature set. Steps in the pipeline could include:
# 
# - a normalization step (not all features have the same size or distribution)
# - a dimensionality reduction or feature selection step
# - ... any other transformer you may find relevant ...
# - an estimator
# - a label encoder inverse transform to return the genre as a string
# 
# Use GridSearchCV to find the scikit learn estimator with the best cross-validated performance.
# 
# *Hints:*
# - Scikit Learn's [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) can center the data and/or scale by the standard deviation.
# - Use a dimensionality reduction technique (e.g. [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)) or a feature selection criteria when possible.
# - Use [GridSearchCV](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV) to improve score.
# - Use a [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) to generate an encoding for the labels.
# - The model needs to return the genre as a string. You may need to create a wrapper class around scikit-learn estimators in order to do that.
# 
# Submit a function that takes a list of records, each a list of the 549 features, and returns a list of genre predictions, one for each record.

# In[ ]:


def all_features_est(records):
    return ['blues' for r in records]

grader.score('music__all_features_model', all_features_est)


# ## Question 2: Raw Features Predictions
# 
# For questions 2 and 3, you will need to extract features from raw audio.  Because this extraction can be rather time-consuming, you will not conduct the feature extraction of the test set in real time during the grading.
# 
# Instead, you will download a set of test files.  After you have trained your model, you will run it on the test files, to make a prediction for each.  Then submit to the grader a dictionary of the form
# 
# ```python
# {
#   "fe_test_0001.mp3": "electronic",
#   "fe_test_0002.mp3": "rock",
#   ...
# }
# ```
# 
# A sets of files for training and testing are available on Amazon S3:

# In[ ]:


# Training files
get_ipython().system(u"aws s3 sync s3://dataincubator-course/mldata/ . --exclude '*'     --include 'music_train.tar.gz'     --include 'music_train_labels.csv'     --include 'music_feature_extraction_test.tar.gz'")


# All songs are sampled at 44100 Hz.
# 
# The simplest features that can be extracted from a music time series are the [zero crossing rate](https://en.wikipedia.org/wiki/Zero-crossing_rate) and the [root mean square energy](https://en.wikipedia.org/wiki/Root_mean_square).
# 
# 1. Build a function or a transformer that calculates these two features starting from a raw file input.  In order to go from a music file of arbitrary length to a fixed set of features you will need to use a sliding window approach, which implies making the following choices:
# 
#  1. what window size are you going to use?
#  2. what's the overlap between windows?
# 
#  Besides that, you will need to decide how you are going to summarize the values of such features for the whole song. Several strategies are possible:
#  -  you could decide to describe their statistics over the whole song by using descriptors like mean, std and higher order moments
#  -  you could decide to split the song in sections, calculate statistical descriptors for each section and then average them
#  -  you could decide to look at the rate of change of features from one window to the next (deltas).
#  -  you could use any combination of the above.
# 
#  Your goal is to build a transformer that will output a "song fingerprint" feature vector that is based on the 2 raw features mentioned above. This vector has to have the same size, regardless of the duration of the song clip it receives.
# 
# 2. Train an estimator that receives the features extracted by the transformer and predicts the genre of a song.  Your solution to Question 1 should be a good starting point.
# 
# Use this pipeline to predict the genres for the 145 files in the `music_feature_extraction_test.tar.gz` set and submit your predictions as a dictionary.
# 
# *Hints*
# - Extracting features from time series can be computationally intensive. Make sure you choose wisely which features to calculate.
# - You can use MRJob or PySpark to distribute the feature extraction part of your model and then train an estimator on the extracted features.

# In[ ]:


def raw_features_predictions():
    return {("fe_test_%04d.mp3" % i): 'blues' for i in xrange(1, 146)}

grader.score('music__raw_features_predictions', raw_features_predictions)


# ## Question 3: All Features Predictions
# The approach of Question 2 can be generalized to any number and kind of features extracted from a sliding window. Use the [librosa library](https://github.com/librosa/librosa) to extract features that could better represent the genre content of a musical piece.
# You could use:
# - spectral features to capture the kind of instruments contained in the piece
# - MFCCs to capture the variations in frequencies along the piece
# - Temporal features like tempo and autocorrelation to capture the rhythmic information of the piece
# - features based on psychoacoustic scales that emphasize certain frequency bands.
# - any combination of the above
# 
# As for question 1, you'll need to summarize the time series containing the features using some sort of aggregation. This could be as simple as statistical descriptors or more involved, your choice.
# 
# As a general rule, build your model gradually. Choose few features that seem interesting, calculate the descriptors and generate predictions.
# 
# Make sure you `GridSearchCV` the estimators to find the best combination of parameters.
# 
# Use this pipeline to predict the genres for the 145 files in the `music_feature_extraction_test.tar.gz` set and submit your predictions as a dictionary.
# 
# **Questions for Consideration:**
# 1. Does your transformer make any assumption on the time duration of the music piece? If so how could that affect your predictions if you receive longer/shorter pieces?
# 
# 2. This model works very well on one of the classes. Which one? Why do you think that is?

# In[ ]:


def all_features_predictions():
    return {("fe_test_%04d.mp3" % i): 'blues' for i in xrange(1, 146)}

grader.score('music__all_features_predictions', all_features_predictions)


# *Copyright &copy; 2016 The Data Incubator.  All rights reserved.*
