
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import matplotlib
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 144


# In[2]:


from grader import score


# # NLP: Analyzing Review Text
# 
# Unstructured data makes up the vast majority of data.  This is a basic intro to handling unstructured data.  Our objective is to be able to extract the sentiment (positive or negative) from review text.  We will do this from Yelp review data.
# 
# The first three questions task you to build models, of increasing complexity, to predict the rating of a review from its text.  These models will be assessed based on the root mean squared error of the number of stars predicted.  There is a reference solution (which should not be too hard to beat) that defines the score of 1.
# 
# The final question asks only for the result of a calculation, and your results will be compared directly to those of a reference solution.
# 
# ## A note on scoring
# It **is** possible to score >1 on these questions. This indicates that you've beaten our reference model - we compare our model's score on a test set to your score on a test set. See how high you can go!
# 
# ## Download and parse the data
# 
# To start, let's download the dataset from Amazon S3:

# In[3]:


get_ipython().system(u"aws s3 sync s3://dataincubator-course/mldata/ . --exclude '*' --include 'yelp_train_academic_dataset_review.json.gz'")


# The training data are a series of JSON objects, in a gzipped file. Python supports gzipped files natively: [gzip.open](https://docs.python.org/2/library/gzip.html) has the same interface as `open`, but handles `.gz` files automatically.
# 
# The built-in json package has a `loads()` function that converts a JSON string into a Python dictionary.  We could call that once for each row of the file. [ujson](http://docs.micropython.org/en/latest/library/ujson.html) has the same interface as the built-in `json` library, but is *substantially* faster (at the cost of non-robust handling of malformed json).  We will use that inside a list comprehension to get a list of dictionaries:

# In[4]:


import gzip
import ujson as json

with gzip.open('yelp_train_academic_dataset_review.json.gz') as f:
    data = [json.loads(line) for line in f]


# If you're having trouble with running out of memory, you can do quite well with a 5% or 10% subsample of the data.  A quick way to take a random subsample is to run the following load instead:

# In[5]:


import gzip
import ujson as json
import random

#Fraction to subsample, needs to be between 0 and 1
subsample = 0.1

with gzip.open('yelp_train_academic_dataset_review.json.gz') as f:
    data = [json.loads(line) for line in f if random.random() < subsample]


# Scikit Learn will want the labels in a separate data structure, so let's pull those out now.

# In[6]:


stars = [row['stars'] for row in data]


# ### Notes:
# 1. [Pandas](http://pandas.pydata.org/) is able to read JSON text directly.  Use the `read_json()` function with the `lines=True` keyword argument.  While the rest of this notebook will assume you are using a list of dictionaries, you can complete it with dataframes, if you so desire. Some of the example code will need to be modified in this case.
# 
# 2. There are obvious miscodings in the data.  There is no need to try to correct them.
# 
# ## Building models
# 
# For the first three questions, you will need to build and train an estimator to predict the star rating from the text of a review.  We recommend building a pipeline out of transformers and estimators provided by Scikit Learn.  You can decide whether these pipelines should take full review objects or just their text as input to the `fit()` and `predict()` methods, but it does pay to be consistent.
# 
# You may find it useful to serialize the trained models to disk.  This will allow you to reload the models after restarting the notebook, without needing to retrain them.  We recommend using the [`dill` library](https://pypi.python.org/pypi/dill) for this (although the [`joblib` library](http://scikit-learn.org/stable/modules/model_persistence.html) also works).  Use
# ```python
# dill.dump(estimator, open('estimator.dill', 'w'))
# ```
# to serialize the object `estimator` to the file `estimator.dill`.  If you have trouble with this, try setting the `recurse=True` keyword args in the call of `dill.dump()`.  The estimator can be deserialized with
# ```python
# estimator = dill.load(open('estimator.dill', 'r'))
# ```
# 
# You may run into trouble with the size of your models and Digital Ocean's memory limit. This is a major concern in real-world applications. Your production environment will likely not be that different from Digital Ocean and being able to deploy there is important. Think about what information the different stages of
# your pipeline need and how you can reduce the memory footprint.
# 
# Additionally, you may notice that your serialized models are very large and take a long time to load.  Some hints to reduce their size:
# 
# - If you are using `GridSearchCV` to find the optimal values of hyperparameters (and you should be), the resultant object will contain many copies of the estimator that aren't needed any more.  Instead of serializing the whole `GridSearchCV`, serialize just the estimator with the correct hyperparameters.  This can be accessed through the `.best_estimator_` attribute of the `GridSearchCV` object.  Alternatively, the `.best_params_` attribute gives the best values of the hyperparameters.
# 
# - The `CountVectorizer` keeps track of all words that were excluded from vectorization in its `.stop_words_` attribute.  This can be interesting to examine, but isn't needed for predictions.  Set this attribute to the empty list before serializing it to save disk space.

# # Questions
# 
# Each of the "model" questions asks you to create a function that models the number of stars given in a review from the review text.  It will be passed a list of dictionaries.  Each of these will have the same format as the JSON objects you've just read in.  This function should return a list of numbers of the same length, giving the predicted star ratings.
# 
# This function is passed to the `score()` function, which will receive input from the grader, run your function with that input, report the results back to the grader, and print out the score the grader returned.  Depending on how you constructed your estimator, you may be able to pass the predict method directly to the `score()` function.  If not, you will need to write a small wrapper function to mediate the data types.
# 
# ## bag_of_words_model
# Build a linear model predicting the star rating based on the count of the words in each document (bag-of-words model).  Use a [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) or [HashingVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer) to produce a feature matrix giving the counts of each word in each review.  Feed this in to linear model, such as `Ridge` or `SGDRegressor`, to predict the number of stars from each review.
# 
# **Hints**:
# 1. Don't forget to use tokenization!  This is important for good performance but it is also the most expensive step.  Try vectorizing as a first initial step and then running grid-serach and cross-validation only on of this pre-processed data.  `CountVectorizer` has to memorize the mapping between words and the index to which it is assigned.  This is linear in the size of the vocabulary.  The `HashingVectorizer` does not have to remember this mapping and will lead to much smaller models.
# 
#  ```python
#  from sklearn.feature_extraction.text import CountVectorizer
# 
#  text = [row['text'] for row in data]
#  X = CountVectorizer().fit_transform(text)
# 
#  # Now, this can be run with many different parameters
#  # without needing to retrain the vectorizer:
#  model.fit(X, stars, hyperparameter=something)
#  ```
# 
# 2. Try choosing different values for `min_df` (minimum document frequency cutoff) and `max_df` in `CountVectorizer`.  Setting `min_df` to zero admits rare words which might only appear once in the entire corpus.  This is both prone to overfitting and makes your data unmanageably large.  Don't forget to use cross-validation or to select the right value.  Notice that `HashingVectorizer` doesn't support `min_df`  and `max_df`.  However, it's not hard to roll your own transformer that solves for these.
# 
# 3. Try using [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) or [RidgeCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV).  If the memory footprint is too big, try switching to [Stochastic Gradient Descent](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) You might find that even ordinary linear regression fails due to the data size.  Don't forget to use [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) to determine the regularization parameter!  How do the regularization parameter `alpha` and the values of `min_df` and `max_df` from `CountVectorizer` change the answer?
# 
# 4. You will likely pick up several hyperparameters between the tokenization step and the regularization of the estimator.  While is is more strictly correct to do a grid search over all of them at once, this can take a long time. Quite often, doing a grid search over a single hyperparameter at a time can produce similar results.  Alternatively, the grid search may be done over a smaller subset of the data, as long as it is representative of the whole.
# 
# 5. Finally, assemble a pipeline that will transform the data from records all the way to predictions.  This will allow you to submit its predict method to the grader for scoring.

# In[ ]:


from sklearn.pipeline import Pipeline

bag_of_words_est = Pipeline([
    # Column selector (remember the ML project?)
    # Vectorizer
    # Frequency filter (if necessary)
    # Regressor
])
bag_of_words_est.fit(data, stars)


# In[ ]:


score('nlp__bag_of_words_model', bag_of_words_est.predict)


# ## normalized_model
# Normalization is key for good linear regression. Previously, we used the count as the normalization scheme.  Add in a normalization transformer to your pipeline to improve the score.  Try some of these:
# 
# 1. You can use the "does this word present in this document" as a normalization scheme, which means the values are always 1 or 0.  So we give no additional weight to the presence of the word multiple times.
# 
# 2. Try using the log of the number of counts (or more precisely, $log(x+1)$). This is often used because we want the repeated presence of a word to count for more but not have that effect tapper off.
# 
# 3. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a common normalization scheme used in text processing.  Use the [TFIDFTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer). There are options for using `idf` and taking the logarithm of `tf`.  Do these significantly affect the result?
# 
# Finally, if you can't decide which one is better, don't forget that you can combine models with a linear regression.

# In[ ]:


score('nlp__normalized_model', normalized_est.predict)


# ## bigram_model
# In a bigram model, we'll consider both single words and pairs of consecutive words that appear.  This is going to be a much higher dimensional problem (large $p$) so you should be careful about overfitting.
# 
# Sometimes, reducing the dimension can be useful.  Because we are dealing with a sparse matrix, we have to use [TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD).  If we reduce the dimensions, we can use a more sophisticated models than linear ones.
# 
# As before, memory problems can crop up due to the engineering constraints. Playing with the number of features, using the `HashingVectorizer`, incorporating `min_df` and `max_df` limits, and handling stop-words in some way are all methods of addressing this issue. If you are using `CountVectorizer`, it is possible to run it with a fixed vocabulary (based on a training run, for instance). Check the documentation.
# 
# **A side note on multi-stage model evaluation:** When your model consists of a pipeline with several stages, it can be worthwhile to evaluate which parts of the pipeline have the greatest impact on the overall accuracy (or other metric) of the model. This allows you to focus your efforts on improving the important algorithms, and leaving the rest "good enough".
# 
# One way to accomplish this is through ceiling analysis, which can be useful when you have a training set with ground truth values at each stage. Let's say you're training a model to extract image captions from websites and return a list of names that were in the caption. Your overall accuracy at some point reaches 70%. You can try manually giving the model what you know are the correct image captions from the training set, and see how the accuracy improves (maybe up to 75%). Alternatively, giving the model the perfect name parsing for each caption increases accuracy to 90%. This indicates that the name parsing is a much more promising target for further work, and the caption extraction is a relatively smaller factor in the overall performance.
# 
# If you don't know the right answers at different stages of the pipeline, you can still evaluate how important different parts of the model are to its performance by changing or removing certain steps while keeping everything else constant. You might try this kind of analysis to determine how important adding stopwords and stemming to your NLP model actually is, and how that importance changes with parameters like the number of features.

# In[ ]:


score('nlp__bigram_model', bigram_est.predict)


# ## food_bigrams
# Look over all reviews of restaurants.  You can determine which businesses are restaurants by looking in the `yelp_train_academic_dataset_business.json.gz` file from the ml project or downloaded below.

# In[ ]:


get_ipython().system(u"aws s3 sync s3://dataincubator-course/mldata/ . --exclude '*' --include 'yelp_train_academic_dataset_business.json.gz'")


# In[ ]:


with gzip.open('yelp_train_academic_dataset_business.json.gz') as f:
    business_data = [json.loads(line) for line in f]


# Each row of this file corresponds to a single business.  The category key gives a list of categories for each; take all where "Restaurants" appears.

# In[ ]:


restaurant_ids = ...


# In[ ]:


assert len(restaurant_ids) == 12876


# The "business_id" here is the same as in the review data.  Use this to extract the review text for all reviews of restaurants.

# In[ ]:


restaurant_reviews = ...


# In[ ]:


assert len(restaurant_reviews) == 574278


# We want to find collocations --- that is, bigrams that are "special" and appear more often than you'd expect from chance. We can think of the corpus as defining an empirical distribution over all ngrams.  We can find word pairs that are unlikely to occur consecutively based on the underlying probability of their words. Mathematically, if $p(w)$ be the probability of a word $w$ and $p(w_1 w_2)$ is the probability of the bigram $w_1 w_2$, then we want to look at word pairs $w_1 w_2$ where the statistic
# 
#   $$ \frac{p(w_1 w_2)}{p(w_1) p(w_2)} $$
# 
# is high.  Return the top 100 (mostly food) bigrams with this statistic with the 'right' prior factor (see below).
# 
# Estimating the probabilities is simply a matter of counting, and there are number of approaches that will work.  One is to use one of the tokenizers to count up how many times each word and each bigram appears in each review, and then sum those up over all reviews.  You might want to know that te `CountVectorizer` has a `.get_feature_names()` method which gives the string associated with each column.  (Question for thought: Why doesn't the `HashingVectorizer` have a similar method?)
# 
# *Questions:* This statistic is a ratio and problematic when the denominator is small.  We can fix this by applying Bayesian smoothing to $p(w)$ (i.e. mixing the empirical distribution with the uniform distribution over the vocabulary).
# 
# 1. How does changing this smoothing parameter affect the word pairs you get qualitatively?
# 
# 2. We can interpret the smoothing parameter as adding a constant number of occurrences of each word to our distribution.  Does this help you determine set a reasonable value for this 'prior factor'?
# 
# 3. For fun: also check out [Amazon's Statistically Improbable Phrases](http://en.wikipedia.org/wiki/Statistically_Improbable_Phrases).
# 
# *Implementation note:*
# As you adjust the size of the Bayesian smoothing parameter, you will notice first nonsense phrases being removed and then legitimate bigrams being removed, leaving you with only generic bigrams.  The goal is to find a value of the smoothing parameter between these two transitions.
# 
# The reference solution is not an aggressive filterer: it errors in favor of leaving apparently nonsensical words. On further consideration, many of these are actually somewhat meaningful. The smoothing parameter chosen in the reference solution is equivalent to giving each word 90 previous appearances prior to considering this data.  This was chosen by generating a list of bigrams for a range of smoothing parameters and seeing how many of the bigrams were shared between neighboring values.  When the shared fraction reached 95%, we judged the solution to have converged.  Note that `min_df` should not be set too high, where it could exclude these borderline words.

# In[ ]:


score('nlp__food_bigrams', lambda: ["kare kare"] * 100)


# *Copyright &copy; 2016 The Data Incubator.  All rights reserved.*
