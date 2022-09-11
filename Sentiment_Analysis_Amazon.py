#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import nltk
import re
import contractions
import warnings

nltk.download('wordnet', quiet=True)

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# In[198]:


# ! pip install bs4 # in case it is not installed

# Dataset: 
#   https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz
#   with columns: 'marketplace','customer_id','review_id', 'product_id', 'product_parent', 'product_title','product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date'


# ## Read Data

# In[199]:


#NOTE: You will potentially have to download the data locally and unzip.
df = pd.read_csv('https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz', compression='gzip', sep='\t', usecols=['star_rating', 'review_body'])


# ## Keep Reviews and Ratings

# In[200]:


# The Reviews and Ratings columns was the only thing read when importing the data.
#  Only read in the required columns.


# In[201]:


# First  check if any of our rows have NAN entries.
df.isna().sum()


# In[202]:


# Drop the NAN rows as they will not help in the training, and there is plenty of other rows with complete entries to work with.
df.dropna(axis=0, inplace=True)
df.isna().sum()


#  ## We select 20000 reviews randomly from each rating class.

# In[203]:


# Ensure that the ratings column has a uniform datatype to remove any possibility of error
df = df.astype({'star_rating':'int64'})


# In[204]:


# Could use: df['star_rating'].value_counts() to see the amount of reviews per rating


# In[205]:


# Randomly sample/keep 20000 entries of each 'group' in the star-ratings column and update the dataframe
df = df.groupby('star_rating').apply(lambda x: x.sample(20000)).reset_index(drop=True)


# # Data Cleaning

# ## PRE-data cleaning avg char length of reviews 

# In[206]:


# Calculate the average character length of the 'review_body' column of the dataframe before it has undergone any data cleaning
# Store it in variable to print at the end of data cleaning
len_before_clean = df['review_body'].apply(len).mean()


# ## Reviews to lower-case

# In[207]:


# Turns all of the reviews in the 'review_body' column to lower-case and assigns result in a new column in data frame called 'clean_data'
df['clean_data'] = df['review_body'].str.lower()


# ## Remove HTML and URLs 

# In[208]:


# Find it better to first get rid of the <br /> comments in data before removing html because traces of <br /> remain when the order is reversed.
df['clean_data']=df['clean_data'].apply(lambda x: re.sub(r'<br />', '', x))


# In[209]:

# Ignore a MarkupResemblesLocatorWarning from bs4. This is fine because we are only using bs4 to to clean messages of potential 
#   html tags from very noisy data sources.
# Neet to do this: import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# bs4 grabs all the text inside tags which is a great primer for removing html
df['clean_data']=df['clean_data'].apply(lambda x: BeautifulSoup(x, features='lxml').get_text())

# This regular expression ensures to replace any text starting with http and any non-whitespace character connected to it with nothing ''.
df['clean_data']=df['clean_data'].apply(lambda x: re.sub(r'http\S+', '', x))
# Warning may pop up saying it is not going to grab content of the html


# ## Perform contractions (done before removing non-alphabetical char)

# In[210]:

# After reviewing 'tfidf.get_feature_names()' from past attempts, it is clear that a large portion of the reviews have a 
#   problem with words being accidentally merged together (human error). Focusing on some words that could be 
#   important influencers of the prediction star_rating, functions have been created to take care of some possible 
#   words merging. When separating these words, we can have a larger pool of influencing words rather than treating 
#   them as individual cases.

def fix_contraction_greats(word):
    
    word = re.sub(r"greata", "great a", word)
    word = re.sub(r"greatalthough", "great although", word)
    word = re.sub(r"greatyou", "great you", word)
    word = re.sub(r"greatwith", 'great with', word)
    word = re.sub(r"greatwhen", "great when", word)
    word = re.sub(r"greatwhat", "great what", word)
    word = re.sub(r"greatwell", "great well", word)
    word = re.sub(r"greatwas", "great was", word)
    word = re.sub(r"greatvery", "great very", word)
    word = re.sub(r"greatupdate", "great update", word)
    word = re.sub(r"greatthey", "great they", word)
    word = re.sub(r"greatthen", "great then", word)
    word = re.sub(r"greatthen", "greata", word)
    word = re.sub(r"greatthe", "great the", word)
    word = re.sub(r"greatthanks", "great thanks", word)
    word = re.sub(r"greatthank", "great thank", word)
    word = re.sub(r"greatso", "great so", word)
    word = re.sub(r"greatshe", "great she", word)
    word = re.sub(r"greatsample", "great sample", word)
    word = re.sub(r"greatproblem", "great problem", word)
    word = re.sub(r"greatnow", "great now", word)
    word = re.sub(r"greats", "great", word) 

    return word

def fix_contraction_good(word):
    
    word = re.sub(r"goodall", "good all", word)
    word = re.sub(r"goodalso", "good also", word)
    word = re.sub(r"goodbad", "good bad", word)
    word = re.sub(r"goodbut", "good but", word)
    word = re.sub(r"goodcolor", "good color", word)
    word = re.sub(r"goodquality", "good quality", word)
    word = re.sub(r"goodhowever", "good however", word)
    word = re.sub(r"goodlooking", "good looking", word)
    word = re.sub(r"goodrecommended", "good recommended", word)
    word = re.sub(r"goodthey", "good they", word)
    word = re.sub(r"goodupdate", "good update", word)
    word = re.sub(r"goodplease", "good please", word)
    word = re.sub(r"goodit", "good it", word)
    word = re.sub(r"goodif", "good if", word)

    return word

def fix_contraction_bad(word):
    
    word = re.sub(r"badbecause", "bad because", word)
    word = re.sub(r"badhowever", "bad however", word)
    word = re.sub(r"badif", "bad if", word)
    word = re.sub(r"badit", "bad it", word)
    word = re.sub(r"badjust", "bad just", word)
    word = re.sub(r"badlike", "bad like", word)
    word = re.sub(r"badlooking", "bad looking", word)
    word = re.sub(r"badlyi", "badly i", word)
    word = re.sub(r"badlyinserted", "badly inserted", word)
    word = re.sub(r"badoverall", "bad overall", word)
    word = re.sub(r"badquality", "bad quality", word)
    word = re.sub(r"badthe", "bad the", word)
    word = re.sub(r"badthis", "bad this", word)
    word = re.sub(r"badvery", "bad very", word)
    word = re.sub(r"badwife", "bad wife", word)
    word = re.sub(r"badyou", "bad you", word)

    return word

def fix_contraction_terrible(word):
    
    word = re.sub(r"terribleafter", "terrible after", word)
    word = re.sub(r"terriblei", "terrible i", word)
    word = re.sub(r"terriblelooks", "terrible looks", word)
    word = re.sub(r"terribleonly", "terrible only", word)
    word = re.sub(r"terribleplease", "terrible please", word)
    word = re.sub(r"terriblethe", "terrible the", word)

    return word


df['clean_data'] = df['clean_data'].apply(lambda x: fix_contraction_greats(x))
df['clean_data'] = df['clean_data'].apply(lambda x: fix_contraction_good(x))
df['clean_data'] = df['clean_data'].apply(lambda x: fix_contraction_bad(x))
df['clean_data'] = df['clean_data'].apply(lambda x: fix_contraction_terrible(x))

# Decided to use this library rather than create my own generalized regular expression function because it can identify words that do not have apostrophes such as dont rather than just words like don't.
# I chose to do this before removing non-alphabetical characters because I did not want to leave contractions without an apostrophe, but turns out it will not matter with this contraction library.
# Need to import contractions.
df['clean_data'] = df['clean_data'].apply(lambda x: contractions.fix(x))


# ## Remove non-alphabetical char

# In[211]:


# Originally thought this should be done after performing contractions so words like "n't" that are found from tokenizing are not changed to "nt".
#   performing contractions could first change "n't" to "not". 
#NOTE: Ideally I wanted to KEEP characters like: ":)", ":(", ":/" (otherwise known as emoticons) to help us with sentiment

df['clean_data'] = df['clean_data'].apply(lambda x: ' '.join([re.sub('[^A-Za-z]+','', x) for x in nltk.word_tokenize(x)]))


# ## Remove extra spaces

# In[212]:


# Replaces instances of multiple whitespace characters in a row with just one space.
df['clean_data'] = df['clean_data'].apply(lambda x: re.sub('\s+', ' ', x))


# ## POST-Data cleaning avg char length of reviews 

# In[213]:


# Calculate and print the average character length of the 'review_body' column of the dataframe after it has undergone data cleaning

len_after_clean = df['clean_data'].apply(len).mean()

print("{:>45} {:>25}".format('(BEFORE data cleaning)','(AFTER data cleaning)'))
print("{:>1} {:>17} {:>26}".format('Avg character length:', len_before_clean, len_after_clean))


# # Pre-processing

# In[214]:


# The average character length of the 'clean_data' column of the dataframe before it has undergone any pre-processing is equivalent to len_after_clean
# Store it in variable to print at the end of pre-processing.
len_before_preproc = len_after_clean


# ## remove the stop words 

# In[215]:


# Assign to a variable a list of english stopwords.
# Checks the reviews word by word to see if they match any words from the list of stop-words and removes them if they do.
# Need to do this: from nltk.corpus import stopwords

stop_list = stopwords.words('english')
df['clean_data'] = df['clean_data'].apply(lambda x: ' '.join([x for x in x.split() if x not in stop_list]))


# ## perform lemmatization  

# In[216]:


# Reduces similar words in meaning and spelling to a similar root word.
# NOTE: changing the lemmatization pos to 'v' or verb is a better focus.
# Need to do this: from nltk.stem import WordNetLemmatizer

lemmatization = WordNetLemmatizer()
df['clean_data'] = df['clean_data'].apply(lambda x: ' '.join([lemmatization.lemmatize(word, pos='v') for word in nltk.word_tokenize(x)]))


# ## POST-Preprocessing avg char length of reviews 

# In[217]:


# Calculate and print the average character length of the 'clean_data' column of the dataframe after it has undergone pre-processing

len_after_preproc = df['clean_data'].apply(len).mean()
      
print("\n{:>45} {:>25}".format('(BEFORE pre-processing)','(AFTER pre-processing)'))
print("{:>1} {:>17} {:>26}".format('Avg character length:', len_before_preproc, len_after_preproc))


# # TF-IDF Feature Extraction

# In[218]:


# This will randomly rearrange the rows so they are not in order of star_val
df = df.sample(frac=1) 

#from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

# (document_id, token_id) and tf-idf score
tfidf_dataset = tfidf.fit_transform(df['clean_data'])


# In[219]:


# Converts our spicy sparse matrix into a panda sparse matrix so we can easily use train_test_split()

tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_dataset)


# In[220]:


# Need 80% training data and 20% testing data, so we set test_size = 0.2.
# Need to do this: from sklearn.model_selection import train_test_split
#X - tfidf
#Y - star ratings

(X_train, X_test, Y_train, Y_test) = train_test_split(tfidf_df, df['star_rating'], test_size=0.2)


# In[221]:


# A function that will be used to print our results according to the assignment
def print_results(precision, recall, fscore):
    precision_l = precision.tolist()
    recall_l = recall.tolist()
    fscore_l = fscore.tolist()

    print ("{:>10} {:>20} {:>20} {:>20}".format('Rating','Precision','Recall','f1-score'))
    for x in precision_l:
          print("{:>10} {:>20} {:>20} {:>20}".format(precision_l.index(x)+1, x, recall_l[precision_l.index(x)], fscore_l[precision_l.index(x)]))

    print ("{:>10} {:>20} {:>20} {:>20}".format('Macro Avg', macro_avg[0], macro_avg[1], macro_avg[2]))


# # Perceptron

# In[223]:


# Use the instance of the Perceptron class to fit the tfidf training data with the star_rating training data.
# Use the Perceptron predict method predict the star_ratings of the tfidf testing data.
# Output the results (precision, recall, fscore, and avg) using the tailored print function.
# Need to do this: from sklearn.linear_model import Perceptron
# Need to do this: from sklearn.metrics import precision_recall_fscore_support

robot = Perceptron()
robot.fit(X_train, Y_train)
Y_predicted = robot.predict(X_test)

(precision,recall,fscore,support) = precision_recall_fscore_support(Y_test, Y_predicted)
macro_avg = precision_recall_fscore_support(Y_test, Y_predicted, average='macro')

print('\nPerceptron Results:\n')
print_results(precision, recall, fscore)


# # SVM

# In[224]:


# Use the instance of the LinearSVC class to fit the tfidf training data with the star_rating training data.
# Use the LinearSVC predict method predict the star_ratings of the tfidf testing data.
# Output the results (precision, recall, fscore, and avg) using the tailored print function.
# Need to do this: from sklearn.svm import LinearSVC
robot = LinearSVC()
robot.fit(X_train, Y_train)
Y_predicted = robot.predict(X_test)

(precision,recall,fscore,support) = precision_recall_fscore_support(Y_test, Y_predicted)
macro_avg = precision_recall_fscore_support(Y_test, Y_predicted, average='macro')

print('\nSVM Results:\n')
print_results(precision, recall, fscore)


# # Logistic Regression

# In[225]:


# Use the instance of the LogisticRegression class to fit the tfidf training data with the star_rating training data.
# Use the LogisticRegression predict method predict the star_ratings of the tfidf testing data.
# Output the results (precision, recall, fscore, and avg) using the tailored print function.
# Need to do this: from sklearn.linear_model import LogisticRegression
robot = LogisticRegression(max_iter=10000)
robot.fit(X_train, Y_train)
Y_predicted = robot.predict(X_test)

(precision,recall,fscore,support) = precision_recall_fscore_support(Y_test, Y_predicted)
macro_avg = precision_recall_fscore_support(Y_test, Y_predicted, average='macro')

print('\nLogistic Regression Results:\n')
print_results(precision, recall, fscore)


# # Multinomial Naive Bayes

# In[226]:


# Use the instance of the MultinomialNB class to fit the tfidf training data with the star_rating training data.
# Use the MultinomialNB predict method predict the star_ratings of the tfidf testing data.
# Output the results (precision, recall, fscore, and avg) using the tailored print function.
# Need to do this: from sklearn.naive_bayes import MultinomialNB
robot = MultinomialNB()
robot.fit(X_train, Y_train)
Y_predicted = robot.predict(X_test)

(precision,recall,fscore,support) = precision_recall_fscore_support(Y_test, Y_predicted)
macro_avg = precision_recall_fscore_support(Y_test, Y_predicted, average='macro')

print('\nNaive Bayes Results:\n')
print_results(precision, recall, fscore)


# In[ ]:




