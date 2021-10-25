# IMPORT STATEMENTS

import numpy # calculations, matrices, etc.
import csv # reading CSV files for data intake
# Bernoulli Naive-Bayes was the text classification algorithm I was introduced
# to doing work with the 20newsgroups dataset, so I'm using it here too.
from sklearn.naive_bayes import BernoulliNB 
# Vectorizing text is a way to use it to train an algorithm. Count vectorizer 
# also allows us to manipulate characters and remove stop words!
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Classification report and confusion matrix are two ways to display data
from sklearn.metrics import classification_report, confusion_matrix

""" This code reads in the CSV files containing instances of real and fake news
and does the most basic things to organize them. It  cleans the data somewhat, 
removing the date, which isn't important to figuring out if news is real or 
fake. It also adds an indication of which dataset the entry belongs to. """

# Read in real news files using Python's built-in CSV module
realNews = [] # will save the content of the real news
with open('news-sources/real.csv', mode = 'r') as file:
	# Convert to a dict-reader, which is more understandable for the data organized
	csvFile = csv.DictReader(file)
	# Go through the entries, clean them up, and make them correct
	for entry in csvFile:
		entry.pop("date") # get rid of the date: it doesn't inform real/fake
		entry["real"] = 1 # add that the entry is real to help train
		realNews.append(entry) # add entry to the list of real news

# Now, read in fake news files using the same methodology
fakeNews = []
with open('news-sources/fake.csv', mode = 'r') as file:
	csvFile = csv.DictReader(file)
	for entry in csvFile:
		entry.pop("date")
		entry["real"] = 0 # the entry is fake so the 'real' datapoint is 0
		fakeNews.append(entry)

""" This code sets up the vectorizer, which is important for parsing and 
interpreting the data. The parameters used are as follows: max_features caps
the amount of considered words (for time/memory purposes), stop_words removes
commonly-used 'stop-words' that don't contribute to the meaning of the text,
n-gram range considers how many surrounding words impact a word (not exactly
but shorthand). """

vectorizer = CountVectorizer(max_features = 100, stop_words = 'english', ngram_range = (1, 3))
x = vectorizer.fit_transform() # currently working here! I need to transform the text and set up a y axis

# DIVIDING INTO TRAINING AND TEST DATA

# This is probably out of place right now and needs to be done earlier?
dataset = realNews + fakeNews # combine the two sets: order doesn't matter.
xTrain, xTest, yTrain, yTest = train_test_split()

# VECTORIZE AND TRAIN ALGORITHM

# VISUALIZE RESULTS AND SEE SCORE

# DONE :)