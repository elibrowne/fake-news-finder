# IMPORT STATEMENTS

import numpy # calculations, matrices, etc.
import csv # reading csv files for data intake
# Bernoulli Naive-Bayes was the text classification algorithm I was introduced
# to doing work with the 20newsgroups dataset, so I'm using it here too.
from sklearn.naive_bayes import BernoulliNB 
# Vectorizing text is a way to use it to train an algorithm.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Classification report and confusion matrix are two ways to display data
from sklearn.metrics import classification_report, confusion_matrix

# READING IN CSV FILES, ORGANIZING DATA

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

# DIVIDING INTO TRAINING AND TEST DATA

# VECTORIZE AND TRAIN ALGORITHM

# VISUALIZE RESULTS AND SEE SCORE

# DONE :)