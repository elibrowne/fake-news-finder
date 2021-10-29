# IMPORT STATEMENTS

import numpy # calculations, matrices, etc.
import csv # reading CSV files for data intake
# Bernoulli Naive-Bayes was the text classification algorithm I was introduced
# to doing work with the 20newsgroups dataset, so I'm using it here too.
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB
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

# Combine the two datasets together before vectorizing. The order doesn't matter
# because we'll randomize with train_test_split().
dataset = realNews + fakeNews
# Taking out every dictionary item of the same key in a list of dictionaries 
# can be done with iterating over it.
text = [ item['text'] for item in dataset ] # extract the text
labels = [ item['real'] for item in dataset ] # take out real/fake values

vectorizer = CountVectorizer(max_features = 100, stop_words = 'english', ngram_range = (1, 3))
x = vectorizer.fit_transform(text) # vectorize the text in each news article
y = numpy.array(labels) # just an array for 0s and 1s

# Here, we use the train_test_split method to set up the training data.
# test_size designates 25% of our dataset to test the model (75% to train).
# random_state = 10 shuffles the data. It's reproducible thanks to the 10.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = 30)

# TRAIN ALGORITHM(S)

# Training the Bernoulli Naive Bayes classification model
bernoulli = BernoulliNB() 
bernoulli.fit(xTrain, yTrain) # fit the model! train it!
# We can determine the accuracy by testing against test data.
bernoulliAcc = bernoulli.score(xTest, yTest)
print("Bernoulli accuracy: " + str(bernoulliAcc)) 

# Training the Multinomial Naive Bayes classification model
# This one seems to be a pretty standard model, so I've compared it with the
# Bernoulli model, which uses a different strategy to classify.
multinomial = MultinomialNB()
multinomial.fit(xTrain, yTrain)
multinomialAcc = multinomial.score(xTest, yTest)
print("Multinomial accuracy: " + str(multinomialAcc))
# random_state = 10: Bernoulli 98.36%; Multinomial 91.50%
# random_state = 20: Bernoulli 98.66%; Multinomial 92.34%
# random_state = 30: Bernoulli 98.39%; Multinomial 91.84%

# DONE :)