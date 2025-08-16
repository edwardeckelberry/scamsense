# YOU HAVE TO DO THIS FIRST
# Step 1: open terminal in VSCode or editor and run:
#python3 -m venv env
#source env/bin/activate
#pip3 install pandas numpy
# Step 2: install extension "Rainbow CSV" in VSCode, or whatever editor you use
# Step 3: run this code and it should print the first 5 rows of the dataset
import string
import pandas as pd
import numpy as np
<<<<<<< Updated upstream
=======
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')


df = pd.read_csv('humor_dataset.csv', encoding='ISO-8859-1')
#spamDf = df.dropna(subset=['label', 'result'])

#this is to shorten the words for processing
stemmer = PorterStemmer()
corpus = []

stopwords_set = set(stopwords.words('english'))

for i in range(len(df)):
    #below is processing the data into just lowercase and removing punctuation
    text = df['message'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

#vectorize it
vectorizer = CountVectorizer()

#x data and y data
x = vectorizer.fit_transform(corpus).toarray()

#y is a binary classification, 0 for ham and 1 for spam
y = df['num'].values

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


clf = RandomForestClassifier(n_jobs= -1)
clf.fit(x_train, y_train)

#99.46 percent accuracy!!
clf.score(x_test, y_test)

# Example test: classify a specific message to see if it matches the label
text_to_classify = df['message'].values[10]

#this can be turned into a function
class_text = text_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
class_text = [stemmer.stem(word) for word in class_text if word not in stopwords_set]
class_text = ' '.join(class_text)

text_corpus = [class_text]

#transform the text into training data then predict the label (ham in this case)
x_text2 = vectorizer.transform(text_corpus)
pred_array = clf.predict(x_text2)
test_result = int(pred_array[0])

#example test: it's 0 == 0, meaning it's a ham message
print(test_result == df['num'].iloc[10]) 
>>>>>>> Stashed changes

df = pd.read_csv('dataset.csv', encoding='ISO-8859-1')
print("data: ", df.head())