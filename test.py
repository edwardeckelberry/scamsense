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
#y should be a number, 0 or 1: if ham or spam + dumor = 0, else 1
#df['result'] = df['label'].apply(lambda x: 0 if x == 'ham' else 1)
#x is the vectorized corpus, y is the result column
y = df['num'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = RandomForestClassifier(n_jobs= -1)
clf.fit(x_train, y_train)

#99.46 percent accuracy!!
clf.score(x_test, y_test)

text_to_classify = df['message'].values[10]

class_text = text_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
class_text = [stemmer.stem(word) for word in class_text if word not in stopwords_set]
class_text = ' '.join(class_text)

text_corpus = [class_text]

x_text2 = vectorizer.transform(text_corpus)
test_result = int(clf.predict(x_text2))
print(test_result == df['num'].iloc[10])  # This should print True if the prediction matches the actual label

