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
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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

#y is a binary classification, 0 for (ham and dumor) or (spam and dumor) and 1 for spam and humor
y = df['num'].values

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Random Forest Classifier is the model used for classification
clf = RandomForestClassifier(n_jobs= -1)
clf.fit(x_train, y_train)

#this scores the accuracy of the model
print("Accuracy: ", clf.score(x_test, y_test))

#f1 score
y_pred = clf.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("F1 Score: ", f1)

#TNR calculation
"""tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tnr = tn / (tn + fp)
print(f"TNR (Specificity): {tnr}") """


# Example test: classify a specific message to see if it matches the label
num_val = 8
text_to_classify = df['message'].values[num_val]

print("Text to classify:", text_to_classify)

#make the text lowercase and remove punctuation
class_text = text_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()

#stem words and remove non-stopwords
class_text = [stemmer.stem(word) for word in class_text if word not in stopwords_set]
#concatenate the words back into a single string, separated by spaces
class_text = ' '.join(class_text)

#put the processed text into a list to match the input format of the vectorizer
text_corpus = [class_text]

#transform the text into training data then predict the label
x_text2 = vectorizer.transform(text_corpus)
test_result = int(clf.predict(x_text2))

print("Predicted label:", test_result)
print("Actual label:", df['num'].iloc[num_val])

#example test: if true, then the prediction matches the actual label
print(test_result == df['num'].iloc[num_val]) 