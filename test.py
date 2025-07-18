""" import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


nltk.download('stopwords')
"""
# Create your own env before running this code:
# Step 1: open terminal and run:
#python3 -m venv env
#source env/bin/activate
#pip3 install pandas numpy
# Step 2: run this code and it should print the first 5 rows of the dataset
import string
import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv', encoding='ISO-8859-1')
print("data: ", df.head())