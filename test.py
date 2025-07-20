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

df = pd.read_csv('humor_dataset.csv', encoding='ISO-8859-1')
print("data: ", df.head())