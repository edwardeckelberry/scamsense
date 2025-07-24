import string
import pandas as pd
import numpy as np
# Load the dataset
df = pd.read_csv('dataset.csv', encoding='latin1')

# Calculate message lengths
df['msg_length'] = df['v2'].astype(str).apply(len)

print("Message Length Statistics:")
print("Mean:", df['msg_length'].mean())
print("Median:", df['msg_length'].median())
print("Mode:", df['msg_length'].mode()[0])

# Most common label
print("\nMost common label (mode):", df['v1'].mode()[0])