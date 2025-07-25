import pandas as pd
from collections import Counter
import string

# Load the dataset
df = pd.read_csv('dataset.csv', encoding='latin1')

# Filter for spam and ham messages
spam_df = df[df['v1'] == 'spam'].dropna(subset=['v2'])
ham_df = df[df['v1'] == 'ham'].dropna(subset=['v2'])

# Combine all messages into one string for each label
all_spam = ' '.join(spam_df['v2'].astype(str)).lower()
all_ham = ' '.join(ham_df['v2'].astype(str)).lower()

# Remove punctuation
translator = str.maketrans('', '', string.punctuation)
all_spam = all_spam.translate(translator)
all_ham = all_ham.translate(translator)

# Split into words
spam_words = all_spam.split()
ham_words = all_ham.split()

# Define words to exclude
exclude_words = set(
    ['i', 'the', 'and', 'to', 'is', 'in', 'it', 'of', 'a', 
     'that', 'this', 'for', 'with', 'as', 'on', 'was', 
     'at', 'by', 'an', 'be', 'are', 'from', 'or', 'not', 
     'but', 'if', 'all'])
spam_words = [word for word in spam_words if word not in exclude_words]
ham_words = [word for word in ham_words if word not in exclude_words]

# Count word frequencies
spam_counts = Counter(spam_words)
ham_counts = Counter(ham_words)

# Find common words
common_words = set(spam_counts) & set(ham_counts)
common_word_counts = [(word, spam_counts[word] + ham_counts[word]) for word in common_words]
common_word_counts.sort(key=lambda x: x[1], reverse=True)
top5_common = common_word_counts[:5]

# Top 5 spam-only words
spam_only_words = [word for word in spam_counts if word not in ham_counts]
spam_only_counts = [(word, spam_counts[word]) for word in spam_only_words]
spam_only_counts.sort(key=lambda x: x[1], reverse=True)
top5_spam = spam_only_counts[:5]

print("Top 5 words common to both spam and ham:")
for word, count in top5_common:
    print(f'"{word}": {count} times')

print("\nTop 5 words unique to spam messages:")
for word, count in top5_spam:
    print(f'"{word}": {count} times')
    
    