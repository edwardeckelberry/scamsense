import pandas as pd
from collections import Counter
import string
import test

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
med_common = len(common_word_counts) // 2
top5_common = common_word_counts[:5]

# Top 5 spam-only words
spam_only_words = [word for word in spam_counts if word not in ham_counts]
spam_only_counts = [(word, spam_counts[word]) for word in spam_only_words]
spam_only_counts.sort(key=lambda x: x[1], reverse=True)
med_spam_only = len(spam_only_counts)//2 
top5_spam = spam_only_counts[:5]

# Find ham-only words
ham_only_words = [word for word in ham_counts if word not in spam_counts]
ham_only_counts = [(word, ham_counts[word]) for word in ham_only_words]
ham_only_counts.sort(key=lambda x: x[1], reverse=True)
med_ham = len(ham_only_counts) // 2
top5_ham = ham_only_counts[:5]

"""print("Top 5 words common to both spam and ham:")
for word, count in top5_common:
    print(f'"{word}": {count} times')

print("\nTop 5 words unique to spam messages:")
for word, count in top5_spam:
    print(f'"{word}": {count} times')"""
    
print("Central Tendency:\n")

#the mean of word frequencies
mean_spam = sum(spam_counts.values()) / len(spam_counts)
mean_ham = sum(ham_counts.values()) / len(ham_counts)
mean_common = sum([count for _, count in common_word_counts]) / len(common_word_counts)
print(f"Mean:\nspam: {mean_spam}\nham: {mean_ham}\nspam and ham: {mean_common}\n")

#the median of word frequencies
print(f"Median:\nspam: {spam_only_counts[med_spam_only][0]}, frequency: {spam_only_counts[med_spam_only][1]}")
print(f"ham: {ham_only_counts[med_ham][0]}, frequency: {ham_only_counts[med_ham][1]}")
print(f"spam and ham: {common_word_counts[med_common][0]}, frequency: {common_word_counts[med_common][1]}")

#the mode of word frequencies
print(f"\nMode:\nspam: {top5_spam[0][0]}, frequency: {top5_spam[0][1]}")
print(f"ham: {ham_words[0] }, frequency: {max(ham_counts.values())}")
print(f"spam and ham: {top5_common[0][0]}, frequency: {top5_common[0][1]}")

#the range of word frequencies
print(f"\nRange:\nspam: {spam_only_counts[0][1] - spam_only_counts[-1][1]}")
print(f"ham: {ham_only_counts[0][1] - ham_only_counts[-1][1]}")

#the midrange of word frequencies
print(f"spam: {(spam_only_counts[0][1] + spam_only_counts[-1][1]) / 2}")
print(f"ham: {(ham_only_counts[0][1] + ham_only_counts[-1][1]) / 2}")
print(f"spam and ham: {(common_word_counts[0][1] + common_word_counts[-1][1]) / 2}")

#the quantiles of word frequencies
print("\nQuantiles:")
print(f"spam: {(spam_only_counts[med_spam_only][1]+spam_only_counts[med_spam_only-1][1]) / 2}")
print(f"ham: {(ham_only_counts[med_ham][1]+ham_only_counts[med_ham-1][1]) / 2}")
print(f"spam and ham: {(common_word_counts[med_common][1]+common_word_counts[med_common-1][1]) / 2}")

#the interquartile range of word frequencies
print("\nInterquartile Range:")
print(f"spam: {(spam_only_counts[med_spam_only//2][1] + spam_only_counts[med_spam_only//2 - 1][1])/ 2 - (spam_only_counts[med_spam_only + med_spam_only//2][1] + spam_only_counts[med_spam_only + med_spam_only//2 - 1][1]) / 2}")
print(f"ham: {(ham_only_counts[med_ham//2][1] + ham_only_counts[med_ham//2 - 1][1]) / 2 - (ham_only_counts[med_ham + med_ham//2][1] + ham_only_counts[med_ham + med_ham//2 - 1][1]) / 2}")
print(f"spam and ham: {(common_word_counts[med_common//2][1] + common_word_counts[med_common//2 - 1][1]) / 2 - (common_word_counts[med_common + med_common//2][1] + common_word_counts[med_common + med_common//2 - 1][1]) / 2}") 

#this is Q1 and Q3 of both spam and hame
boxplot = (common_word_counts[med_common//2][1] + common_word_counts[med_common//2 - 1][1]) / 2 - (common_word_counts[med_common + med_common//2][1] + common_word_counts[med_common + med_common//2 - 1][1]) / 2
max_boxplot = max([spam_only_counts[0][1], ham_only_counts[0][1], common_word_counts[0][1]])
min_boxplot = min([spam_only_counts[-1][1], ham_only_counts[-1][1], common_word_counts[-1][1]])
#the box plot of word frequencies
print("\nBox Plot:")
print(f"For spam and ham: {boxplot}\nmax: {max_boxplot}\nmin: {min_boxplot}")

#the variance of word frequencies
import math
print("\nVariance:")
spam_var = math.sqrt(sum((x - mean_spam) ** 2 for x in spam_counts.values()) / len(spam_counts))
ham_var = math.sqrt(sum((x - mean_ham) ** 2 for x in ham_counts.values()) / len(ham_counts))
common_var = math.sqrt(sum((x - mean_common) ** 2 for x in [count for _, count in common_word_counts]) / len(common_word_counts))
print(f"spam: {spam_var}\nham: {ham_var}\nspam and ham: {common_var}")

#the standard deviation of word frequencies
print("\nStandard Deviation:")
spam_std = math.sqrt(spam_var)
ham_std = math.sqrt(ham_var)
common_std = math.sqrt(common_var)
print(f"spam: {spam_std}\nham: {ham_std}\nspam and ham: {common_std}")

#the skewness of word frequencies
skew_spam = (3 * (mean_spam - spam_only_counts[med_spam_only][1])) / spam_std
skew_ham = (3 * (mean_ham - ham_only_counts[med_ham][1])) / ham_std
skew_common = (3 * (mean_common - common_word_counts[med_common][1])) / common_std
print("\nSkewness:")
print(f"spam: {skew_spam}\nham: {skew_ham}\nspam and ham: {skew_common}")

#class imbalance
spam_count = len(spam_df)
ham_count = len(ham_df)
total_count = spam_count + ham_count
spam_ratio = spam_count / total_count
ham_ratio = ham_count / total_count
print("\nClass Imbalance:")
print(f"spam: 1:2\nham: 1:5\ntotal: 1:8")

correlation = common_var / (spam_std * ham_std)
print(f"\nCorrelation between spam and ham word frequencies: {correlation}")

#degree of freedom = number of rows -1 x number of columns -1
df_common = (test.df.shape[1]-1) * (len(test.df)-1)
#now we want to think of the null hypothesis:
#null hypothesis: there is no difference between the detected spam and random selection
#alternative hypothesis: there is a difference between the detected spam and random selection
#we can use a chi-squared test to test the null hypothesis
from scipy.stats import chisquare
observed = [spam_count, ham_count]
expected = [total_count / 2, total_count / 2]
chi2_stat, p_value = chisquare(observed, expected)
print(f"\nChi-squared test statistic: {chi2_stat}, p-value: {p_value}")
if p_value < 0.05:
    print("Reject the null hypothesis: there is a significant difference between the detected spam and random selection.")
else:
    print("Fail to reject the null hypothesis: there is no significant difference between the detected spam and random selection.")

