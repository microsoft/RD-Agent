# TODO: Fix
import re

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("/kaggle/input/train.csv")
test = pd.read_csv("/kaggle/input/test.csv")
submission = pd.read_csv("/kaggle/input/sample_submission.csv")


features = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
target = train[features]


text_train = train["full_text"]
text_test = test["full_text"]

text = pd.concat([text_train, text_test], ignore_index=True)


count_words = text.str.findall(r"(\w+)").str.len()
print(count_words.sum())


""" Cleaning Text """
text = text.str.lower()

# removing special characters and numbers
text = text.apply(lambda x: re.sub("[^a-z]\s", "", x))

# remove hash tags
text = text.str.replace("#", "")

# remove words less than 3 character and greater than 7
text = text.apply(lambda x: " ".join([w for w in x.split() if len(w) > 2 and len(w) < 8]))

# removing stopwords
# text = text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))

count_words = text.str.findall(r"(\w+)").str.len()
print(count_words.sum())


most_freq_words = pd.Series(" ".join(text).lower().split()).value_counts()[:25]
text = text.apply(lambda x: " ".join(word for word in x.split() if word not in most_freq_words))

count_words = text.str.findall(r"(\w+)").str.len()

apostrophe_dict = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have",
}


def lookup_dict(txt, dictionary):
    for word in txt.split():
        if word.lower() in dictionary:
            if word.lower() in txt.split():
                txt = txt.replace(word, dictionary[word.lower()])
    return txt


text = text.apply(lambda x: lookup_dict(x, apostrophe_dict))

# Remove rare words
from collections import Counter
from itertools import chain

# split words into lists
v = text.str.split().tolist()
# compute global word frequency
c = Counter(chain.from_iterable(v))
# filter, join, and re-assign
text = [" ".join([j for j in i if c[j] > 1]) for i in v]
text = pd.Series(text)

total_word = 0
for x, word in enumerate(text):
    num_word = len(word.split())
    # print(num_word)
    total_word = total_word + num_word
print(total_word)
