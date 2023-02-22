"""
Utilities module containing preprocessing and evaluation functions.

Author: Ethan Jones
"""

import re
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as stop

def tokenise(phrase):
    """
    A custom tokeniser using stop words and stemming from the NLTK package.
    Parameters
    ----------
    `phrase` : List
        The phrase to be tokenised.
    Returns
    -------
    `tokenised_phrase` : List
        The now tokenised phrase.
    """
    stemmer  = SnowballStemmer(language='english')
    sw = [stop_word for stop_word in stop.words('english')]
    tokenised_phrase = [stemmer.stem(token) for token in word_tokenize(phrase) if token not in sw]
    return tokenised_phrase

def preprocess_reviews(data):
    """
    Central function that branches out and applies all preprocessing steps on a given input
    dataframe. Currently expands shortened terms, casts all terms to lowercase, applies
    stemming and removes stopwords.
    Note that this function is applied to all dev, test and train datasets for consistency and the
    steps are only applied to the phrase columns.
    Parameters
    ----------
    `data` : pd.DataFrame
        The input dataframe that needs its phrases preprocessing. Must have a column named 'Phrase'.
    Returns
    -------
    `data` : pd.DataFrame
        The dataframe with the preprocessing steps applied to the ["Phrase"] column.
    """
    data["Phrase"] = data["Phrase"].str.lower()
    data["Phrase"] = data["Phrase"].apply(lambda phrase: tokenise(phrase))
    data["Phrase"] = data["Phrase"].apply(lambda phrase: expand_shortened_phrase(phrase))
    data["Phrase"] = data["Phrase"].apply(lambda phrase: remove_punc(phrase))
    return data

def remove_punc(phrase):
    """
    Removes all basic punctuation, ignores '?' and '!' due to the emotion they can carry in natural
    language. Uses regex to remove.
    Paramaters
    ----------
    `phrase` : List
        The phrase with potential punctuation.
    Returns
    -------
    `phrase_arr` : List
        The phrase with punctuation removed.
    """
    phrase_str = " ".join(phrase)
    phrase_str = re.sub(r"[^\w\s!?]", "", phrase_str) 
    phrase_arr = phrase_str.split()
    return phrase_arr

def expand_shortened_phrase(phrase):
    """
    Function to expand shortend phrases into separate words i.e. don't -> do not. Uses the Python
    re library to substitute shortened words and patterns for separate terms. Also handles the
    substiution of redundant punctuation e.g. {/$%^;:.
    Parameters
    ----------
    `phrase` : list
        The phrase or sentence.
    Returns
    -------
    `phrase_arr` : list
        The phrase or sentence expanded and punctuation removed.
    """
    phrase_str = " ".join(phrase)
    phrase_str = re.sub(r"won't", "will not", phrase_str)
    phrase_str = re.sub(r"can\'t", "can not", phrase_str)
    phrase_str = re.sub(r"n\'t", " not", phrase_str)
    phrase_str = re.sub(r"\'re", " are", phrase_str)
    phrase_str = re.sub(r"[iI]t 's", "it is", phrase_str)
    phrase_str = re.sub(r"\[A-Z]'d", " would", phrase_str)
    phrase_str = re.sub(r"\'ll", " will", phrase_str)
    phrase_str = re.sub(r"\'t", " not", phrase_str)
    phrase_str = re.sub(r"\'ve", " have", phrase_str)
    phrase_str = re.sub(r"\'m", " am", phrase_str)
    phrase_arr = phrase_str.split()
    return phrase_arr

def map_sentiment_vals(data, num_cats):
    """
    A function that maps the input data to a given number of sentiment classes - either 3 or 5.
    Parameters
    ----------
    `data`: pd.DataFrame
        A dataframe containing the data whose sentiment classes need mapping. Must have a column
        named 'Sentiment'.
    `num_cats` : int
        The number of classes the data should be mapped to.
    Returns
    -------
    `data` : pd.DataFrame
        The dataframe with the appropriately mapped sentiment classes.
    """
    if num_cats == 5: return data
    elif num_cats == 3:
        data["Sentiment"] = [score-1 if score not in [0,4] else score-2 if score==4 else score for score in data["Sentiment"]]
    else:
        sys.exit(-1)
    return data

def remove_stopwords(phrase):
    """
    Removes common, stopwords from a given phrase. Stopwords are from the Python nltk library with
    some additional non-domain words added.
    Parameters
    ----------
    `phrase` : List
        The phrase, whose terms will be evaluated against the pre-defined stopwords list.
    Returns
    -------
    `phrase` : List
        The phrase with stopwords removed.
    """
    non_domain_terms = ['of','the', 'to', 'in', 'a', 'is', 'was', 'for', 'which', 'has', 'that', 'any',
                        'and', 'as', 'like', 'an', 'or', 'it', 'this', 'they', 'by', 'his', 'her', 'also',
                        'be', 'before', 'been', 'know', 'later', 'make', 'my', 'you', 'need', "why", 'an',
                        'those', 'on', 'own', '’ve', 'yourselves', 'around', 'between', 'four', 'been',
                        'alone', 'off', 'am', 'then', 'other', 'can']
    try:
        stopwords = stop.words('english')
        stopwords.extend(non_domain_terms)
    except:
        stopwords = non_domain_terms
    terms_to_remove = list(set([stopword for stopword in stopwords if stopword in phrase]))
    phrase = [term for term in phrase if term not in terms_to_remove]
    return phrase

def macro_f1(data, num_classes, heatmap=False):
    """
    Generic implementation of the Macro-F1 evaluation metric.
    References:
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2996198/
    Parameters
    ----------
    `data` : pd.DataFrame
        A dataframe with the columns 'Sentiment' and 'Predicted'.
    `num_classes` : int
        The number of sentiment classes within the dataset.
    Returns
    -------
    `macro_f1` : float
        The Macro-F1 score for the predictions.
    """
    f1_scores = []
    for sent_class in range(num_classes):
        true_pos = data.loc[(data['Sentiment'] == sent_class) & (data['Predicted'] == sent_class)]
        true_neg = data.loc[(data['Sentiment'] != sent_class) & (data['Predicted'] != sent_class)]
        false_pos = data.loc[(data['Sentiment'] == sent_class) & (data['Predicted'] != sent_class)]
        false_neg = data.loc[(data['Sentiment'] != sent_class) & (data['Predicted'] == sent_class)]
        f1 = ((2*len(true_pos.index))/((2*len(true_pos.index))+len(false_pos.index)+len(false_neg.index)))
        f1_scores.append(f1)
        if heatmap:
            draw_heatmap(len(true_pos.index), len(true_neg.index), len(false_pos.index),
                         len(false_neg.index), sent_class, num_classes)
    macro_f1 = sum(f1_scores) / num_classes
    return macro_f1

def draw_heatmap(true_positives, true_negatives, false_positives, false_negatives, sent,
                 num_classes):
    """
    Function designed to draw a confusion matrix in the form of a Seaborn heatmap.
    Parameters
    ----------
    `true_positives` : int
        The number of true positives from the predicted sentiments.
    `true_negatives` : int
        The number of true negatives from the predicted sentiments.
    `false_positives` : int
        The number of false positives from the predicted sentiments.
    `false_negatives` : int
        The number of false negatives from the predicted sentiments.
    `sent` : int
        The sentiment class.
    `num_classes` : int
        The number of sentiment classes.
    """
    confusion_matrix = np.array([[true_negatives, false_positives],
                                 [false_negatives, true_positives]
                                ])
    class_labels3 = {0: "Negative", 1: "Neutral", 2: "Positive"}
    class_labels5 = {0: "Negative", 1: "Slightly negative", 2: "Neutral", 3: "Slightly positive",
                     4: "Positive"}
    if num_classes == 3:
        sent_class = class_labels3.get(sent)
    else:
        sent_class = class_labels5.get(sent)
    plt.title(f"Confusion matrix for sentiment class {sent_class}", fontsize=20)
    sns.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.show()