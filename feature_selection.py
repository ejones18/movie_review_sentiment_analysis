"""
Siloed feature_selection module containing functions pertaining to the selection and extraction of
features.

Author: Ethan Jones
"""

import utilities as utils
from NB_sentiment_analyser import movie_review_nb

import sys
import random

import numpy as np
from nltk.corpus import stopwords as stop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramAssocMeasures

def feature_extraction(phrase, features):
    """
    Feature selection function - extracts words within reviews that are apart of a pre-defined
    set of features.
    Parameters
    ----------
    `phrase` : List
        The movie review to undergo the feature extraction process.
    `features` : List
        A list of pre-defined features.
    Return
    ------
    `new_phrase` : List or str
        The phrase with only the words from the feature lexicon. If no features found, defaults to
        "no-features" which is later removed.
    """
    new_phrase_arr = []
    for word in phrase:
        if word in features:
            new_phrase_arr.append(word)
    if len(new_phrase_arr) == 0:
        new_phrase = "no-features"
    else:
        new_phrase = new_phrase_arr
    return new_phrase

def get_features_stochastic(words, training_df_, dev_df_, num_classes, use_base=False, n=2500):
    """
    NOTE: THIS FUNCTION IS DEPRECATED AND ONLY KEPT AS EVIDENCE OF SYSTEM DEVELOPMENT.
    A stochastic feature selection function. Sets a macro_f1 limit and aims to find a combination of
    n features that beats it. Iteratively calls the naive bayes classifier to get macrof1 score for
    the dev dataset based on the features stochastically chosen. This method was used as part of the
    exploratory data analysis to find the base features mentioned elsewhere.
    Parameters
    ----------
    `words` : set
        A set of all the unique terms in the training dataset.
    `training_df_` : pd.DataFrame
        The training dataset.
    `dev_df_` : pd.DataFrame
        The dev dataset.
    `num_classes` : int
        The number of sentiment classes.
    `use_base` : Bool (defaults to False)
        Choice whether to use the manually chosen base features as well as those stochastically
        chosen.
    `n` : int (defaults to 500)
        Number of stochastic features to return.
    Returns
    -------
    `features_arr` : List
        A list containing the features to use based on the above criterion.
    """
    print("This feature selection method is deprecated due to poor optimisation and will not run until line 379 is uncommented. Do so at your own risk.")
    sys.exit(-1)
    f1_limit = 0 #Set f-limit here
    predicted_score = 0
    words = list(words)
    nb = movie_review_nb()
    while predicted_score < f1_limit:
        training_df = training_df_.copy()
        dev_df = dev_df_.copy()
        features_ = []
        if use_base:
            base_features = ["good", "great", "funny", "witty", "terrible", "waste", "rubbish",
                             "horrible", "worst", "best", "boring", "entertaining", "pleasure",
                             "joy", "clever", "enjoyed", "flawed", "beautiful", "sad",
                             "strange", "puzzling", "too", "violent", "original", "touching",
                             "disappointed", "charm", "error", "hate", "dislike", "like",
                             "perfect", "low", "imaginative", "thrilling", "emotion", "well",
                             "interest", "spooky", "depressing", "rewards", "incoherent", 
                             "patience", "lack", "watchable", "amusing", "surprising", "faithful",
                             "weak", "poor", "dull", "problem", "pleasing", "disaster", "love", 
                             "shocking", "insulting", "very", "not", "mess", "dazzle", "delight",
                             "passion", "timid", "chore", "wrong", "puzzling", "bad",
                             "compelling", "negative", "positive", "superb", "super",
                             "delightful", "awful", "beautiful", "smart", "fun", "movie", "film",
                             "solid"]
            base_features.extend(features_)
            features_arr = list(set(base_features))
        else:
            features_arr = list(set(features_))
        f_n = random.randint(400, n)
        features_ = random.choices(words, k=f_n)
        training_df_filtered = training_df.loc[training_df["Phrase"] != "no-features"]
        dev_df_filtered = dev_df.loc[dev_df["Phrase"] != "no-features"]
        training_df_filtered["Phrase"] = training_df_filtered["Phrase"].apply(lambda phrase: feature_extraction(phrase, features_))
        dev_df_filtered["Phrase"] = dev_df_filtered["Phrase"].apply(lambda phrase: feature_extraction(phrase, features_))
        if num_classes == 3:
            pos_prob, neg_prob, neu_prob = nb.calculate_prior(training_df_filtered)
            dev_df_pred = nb.naive_bayes(training_df_filtered, dev_df_filtered, pos_prob=pos_prob, 
                                         neg_prob=neg_prob, neu_prob=neu_prob)
        else:
            pos_prob, slipos_prob, neg_prob, slineg_prob, neu_prob = nb.calculate_prior(training_df_proc)
            dev_df_pred = nb.naive_bayes(training_df_proc, dev_df_filtered, pos_prob=pos_prob,
                                           neg_prob=neg_prob, neu_prob=neu_prob, slipos_prob=slipos_prob,
                                           slineg_prob=slineg_prob)
        predicted_score = utils.macro_f1(dev_df_pred, 3, False)*100
    import pdb; pdb.set_trace()
    return features_arr

def get_features_tf_idf(training_df, num_classes, use_base=False, n=6000):
    """
    Function to select a set of n features from both the training and dev datasets based on the 
    tfidf scores within a passed in dataframe. Sklearn and numpy are used for this feature 
    selection method.
    References:
    - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    Parameters
    ----------
    `training_df` : pd.DataFrame
        The training dataset
    `use_base` : Bool (defaults to False)
        Choice whether to use the manually chosen base features as well as those stochastically
        chosen.
    `n` : int (defaults to 1300)
        Number of features to return.
    `chi_square_limit` : int (defaults to 5 as this seems the best value from testing)
        The chi-squared statistic threshold for features to be selected regardless of the value of n.
    Returns
    -------
    `chosen_features` : List
        A list containing the features to use based on the above criterion.
    """
    if use_base:
        base_features = ["good", "great", "funny", "witty", "terrible", "waste", "rubbish",
                         "horrible", "worst", "best", "boring", "entertaining", "pleasure",
                         "joy", "clever", "enjoyed", "flawed", "beautiful", "sad",
                         "strange", "puzzling", "too", "violent", "original", "touching",
                         "disappointed", "charm", "error", "hate", "dislike", "like",
                         "perfect", "low", "imaginative", "thrilling", "emotion", "well",
                         "interest", "spooky", "depressing", "rewards", "incoherent", 
                         "patience", "lack", "watchable", "amusing", "surprising", "faithful",
                         "weak", "poor", "dull", "problem", "pleasing", "disaster", "love", 
                         "shocking", "insulting", "very", "not", "mess", "dazzle", "delight",
                         "passion", "timid", "chore", "wrong", "puzzling", "bad",
                         "compelling", "negative", "positive", "superb", "super",
                         "delightful", "awful", "beautiful", "smart", "fun", "movie", "film",
                         "solid"]
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=n)
    vectoriser.fit(training_df["Phrase"].apply(lambda phrase: ' '.join(phrase)))
    inputs = vectoriser.transform(training_df["Phrase"].apply(lambda phrase: ' '.join(phrase)))
    feature_array = np.array(vectoriser.get_feature_names())
    features_sorted = np.argsort(inputs.toarray()).flatten()[::-1]
    chosen_features = feature_array[features_sorted][:n]
    chosen_features = chosen_features.tolist()
    chosen_features = list(set(chosen_features))
    if use_base:
        chosen_features.extend(base_features)
    return chosen_features

def get_features_chi(training_df, num_classes, use_base=False, n=6000, chi_square_limit=None):
    """
    An attempt at implementing chi-squared testing for feature selection from the training 
    and dev datasets using NLTK. Behaviour may not be perfect due to lack of advice and guidance 
    from the NLTK documentation.
    References: 
    - https://stats.stackexchange.com/questions/24179/how-exactly-does-chi-square-feature-selection-work
    - https://streamhacker.com/tag/chi-square/
    - https://stats.stackexchange.com/questions/329047/chi2-feature-selection-using-large-words-classes-in-nlp
    Parameters
    ----------
    `training_df` : pd.DataFrame
        The training dataset.
    `num_classes` : int
        The number of sentiment classes.
    `use_base` : Bool (defaults to True)
        Choice whether to use the manually chosen base features as well as those stochastically
        chosen.
    `n` : int (defaults to 3500)
        Number of features to return.
    `chi_square_limit` : int (defaults to 5 as this seems the best value from testing)
        The chi-squared statistic threshold for features to be selected regardless of the value of n.
    Returns
    -------
    `chosen_features` : List
        A list containing the features to use based on the above criterion.
    """
    if use_base:
        base_features = ["good", "great", "funny", "witty", "terrible", "waste", "rubbish",
                         "horrible", "worst", "best", "boring", "entertaining", "pleasure",
                         "joy", "clever", "enjoyed", "flawed", "beautiful", "sad",
                         "strange", "puzzling", "too", "violent", "original", "touching",
                         "disappointed", "charm", "error", "hate", "dislike", "like",
                         "perfect", "low", "imaginative", "thrilling", "emotion", "well",
                         "interest", "spooky", "depressing", "rewards", "incoherent", 
                         "patience", "lack", "watchable", "amusing", "surprising", "faithful",
                         "weak", "poor", "dull", "problem", "pleasing", "disaster", "love", 
                         "shocking", "insulting", "very", "not", "mess", "dazzle", "delight",
                         "passion", "timid", "chore", "wrong", "puzzling", "bad",
                         "compelling", "negative", "positive", "superb", "super",
                         "delightful", "awful", "beautiful", "smart", "fun", "movie", "film",
                         "solid"]
        base_features = utils.stem_terms(base_features)
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    negative_words = [terms for all_reviews in training_df.loc[training_df['Sentiment'] == 0]['Phrase'] for terms in all_reviews]
    if num_classes == 3:
        positive_words = [terms for all_reviews in training_df.loc[training_df['Sentiment'] == 2]['Phrase'] for terms in all_reviews]
        neutral_words = [terms for all_reviews in training_df.loc[training_df['Sentiment'] == 1]['Phrase'] for terms in all_reviews]
    else:
        slightly_negative_words = [terms for all_reviews in training_df.loc[training_df['Sentiment'] == 1]['Phrase'] for terms in all_reviews]
        neutral_words = [terms for all_reviews in training_df.loc[training_df['Sentiment'] == 2]['Phrase'] for terms in all_reviews]
        slightly_positive_words = [terms for all_reviews in training_df.loc[training_df['Sentiment'] == 3]['Phrase'] for terms in all_reviews]
        positive_words = [terms for all_reviews in training_df.loc[training_df['Sentiment'] == 4]['Phrase'] for terms in all_reviews]
    for word in positive_words:
        word_fd[word] = word_fd[word] + 1
        label_word_fd['pos'][word] = label_word_fd['pos'][word] + 1
    for word in neutral_words:
        word_fd[word] = word_fd[word] + 1
        label_word_fd['neu'][word] = label_word_fd['neu'][word] + 1
    for word in negative_words:
        word_fd[word] = word_fd[word] + 1
        label_word_fd['neg'][word] = label_word_fd['neg'][word] + 1
    pos_word_count = len(positive_words)
    neu_word_count = len(negative_words)
    neg_word_count = len(negative_words)
    total_word_count = pos_word_count + neu_word_count + neg_word_count
    if num_classes == 5:
        for word in slightly_negative_words:
            word_fd[word] = word_fd[word] + 1
            label_word_fd['slineg'][word] = label_word_fd['slineg'][word] + 1
        for word in slightly_positive_words:
            word_fd[word] = word_fd[word] + 1
            label_word_fd['slipos'][word] = label_word_fd['slipos'][word] + 1
        slipos_word_count = len(slightly_positive_words)
        slineg_word_count = len(slightly_negative_words)
        total_word_count = total_word_count + slineg_word_count + slipos_word_count
    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                                               (freq, pos_word_count), total_word_count)
        neu_score = BigramAssocMeasures.chi_sq(label_word_fd['neu'][word],
                                               (freq, neu_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                                               (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score + neu_score
        if num_classes == 5:
            slipos_score = BigramAssocMeasures.chi_sq(label_word_fd['slipos'][word],
                                                      (freq, pos_word_count), total_word_count)
            slineg_score = BigramAssocMeasures.chi_sq(label_word_fd['slineg'][word],
                                                      (freq, pos_word_count), total_word_count)
            word_scores[word] = word_scores[word] + slipos_score + slineg_score
    scores = {}
    best = sorted(word_scores.items(), key=lambda w: w[1], reverse=True)[:n]
    if chi_square_limit is None:
        chosen_features = [w[0] for w in best]
    else:
        chosen_features = [w[0] for w in best if w[1] > chi_square_limit]
    if use_base:
        chosen_features.extend(base_features)
    return chosen_features