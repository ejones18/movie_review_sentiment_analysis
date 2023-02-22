# -*- coding: utf-8 -*-
"""
Script for assignment 2 for COM3110: Text Processing.

A Naive Bayes classifer for a set of movie reviews.

Author: Ethan Jones
"""
import warnings
warnings.filterwarnings("ignore")

import utilities as utils
import feature_selection as fs

import os
import sys
import re
import math

import argparse
import pandas as pd

USER_ID = "aca19ej"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "moviereviews")

def parse_args():
    """
    Defines the command line argument interface. Note that this method was supplied and hasn't been
    changed.
    Returns
    -------
    `args` : Dict
        A dictionary containing the arguments as per the user's command for running this script.
    """
    parser = argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-feature_selection_method', type=str, default="chi_squared", choices=["chi_squared", "manual", "tfidf"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-apply_smoothing', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    return args

class movie_review_nb:
    """"""
    def __init__(self):
        self.args = parse_args()
        self.training = pd.read_csv(os.path.join(DATA_DIR, self.args.training), sep='\t')
        self.dev = pd.read_csv(os.path.join(DATA_DIR, self.args.dev), sep='\t')
        self.test = pd.read_csv(os.path.join(DATA_DIR, self.args.test), sep='\t')
        self.features = self.args.features
        self.output_files = self.args.output_files
        self.confusion_matrix = self.args.confusion_matrix
        self.number_classes = self.args.classes
        self.smoothing = self.args.apply_smoothing
        self.feature_selection_method = self.args.feature_selection_method

    def calculate_prior(self, data):
        """
        Calculates the prior probabilities from an inputted dataframe and a given number of sentiment
        classes.
        Parameters
        ----------
        `data`: pd.DataFrame
            A Pandas dataframe containing data from the Rotten Tomatoes dataset. Must contain a column
            named 'Sentiment'.
        Returns
        -------
        `prior scores` : Tuple
            A Tuple of either length 3 or 5 containing the prior score for each sentiment class in the
            provided data.
        """
        num_reviews = len(data.index)
        neg_reviews = data.loc[data["Sentiment"] == 0]
        prob_neg = len(neg_reviews) / num_reviews
        if self.number_classes == 3:
            neu_reviews = data.loc[data["Sentiment"] == 1]
            pos_reviews = data.loc[data["Sentiment"] == 2]
            prob_pos = len(pos_reviews.index) / num_reviews
            prob_neu = len(neu_reviews.index) / num_reviews
            return (prob_pos, prob_neg, prob_neu)
        else:
            slineg_reviews = data.loc[data["Sentiment"] == 1]
            neu_reviews = data.loc[data["Sentiment"] == 2]
            slipos_reviews = data.loc[data["Sentiment"] == 3]
            pos_reviews = data.loc[data["Sentiment"] == 4]
            prob_pos = len(pos_reviews.index) / num_reviews
            prob_slipos = len(pos_reviews.index) / num_reviews
            prob_slineg = len(neg_reviews) / num_reviews
            prob_neu = len(neu_reviews.index) / num_reviews
            return (prob_pos, prob_slipos, prob_neg, prob_slineg, prob_neu)

    def calculate_likelihoods(self, data):
        """
        Calculate the feature likelihoods of each unique term within a passed in dataframe. Includes
        an option to add Laplace, +1, smoothing.
        Parameters
        ----------
        `data` : pd.DataFrame
            DataFrame whereby there is a column named 'Phrase'. This column will be the subject of the
            term likelihood calculations.
        `smoothing` : Bool (defaults to False)
            Boolean option whether to add Laplace smoothing.
        Returns
        -------
        `feature_likelihoods` : Dict
            A dictionary whose keys are the unique terms within the dataset passed in and whose keys are
            the its corresponding feature likelihood.
        """
        all_terms = [term for reviews in data['Phrase'] for term in reviews]
        total_words = len(all_terms)
        unique_words = set(all_terms)
        feature_likelihoods = {}
        laplace_smoothing_term_num = total_words + len(unique_words)
        for term in unique_words:
            term_count = all_terms.count(term)
            if self.smoothing:
                feature_likelihoods.update({term: term_count+1 / laplace_smoothing_term_num})
            else:
                feature_likelihoods.update({term: term_count / total_words})
        return feature_likelihoods

    def naive_bayes(self, train, dev, **kwargs):
        """
        A Pythonic implementation of a naive bayes classifer using 'vanilla Python'. Caters for both
        3 and 5 class sentiment datasets and uses optimisation techniques to speed up classification.
        If term appears that is not in the train dataset, then it is ignored.
        Contains calls to function that calculates feature likelihoods as well as calculating the
        posterior probabilities of each sentiment class.
        Parameters
        ----------
        `train` : pd.DataFrame
            The training dataset. Requires ["Sentiment"] column.
        `dev` : pd.DataFrame
            The dataset to predict sentiment.
        `**kwargs` : Dict
            Dictionary of keyword arguments - will contain the prior probabilities of each of the
            different classes therefore contains vary depending on the number_classes variable.
        Returns
        -------
        `dev` : pd.DataFrame
            The dataset to predict sentiment with the predictions now applied as a new
            columns ["Predicted"].
        """
        positive_prob = kwargs.get('pos_prob')
        neutral_prob = kwargs.get('neu_prob')
        negative_prob = kwargs.get('neg_prob')
        neg_reviews = train.loc[train["Sentiment"] == 0]
        if self.number_classes == 3:
            neu_reviews = train.loc[train["Sentiment"] == 1]
            pos_reviews = train.loc[train["Sentiment"] == 2]
        else:
            slightly_pos_prob = kwargs.get('slipos_prob')
            slightly_neg_prob = kwargs.get('slineg_prob')
            slightly_neg_reviews = train.loc[train["Sentiment"] == 1]
            neu_reviews = train.loc[train["Sentiment"] == 2]
            sligtly_pos_reviews = train.loc[train["Sentiment"] == 3]
            pos_reviews = train.loc[train["Sentiment"] == 4]
            slightly_neg_likelihoods = self.calculate_likelihoods(slightly_neg_reviews)
            slightly_pos_likelihoods = self.calculate_likelihoods(sligtly_pos_reviews)
        pos_likelihoods = self.calculate_likelihoods(pos_reviews)
        neu_likelihoods = self.calculate_likelihoods(neu_reviews)
        neg_likelihoods = self.calculate_likelihoods(neg_reviews)
        predicted_sentiment_arr = []
        for words in dev['Phrase']:
            positive_prob_arr = [positive_prob]
            neutral_prob_arr = [neutral_prob]
            negative_prob_arr = [negative_prob]
            if self.number_classes == 5:
                slightly_pos_prob_arr = [slightly_pos_prob]
                slightly_neg_prob_arr = [slightly_neg_prob]
            for term in words:
                term_pos_like = pos_likelihoods.get(term)
                term_neu_like = neu_likelihoods.get(term)
                term_neg_like = neg_likelihoods.get(term)
                if self.number_classes == 3:
                    if all([term_pos_like, term_neu_like, term_neg_like]) == 0:
                        continue
                else:
                    term_slightly_pos_like = slightly_pos_likelihoods.get(term)
                    term_slightly_neg_like = slightly_neg_likelihoods.get(term)
                    if all([term_pos_like, term_neu_like, term_neg_like, term_slightly_pos_like,
                            term_slightly_neg_like]) == 0:
                        continue
                    slightly_pos_prob_arr.append(term_slightly_pos_like)
                    slightly_neg_prob_arr.append(term_slightly_neg_like)
                positive_prob_arr.append(term_pos_like)
                neutral_prob_arr.append(term_neu_like)
                negative_prob_arr.append(term_neg_like)
            if self.number_classes == 3:
                posterior_probs = {math.prod(negative_prob_arr) : 0, math.prod(neutral_prob_arr) : 1,
                                   math.prod(positive_prob_arr) : 2}
            else:
                posterior_probs = {math.prod(negative_prob_arr) : 0, math.prod(slightly_neg_prob_arr) : 1,
                                   math.prod(neutral_prob_arr) : 2, math.prod(slightly_pos_prob_arr): 3,
                                   math.prod(positive_prob_arr) : 4}
            max_prob = max([*posterior_probs.keys()])
            pred_sentiment = posterior_probs.get(max_prob)
            predicted_sentiment_arr.append(pred_sentiment)
        dev["Predicted"] = predicted_sentiment_arr
        return dev

def main():
    nb = movie_review_nb()
    training_df_proc = utils.preprocess_reviews(nb.training)
    dev_df_proc = utils.preprocess_reviews(nb.dev)
    test_df_proc = utils.preprocess_reviews(nb.test)
    training_df_proc_mapped = utils.map_sentiment_vals(training_df_proc, nb.number_classes)
    dev_df_proc_mapped = utils.map_sentiment_vals(dev_df_proc, nb.number_classes)
    if nb.features == "features":
        if nb.feature_selection_method == "chi_squared":
            chosen_features = fs.get_features_chi(training_df_proc_mapped, nb.number_classes)
        elif nb.feature_selection_method == "manual":
            chosen_features = ["good", "great", "funny", "witty", "terrible", "waste", "rubbish",
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
        elif nb.feature_selection_method == "tfidf":
            chosen_features = fs.get_features_tf_idf(training_df_proc_mapped, dev_df_proc_mapped,
                                                     nb.number_classes)
        else:
            print("The feature selection method inputted is not recognised.")
            sys.exit(-1)
        training_df_proc_mapped["Phrase"] = training_df_proc_mapped["Phrase"].apply(lambda phrase: fs.feature_extraction(phrase, chosen_features))
        training_df_proc_filtered = training_df_proc_mapped.loc[training_df_proc_mapped["Phrase"] != "no-features"]
        dev_df_proc_mapped["Phrase"] = dev_df_proc_mapped["Phrase"].apply(lambda phrase: fs.feature_extraction(phrase, chosen_features))
        dev_df_proc_filtered = dev_df_proc_mapped.loc[dev_df_proc_mapped["Phrase"] != "no-features"]
        test_df_proc["Phrase"] = test_df_proc["Phrase"].apply(lambda phrase: fs.feature_extraction(phrase, chosen_features))
        test_df_proc = test_df_proc.loc[test_df_proc["Phrase"] != "no-features"]
    else:
       training_df_proc_filtered = training_df_proc_mapped
       dev_df_proc_filtered = dev_df_proc_mapped
    if nb.number_classes == 3:
        pos_prob, neg_prob, neu_prob = nb.calculate_prior(training_df_proc_filtered)
        dev_df_pred = nb.naive_bayes(training_df_proc_filtered, dev_df_proc_filtered,
                                     pos_prob=pos_prob, neg_prob=neg_prob, neu_prob=neu_prob)
        test_df_pred = nb.naive_bayes(training_df_proc, test_df_proc, pos_prob=pos_prob,
                                      neg_prob=neg_prob, neu_prob=neu_prob)
    elif nb.number_classes == 5:
        pos_prob, slipos_prob, neg_prob, slineg_prob, neu_prob = nb.calculate_prior(training_df_proc_filtered)
        dev_df_pred = nb.naive_bayes(training_df_proc_filtered, dev_df_proc_filtered,
                                     pos_prob=pos_prob, neg_prob=neg_prob, neu_prob=neu_prob,
                                     slipos_prob=slipos_prob, slineg_prob=slineg_prob)
        test_df_pred = nb.naive_bayes(training_df_proc_filtered, test_df_proc, pos_prob=pos_prob,
                                      neg_prob=neg_prob, neu_prob=neu_prob,
                                      slipos_prob=slipos_prob, slineg_prob=slineg_prob)
    else:
        print("The number of classes must be either 3 or 5.")
        sys.exit(-1)
    dev_f1_score = utils.macro_f1(dev_df_pred, nb.number_classes, nb.confusion_matrix)
    if nb.features == "all_words":
        print("%s\t%d\t%s\t%f" % (USER_ID, nb.number_classes, "False", dev_f1_score))
    else:
        print("%s\t%d\t%s\t%f" % (USER_ID, nb.number_classes, "True", dev_f1_score))
    if nb.output_files:
        dev_df_to_save = dev_df_pred.loc[:, ['SentenceId', 'Predicted']]
        test_df_to_save = test_df_pred.loc[:, ['SentenceId', 'Predicted']]
        dev_df_to_save.to_csv(f'dev_predictions_{nb.number_classes}classes_{USER_ID}.tsv', sep="\t")
        test_df_to_save.to_csv(f'test_predictions_{nb.number_classes}classes_{USER_ID}.tsv', sep="\t")

if __name__ == "__main__":
    main()
    