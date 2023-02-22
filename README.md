# Rotten Tomatoes Movie Reviews - Sentiment Analysis

A naive-bayes classifier, built from the ground up with no frameworks, for the Rotten Tomatoes movie review dataset. Includes detailed feature selection methods including TFIDF & chi-squared testing, data processing and metric evalutation.

Note: This model was developed and tested in Python v3.9+

## Instructions before running
In order to run the ```NB_sentiment_analyser.py``` script, a few libraries outside of the Python standard library[1] need installing so all the functionality can be used.

These can be installed using the following command: ```pip install -r "requirements.txt"```

It is encouraged that these packages are installed in either a conda environment[2] or a .venv environment[3] in order to stop conflicts with pre-existing libraries.

The following command will also need to be run in order to access the stopwords from the NLTK package: ```python -m nltk.downloader stopwords``` as seen from [4][5]

### Command line interface
Running the script with the `-h` flag i.e. `NB_sentiment_analyser.py -h` will yield the command line helper:

```
usage: NB_sentiment_analyser.py [-h] [-classes CLASSES] [-features {all_words,features}] [-feature_selection_method {chi_squared,manual,tfidf}] [-output_files] [-confusion_matrix] [-apply_smoothing] training dev test

A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset

positional arguments:
  training
  dev
  test

optional arguments:
  -h, --help            show this help message and exit
  -classes CLASSES
  -features {all_words,features}
  -feature_selection_method {chi_squared,manual,tfidf}
  -output_files
  -confusion_matrix
  -apply_smoothing
```

#### Example usage

Running ```python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 3 -features features -feature_selection_method manual``` will yield
a naive bayes classifcation for 3 classes and using my manually chosen features.

Running ```python NB_sentiment_analyser.py train.tsv dev.tsv test.tsv -classes 5 -features features -feature_selection_method tfidf -confusion_matrix -output_files```
will yield a naive bayes classifcation for 5 classes with chosen features using TFIDF thresholding. The results will be saved to files and the confusion matrices will be displayed.

## Solution overview

As previously alluded to, the functionality of my solution is separated out into classes and siloed modules - as is best practice for Pythonic libraries or packages. These are as follows:

* ```feature_selection``` - this module is home to all functions pertaining to the feature selection and extraction functionalities.
* ```utilities``` - this module is home to all misc functions as well as those that calculate the performance metrics. It can be argued that the latter should be packaged in with the model, but I opted for this design choice for ease.
* ```NB_sentiment_analyser``` (central script and home to the movie_review_nb class) - this module is home to the naive bayes classifier class and also handles the command line arguments.

These modules are imported into one another to ensure streamlined functionality and in a way that doesn't break the rule r.e. circular imports. Note the previous statement r.e. where these modules need to be placed in order for them to be imported correctly.

## References
* [1] https://docs.python.org/3/library/
* [2] https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
* [3] https://docs.python.org/3/library/venv.html
* [4] https://stackoverflow.com/questions/41610543/corpora-stopwords-not-found-when-import-nltk-library
* [5] https://www.nltk.org/data.html#command-line-installation
