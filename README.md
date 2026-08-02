# Sentiment140

Binary sentiment classification on tweets, using
[Sentiment140](http://help.sentiment140.com/for-students/) (1.6 million
tweets, auto-labeled by each tweet's emoticon, then stripped before
release).

## Data

This works from a 20,000-tweet stratified sample (10,000 per
class), pulled from a parquet mirror of the full dataset rather than
downloading all 1.6M rows. `00_data_setup_eda.ipynb` fetches this
automatically if it doesn't already exist locally. The official
`testdata.manual.2009.06.14.csv` stays bundled as-is.

## Notebooks

1. `00_data_setup_eda.ipynb`: builds the stratified sample, checks
   whether emoticons leaked into the tweet text (they didn't, Stanford
   already stripped them before release), and flags that the official
   test set has 3 classes (negative/neutral/positive) while training
   only has 2.
2. `01_statistical_testing.ipynb`: real hypothesis tests, mentioning
   someone and using exclamation marks both correlate significantly
   with positive sentiment, tweet length doesn't.
3. `02_feature_engineering_selection.ipynb`: a capped 5,000-feature
   TF-IDF matrix, keeping the feature space compact for classical ML
   models.
4. `03_model_training_evaluation.ipynb`: logistic regression and random
   forest, evaluated on a real internal held-out split and on
   Stanford's own hand-labeled test tweets, a genuinely external check.
5. `04_clustering.ipynb`: KMeans and Gaussian mixture clustering by
   TF-IDF profile, which turns out to find topic, not sentiment.

## Results

Logistic regression reaches 78.8% accuracy on Stanford's official
hand-labeled test set (slightly better than its own 76.6% internal
validation, a good sign it isn't overfitting), against a 50.7% baseline.
Unsupervised clustering can't recover sentiment at all (adjusted Rand
index around 0.01), TF-IDF captures topic, not polarity, which is why
the supervised models, with labels to learn from, do so much better.

Full write-up with charts: `docs/index.html` (also published via GitHub
Pages).

## Future work

- Scale up training beyond 20,000 tweets, the full 1.6M-tweet corpus
  almost certainly has more signal than a TF-IDF logistic regression
  can currently extract.
- Try pretrained word embeddings or a transformer, now that a correct
  baseline and evaluation exist to compare against.
- Build a genuine 3-class model to make full use of the official test
  set instead of excluding its 139 neutral tweets.
