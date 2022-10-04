# Non negative matrix factorisation topic modelling

## Overview

Demonstration for applying non-negative matrix factorisation (NMF) topic modelling to the [20 newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).

The steps in the process are:
- Get the 20 newsgroups dataset
- Remove stop words and transform the data to numerical form using term frequency - inverse document frequency (tf-idf)
- Fit the model with 100 topics
- Name the topics based on key words in each topic
- Compare it to the categories which already exist in the data.

## How do I use this project?

The following Python packages are required:
- pandas
- sklearn
- seaborn
- matplotlib
