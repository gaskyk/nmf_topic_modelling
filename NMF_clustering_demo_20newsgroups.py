"""
Topic modelling of 20 newsgroups data using NMF
===============================================
The code demonstrates topic modelling on the 20 newsgroups
dataset using NMF (non-negative matrix factorisation)

Requirements
:requires: pandas
:requires: sklearn
:requires: seaborn
:requires: matplotlib
"""


# Import libraries
# Standard libraries
import pandas as pd

# Dataset
from sklearn.datasets import fetch_20newsgroups

# Libraries for clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Libraries for visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Import data
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
docs = data['data']
targets = data['target']
target_names = data['target_names']
classes = [data['target_names'][i] for i in data['target']]

# Use tf-idf features for NMF. First remove stop words, words occurring in less than
# 3 documents or in at least 95% of documents
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=3, stop_words="english")
tfidf = tfidf_vectorizer.fit_transform(docs)

# Set NMF model parameters. n_components is the number of topics required
model = NMF(
    n_components=100,
    random_state=1,
    init="random",
    beta_loss="kullback-leibler",
    solver="mu",
    max_iter=1000,
    alpha=0.00005,
    l1_ratio=0.5
)

# Fit the NMF model
W = model.fit_transform(tfidf)
H = model.components_

# Convert the results of the model to a pandas data frame
nmf_features = pd.DataFrame(W)
components_df = pd.DataFrame(H, columns=tfidf_vectorizer.get_feature_names())

# Get the topic to which the text is most strongly associated
nmf_features['Topic_number'] = nmf_features.idxmax(axis=1)

# Get a name for the topic based on the 3 most relevant words for that topic
topic_names_list = []
for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    temp = tmp.nlargest(3)
    topic_name = " ".join(temp.index.tolist())
    topic_names_list.append(topic_name)

# Add topic name and topic number to the components data frame
components_df['Topic_name'] = topic_names_list
components_df['Topic_number'] = components_df.index.tolist()

# Create a dataset with the topic number and name attached to the documents and classes
joined_df = pd.concat([pd.Series(docs), pd.Series(classes), nmf_features['Topic_number']], axis=1)
joined_df.columns = ['Docs', 'Classes', 'Topic_number']
joined_df = pd.merge(joined_df, components_df[['Topic_number', 'Topic_name']], on='Topic_number', how='inner')

# Create a pivot table from the data for visual comparison between topics and news
# group categories
pivot_df = pd.crosstab(index=joined_df['Topic_name'], columns=joined_df['Classes'])

# Create a heat map using the seaborn library
ax = sns.heatmap(pivot_df, annot=True, cmap="YlGnBu")\
    .set(title='Heat map comparing NMF topics and categories in the 20 newsgroups dataset',
         xlabel='Categories', ylabel='NMF topic name')
plt.show()

# Create a bar plot for the rec.sport.hockey category
hockey = pivot_df['rec.sport.hockey'].sort_values(ascending=False).head(10)
ax2 = sns.barplot(x=hockey, y=hockey.index)\
    .set(title='NMF topics for the rec.sport.hockey category in 20 newsgroups', xlabel='Count', ylabel='NMF topic name')
plt.show()
