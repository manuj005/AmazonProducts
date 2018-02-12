import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import cross_validation
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Initializing WordNetLemmatizer and Label Encoders
lemmatizer = WordNetLemmatizer()
le1 = preprocessing.LabelEncoder()

dataset = pd.read_excel('Training_Data_Assessment.xlsx')

# Create label for the categorical data
dataset['CategoryLabel'] = le1.fit_transform(dataset['CategoryName'])

# Function to create a lexicon of words from 'Title' column
def create_lexicon(data):
    # First create a list of all the possible words in Title using tokenize
    all_words = []
    for fields in data['Title']:
        words = word_tokenize(fields)
        for word in words:
            all_words.append(word.lower())

    # Counter will count the occurance of each word give the result in format of \
    # a dictionary with word as key and times of occurance as value
    # Will keep only words in lexicon which are not too frequent or rarely used
    title_lexicon = []
    w_counts = Counter(all_words)
    for word in w_counts:
        if 453 > w_counts[word] > 5:
            title_lexicon.append(word)
    return title_lexicon

title_lexicon = create_lexicon(dataset)

# Function to create features out of the 'Title' and 'CategoryLabel' columns
def create_features(data):
    featureset = []
    for index, row in data.iterrows():
        all_words = word_tokenize(row['Title'])
        # Initializing feature with zeros the same length as the lexicon
        # Adding 1 to the feature at same indexes where the word matched with index in lexicon
        feature = np.zeros(len(title_lexicon))
        for word in all_words:
            if word.lower() in title_lexicon:
                pos = title_lexicon.index(word.lower())
                feature[pos] += 1
            featureset.append([list(feature), row['CategoryLabel']])

    # Now shuffling the features so that the classifier does not get biased in any way
    random.shuffle(featureset)
    X = []
    Y = []
    for item in featureset:
        X.append(item[0])
        Y.append(item[1])
    return (X,Y)

features = create_features(dataset)
# Creating Training and Test datasets
X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(features[0], features[1], test_size = 0.2)

# Trying out various classifiers:
# SVM model couldn't train in 3 hrs, dropped it
# Logistic regression gave accuracy around 86%
# Decision Tree gave around 87%
# Logistic regression gave 79%


#clf = AdaBoostClassifier()
clf = DecisionTreeClassifier(min_samples_split= 40)
#clf = svm.SVC()
#clf = LogisticRegression()
#clf = GaussianNB()

# Saving the classifier as a pickle file so that time for training the model could be saved
'''
clf.fit(X_Train, Y_Train)
with open('my_pickle.pkl', 'wb') as fid:
    pickle.dump(clf, fid)
'''
clf_pkl = pickle.load(open('my_pickle.pkl', 'rb'))
pred = clf_pkl.predict(X_Test)
accu = accuracy_score(pred, Y_Test)

print(accu)

# Took first 5000 entries to classify into file 'short1'
# Creating features
# storing the predictions in a new column
data_predict = pd.read_excel('short1.xlsx')
# Retreiving the category from the label
pred_features = []
for index, row in data_predict.iterrows():
    all_words = word_tokenize(row['Title'])
    feature = np.zeros(len(title_lexicon))
    for word in all_words:
        if word.lower() in title_lexicon:
            pos = title_lexicon.index(word.lower())
            feature[pos] += 1
        pred_features.append(list(feature))


new_predictions = clf_pkl.predict(pred_features)

predictions = pd.DataFrame(new_predictions)
predictions.columns = ['pred']

data_predict['Category'] = predictions[['pred']]
data_predict['Category'] = le1.inverse_transform(data_predict['Category'])

data_predict.to_csv('classification.csv')
print('done.. ;)')
