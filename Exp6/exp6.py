import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

msg = pd.read_csv('6-dataset.csv', names=['message', 'label'])

print('The dimensions of the dataset:', msg.shape)

print("Unique values in 'label' column:", msg['label'].unique())

msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

if msg['labelnum'].isna().any():
    print("Warning: Unmapped labels found. Mapping them to 'neg' (0).")
    msg['labelnum'] = msg['labelnum'].fillna(0)

msg['labelnum'] = msg['labelnum'].astype('int64')

X = msg.message
y = msg.labelnum

print(X)
print(y)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)  # Adjusted test_size to get ~30% test data

print('\nThe total number of Training Data:', ytrain.shape)
print('\nThe total number of Test Data:', ytest.shape)

count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

print('\nThe words or Tokens in the text documents:\n')
print(count_vect.get_feature_names())

clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

print('\nAccuracy of the classifier is:', metrics.accuracy_score(ytest, predicted))
print('\nConfusion matrix:')
print(metrics.confusion_matrix(ytest, predicted))
print('\nThe value of Precision:', metrics.precision_score(ytest, predicted, pos_label=1))
print('\nThe value of Recall:', metrics.recall_score(ytest, predicted, pos_label=1))
