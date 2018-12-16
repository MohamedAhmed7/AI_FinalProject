# import clean data to deal with
# train data --> train_df.comment_text
# test data --> test_df.comment_text
from preprocessing import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

train_df['tags'] = 'clean'
for index, row in train_df.iterrows():
    if(row['clean'] == False):
        for lable in cols_target:
            if(row[lable] == 1):
                train_df.iloc[index, train_df.columns.get_loc('tags')] = lable





cols_target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


X_train = train_df.comment_text
X_test  = test_df.comment_text
y_train = train_df.tags

rowsums=test_df.iloc[:,2:].sum(axis=1)
test_df['clean']=(rowsums==0)



test_df['tags'] = 'clean'
for index, row in test_df.iterrows():
    if(row['clean'] == False):
        for lable in cols_target:
            if(row[lable] == 1):
                #print(index , " changed")
                test_df.iloc[index, test_df.columns.get_loc('tags')] = lable

y_test = test_df.tags


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % np.round((accuracy_score(y_pred, y_test) * 100)))
#print(classification_report(y_test, y_pred,target_names=cols_target))







