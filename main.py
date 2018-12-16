import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re

# import and instantiate TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=12.0)


from func import clean_text




train_df = pd.read_csv('../toxicCommentsDS/train.csv')
test_df = pd.read_csv('../toxicCommentsDS/test.csv')

cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))

# clean the comment_text in train_df [Thanks to Pulkit Jha for the useful pointer.]
train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))

# clean the comment_text in test_df [Thanks, Pulkit Jha.]
test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))


#remove char_length col
train_df = train_df.drop('char_length',axis=1)



X = train_df.comment_text
test_X = test_df.comment_text



vect = TfidfVectorizer(max_features=5000,stop_words='english')


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_dtm = vect.fit_transform(X)



# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_X_dtm = vect.transform(test_X)



# create submission file
submission_binary = pd.read_csv('../toxicCommentsDS/sample_submission.csv')


for label in cols_target:
	print('... Processing {}'.format(label))
	y = train_df[label]
	# train the model using X_dtm & y
	logreg.fit(X_dtm, y)
	# compute the training accuracy
	y_pred_X = logreg.predict(X_dtm)
	print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
	# compute the predicted probabilities for X_test_dtm
	test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
	submission_binary[label] = test_y_prob

# generate submission file
submission_binary.to_csv('submission_binary.csv',index=False)




# create submission file
submission_chains = pd.read_csv('../toxicCommentsDS/sample_submission.csv')

# create a function to add features
def add_feature(X, feature_to_add):
	'''
	Returns sparse feature matrix with added feature.
	feature_to_add can also be a list of features.
	'''
	from scipy.sparse import csr_matrix, hstack
	return hstack([X, csr_matrix(feature_to_add).T], 'csr')


for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm,y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))
    # make predictions from test_X
    test_y = logreg.predict(test_X_dtm)
    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
    submission_chains[label] = test_y_prob
    # chain current label to X_dtm
    X_dtm = add_feature(X_dtm, y)
    print('Shape of X_dtm is now {}'.format(X_dtm.shape))
    # chain current label predictions to test_X_dtm
    test_X_dtm = add_feature(test_X_dtm, test_y)
    print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))


# generate submission file
submission_chains.to_csv('submission_chains.csv', index=False)




# create submission file
submission_combined = pd.read_csv('../toxicCommentsDS/sample_submission.csv')

# corr_targets = ['obscene','insult','toxic']
for label in cols_target:
    submission_combined[label] = 0.5*(submission_chains[label]+submission_binary[label])


# generate submission file
submission_combined.to_csv('submission_combined.csv', index=False)







