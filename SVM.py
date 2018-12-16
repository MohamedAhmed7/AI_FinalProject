from preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import SGDClassifier


METHOD_NAME = "SVM"



x_train = train_df.comment_text

x_test = test_df.comment_text



y_train = train_df['toxic']
y_test = test_df['toxic']

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(x_train, y_train)


y_pred = sgd.predict(x_test)

# create submission file
submission_SVM = pd.read_csv(DATASET_SRC + '/sample_submission.csv')

accuracy = 0;
for label in cols_target:
	print('... Processing {}'.format(label))
	y_train = train_df[label]
	y_test = test_df[label]

	sgd.fit(x_train, y_train)

	y_pred = sgd.predict(x_test)

	accuracy += accuracy_score(y_pred, y_test)
	submission_SVM[label] = y_pred


# generate submission file
submission_SVM.to_csv(OUTPUT_SRC + '/' + DATASET_SRC + '/'+METHOD_NAME+'/submission.csv', index=False)







acc = round(accuracy/NUM_OF_CLASSES, 5)*100

print('SVM accuracy {0}%'.format(acc))


text_file = open(OUTPUT_SRC + '/' + DATASET_SRC + '/'+METHOD_NAME+'/accuracy.txt', "w")
text_file.write(f"accuracy =  {acc}\n")
text_file.close()

