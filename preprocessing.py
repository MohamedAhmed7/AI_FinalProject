import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

DATASET_SRC = "smallDataset"
OUTPUT_SRC = "output"

NUM_OF_CLASSES = 6


train_df = pd.read_csv(DATASET_SRC + '/train.csv')
test_df  = pd.read_csv(DATASET_SRC + '/testMod.csv')

# add clean comments cell
rowsums=train_df.iloc[:,2:].sum(axis=1)
train_df['clean']=(rowsums==0)
train_df['clean'].sum()


colors_list = ["brownish green", "pine green", "ugly purple",
               "blood", "deep blue", "brown", "azure"]

# visualizing classes (training stats)
palette= sns.xkcd_palette(colors_list)

x=train_df.iloc[:,2:].sum()

plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values,palette=palette)
plt.title("Class")
plt.ylabel('Occurrences', fontsize=12)
plt.xlabel('Type ')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, label,
            ha='center', va='bottom')

plt.show()



cols_target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))

# clean the comment_text in train_df [Thanks to Pulkit Jha for the useful pointer.]
train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))

# clean the comment_text in test_df [Thanks, Pulkit Jha.]
test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))


# here data are ready for any model
#remove char_length col
train_df = train_df.drop('char_length',axis=1)
