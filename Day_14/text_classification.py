! pip install -q kaggle
from google.colab import files
files.upload()

! mkdir ~/.kaggle
 ! cp kaggle.json ~/.kaggle/
 ! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
!unzip '/content/imdb-dataset-of-50k-movie-reviews.zip'

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/IMDB Dataset.csv')
df.head()

df.shape

"""Gettings only 20k records to process faster"""

df_final = pd.concat([df[df.sentiment=='positive'].head(10000), df[df.sentiment=='negative'].head(10000)])

df_final.shape

df_final.sentiment.value_counts()

stopword_list = stopwords.words('english')

def clean_function(text):
  #text  = text.lower()
  text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  text = re.sub('\[[^]]*\]', '', text)
  ps = nltk.porter.PorterStemmer()
  text = ' '.join([ps.stem(word) for word in text.split()])
  return text

df_final['review'] = df_final['review'].apply(clean_function)

def remove_stopwords(text):
  tokens = word_tokenize(text)
  tokens = [token.strip() for token in tokens]
  filtered_tokens = [token.lower() for token in tokens if token not in stopword_list]
  filtered_text = ' '.join(filtered_tokens)
  return filtered_text

df_final['review'] = df_final['review'].apply(remove_stopwords)

df_final.head()

df_final['review'][0]

label_encoder = LabelEncoder() 
df_final['sentiment']= label_encoder.fit_transform(df_final['sentiment'])

df_final.sentiment.shape

df_final.to_csv('final.csv')

X = df_final['review']
y = df_final['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))

tv_x_train = tv.fit_transform(X_train)
tv_x_test = tv.transform(X_test)

tv_x_train.shape, tv_x_test.shape

from sklearn import decomposition, preprocessing
from sklearn.decomposition import TruncatedSVD

svd = decomposition.TruncatedSVD(n_components=120)

svd.fit(tv_x_train)
xtrain_svd = svd.transform(tv_x_train)
xvalid_svd = svd.transform(tv_x_test)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

classifier = SVC(C=1.0, probability=True)
classifier.fit(xtrain_svd_scl, y_train)

classifier_predict=classifier.predict(xvalid_svd_scl)
print(classifier_predict)

print(accuracy_score(y_test, classifier_predict))

print('Classification Report')
print(classification_report(y_test, classifier_predict ,target_names=['Positive','Negative']))

# Fitting a simple Naive Bayes on TFIDF
clf = MultinomialNB()
clf.fit(tv_x_train, y_train)
predictions = clf.predict(tv_x_test)

predictions

report=classification_report(y_test,predictions,target_names=['Positive','Negative'])
print(report)

print(accuracy_score(y_test, predictions))