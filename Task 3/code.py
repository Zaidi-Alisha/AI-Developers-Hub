import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#df = pd.read_csv(r'C:\Users\Yousuf Traders\Desktop\task 3\news.csv')
df = pd.read_csv('news.csv')

#remove unnecessary column
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

#checking and filling missing values
df.fillna("", inplace=True)

#to preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text) 
    words = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words] 
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words] 
    return ' '.join(words)

df['cleaned'] = df['text'].apply(clean_text)

#representing text in numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'FAKE': 0, 'REAL': 1})

#split dataset so that training and testing can be done
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#applying naive bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

#applying random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#save model and vectorizer
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
