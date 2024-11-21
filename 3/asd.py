import os
import xml.etree.ElementTree as ET
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# import gensim
# from gensim.summarization import summarize

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = root.find('TEXT').text.strip()
    tags = {tag.tag: tag.attrib['met'] for tag in root.find('TAGS')}
    return text, tags
xml_dir = 'n2c2/n2c2/part1'
data = []

# Parse all XML files
for file_name in os.listdir(xml_dir):
    if file_name.endswith('.xml'):
        file_path = os.path.join(xml_dir, file_name)
        text, tags = parse_xml(file_path)
        filtered_tags = {key: tags[key] for key in ['ABDOMINAL', 'CREATININE', 'MAJOR-DIABETES']}
        filtered_tags['text'] = text
        data.append(filtered_tags)

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to the text column
df['clean_text'] = df['text'].apply(preprocess_text)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
# Encode the labels
df['ABDOMINAL_encoded'] = df['ABDOMINAL'].apply(lambda x: 1 if x == 'met' else 0)
df['CREATININE_encoded'] = df['CREATININE'].apply(lambda x: 1 if x == 'met' else 0)
df['MAJOR-DIABETES_encoded'] = df['MAJOR-DIABETES'].apply(lambda x: 1 if x == 'met' else 0)

# Combine labels into a single DataFrame
y = df[['ABDOMINAL_encoded', 'CREATININE_encoded', 'MAJOR-DIABETES_encoded']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train a OneVsRestClassifier with Linear SVM
model = OneVsRestClassifier(LinearSVC())
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# new_data = ["New patient condition description..."]
# new_data_clean = preprocess_text(new_data[0])
# new_data_tfidf = vectorizer.transform([new_data_clean])
# predictions = model.predict(new_data_tfidf)
# print(predictions)