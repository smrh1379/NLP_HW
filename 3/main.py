from sklearn.linear_model import LogisticRegression
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import numpy as np

# Function to parse a single XML file
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = root.find('TEXT').text.strip()
    tags = {tag.tag: tag.attrib['met'] for tag in root.find('TAGS')}
    return text, tags

# Directory containing the XML files
xml_dir = 'n2c2/n2c2/part1'

# List to store parsed data
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

# Display the first few rows of the dataframe
print(df.head())
# Convert text data to TF-IDF feature vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Define the target labels for classification
target_labels = ['ABDOMINAL', 'CREATININE', 'MAJOR-DIABETES']

# Encode the labels
df['ABDOMINAL_encoded'] = df['ABDOMINAL'].apply(lambda x: 1 if x == 'met' else 0)
df['CREATININE_encoded'] = df['CREATININE'].apply(lambda x: 1 if x == 'met' else 0)
df['MAJOR-DIABETES_encoded'] = df['MAJOR-DIABETES'].apply(lambda x: 1 if x == 'met' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df[['ABDOMINAL_encoded', 'CREATININE_encoded', 'MAJOR-DIABETES_encoded']], test_size=0.1, random_state=42)

# Initialize a dictionary to store the models for each label
models = {}

# Function to perform cross-validation and report mean and standard deviation
def cross_validate_and_report(model, X, y, cv=10):
    accuracy = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(accuracy_score))
    f1 = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(f1_score))
    return accuracy, f1

# Train a Linear SVM model for each label with cross-validation
for label in ['ABDOMINAL_encoded', 'CREATININE_encoded', 'MAJOR-DIABETES_encoded']:
    # model = LinearSVC()
    model = LogisticRegression()
    accuracy, f1 = cross_validate_and_report(model, X_train, y_train[label])
    models[label] = model.fit(X_train, y_train[label])  # Fit the model on the entire training set
    print(f'Cross-validation results for {label}:')
    print(f'Accuracy: Mean={np.mean(accuracy)}, Std={np.std(accuracy)}')
    print(f'F1 Score: Mean={np.mean(f1)}, Std={np.std(f1)}')
    print('\n')

# Predict new data
new_data = ["New patient condition description..."]
new_data_tfidf = vectorizer.transform(new_data)
predictions = {label: model.predict(new_data_tfidf)[0] for label, model in models.items()}
print(predictions)

