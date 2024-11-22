import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

# Load the dataset from a CSV file
file_path = os.path.abspath("Ethos_Dataset_Binary.csv")
data = pd.read_csv(file_path)

# Replace commas with periods in the 'comment;isHate' column
data['comment;isHate'] = data['comment;isHate'].str.replace(',', '.')

# Filter out rows that do not have exactly one ';' character, ensuring valid data format
valid_rows = data['comment;isHate'].str.count(';') == 1
data = data[valid_rows]

# Split the 'comment;isHate' column into 'comment' and 'isHate' columns
split_data = data['comment;isHate'].str.split(';', expand=True)
split_data.columns = ['comment', 'isHate']

# Recombine the cleaned data into the original DataFrame
data = pd.concat([split_data, data.drop(columns=['comment;isHate'])], axis=1)

# Convert the 'isHate' column to numeric, dropping any rows with non-numeric values
data['isHate'] = pd.to_numeric(data['isHate'], errors='coerce')
data = data.dropna(subset=['isHate'])
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Map numerical labels to human-readable labels for easy interpretation
data["labels"] = data['isHate'].map({0: "Not-Hate Speech", 1: "Hate Speech"})

# Import necessary modules for text processing
import nltk
import re
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = re.sub(r'[^\w\s]', '', text)  # Removing characters except words and spaces
    text = text.lower()  # Lowercasing for consistency
    text = re.sub('\n', '', text)
    return text

data["comment"] = data["comment"].apply(clean_data)

# Set up the feature (X) and target (y) variables
x = np.array(data["comment"])
y = np.array(data["isHate"])

# Convert text data into a matrix of token counts
cv=CountVectorizer()
x = cv.fit_transform(x)

# Split the dataset into a training set and a test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
y_train = y_train.astype(int)

# Initialize the Decision Tree classifier and train it on the training data
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = dt.predict(x_test)
y_test = y_test.astype(int)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report

#print(classification_report(y_test, y_pred, target_names=['Non-Hate Speech', 'Hate Speech']))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

sample = " Throwing stones to paraplegic kids is my favourite hobby "
sample_cleaned = clean_data(sample)

print("Cleaned Sample:", sample_cleaned)

# Transform the cleaned sample
data1 = cv.transform([sample_cleaned]).toarray()

prediction = dt.predict(data1)

# Mapping the prediction back to a label
label_map = {0: "Not-Hate Speech", 1: "Hate Speech"}
predicted_label = label_map[prediction[0]]

print("Prediction:", predicted_label)