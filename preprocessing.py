import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopword = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()                  # Lowercase
    text = re.sub('\n', ' ', text)       # Remove newlines
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stopword])
    return text
