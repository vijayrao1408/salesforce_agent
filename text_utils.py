import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
nltk_stop_words = set(nltk.corpus.stopwords.words('english'))

def tokenize_lemma(text):
    words = text.split()
    return [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in nltk_stop_words and len(w) > 1
    ]