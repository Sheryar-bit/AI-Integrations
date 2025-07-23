import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import spacy

nlp_spacy = spacy.load("en_core_web_sm")


text = "Hello ! I'm learning some NLP_Pipelining, and it's really fun."
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

def spacy_lemmatize(text):
    doc = nlp_spacy(text)
    return [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]

spacy_lemmas = spacy_lemmatize(text)

print("Original Text:\n", text)
print("\nTokenized:\n", tokens)
print("\nAfter Stopword Removal:\n", filtered_tokens)
print("\nAfter Stemming:\n", stemmed_tokens)
print("\nAfter Lemmatization (WordNet):\n", lemmatized_tokens)
print("\nAfter Lemmatization (spaCy):\n", spacy_lemmas)
