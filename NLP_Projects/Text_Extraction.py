# step 1 is always data collection
import pandas as pd

df = pd.read_csv('papers.csv');
df = df.iloc[:5000, :]


# print(df)

# step 2 is text preprocessing(cleaning)[lower Case, remove HTML tags, special chars, stop words, lemmetize]

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt_tab') #downloading punkt for tokenization
# nltk.download('stopwords') //we need to download stop words from the NLTK pacjage (no big deal i guess)
stop_words = stopwords.words('english')
stop_words = set(stop_words)  # convert to set to perform unioin operation
custom_words = ['fig', 'figure', 'image', 'sample', 'using', 'show', 'result',
                'etc']  # some custom stop words that I have added

# Combining custom stop words to the orignal nltk package stopwords

stop_words = list(set(stop_words).union(set(custom_words)))
# print(stop_words)


# print(stop_words)

# Now the text cleaning
def preprocessing(txt):
    text = txt.lower()  # to lower text
    txt = re.sub(r'<.*?>', '', txt)  # removing HTML tags
    txt = re.sub(r'^[a-zA-Z]', '', txt)  # reomve any special char or any dig with space just each for alphabats
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word not in stop_words]
    txt = [word for word in txt if len(word) > 3]  # reomving slang words that of 3 words [hop,pop,mim]
    stemming = PorterStemmer()
    txt = [stemming.stem(word) for word in txt]  # what it does is that [loving --> love, moving --> move]

    return ' '.join(txt)



docs = df['paper_text'].apply(lambda x: preprocessing(x))
print(docs)

# #Countvectorizer -->
# CountVectorizer is a feature extraction method that converts text into a bag-of-words (BoW) model. It turns the text data into a matrix of token counts (frequency of words or phrases).
#1. Unigrams (1-word tokens):
# These are just individual words -> ['I', 'love', 'to', 'code']
#2. Bigrams (2-word combinations):
# These are pairs of consecutive words -> ['I love', 'love to', 'to code']
#3. Trigrams (3-word combinations):
# These are pairs of consecutive words -> ['I love to', 'love to code']


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.95, max_features=5000, ngram_range=(1,3))
word_count_vectors = cv.fit_transform(docs)


#to be continued