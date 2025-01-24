import string
from unidecode import unidecode
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("romanian")

def word_preprocessing(word):
    word = word.lower()
    if word in nltk.corpus.stopwords.words("romanian"):
        return ""
    
    word_without_accents = unidecode(word)
    
    stem_word = stemmer.stem(word_without_accents)

    return stem_word

def text_preprocessing(text):
    for char in string.punctuation:
        text = text.replace(char, " ")

    words = nltk.word_tokenize(text, language="romanian", preserve_line=True)
    words_out = []
    for word in words:
        words_out.append(word_preprocessing(word))

    return words_out