import PyPDF2
from spire.doc import Document
import os
from unidecode import unidecode
import nltk

from retireval_system.utils import text_preprocessing, word_preprocessing
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("romanian")

class Indexer():
    def __init__(self):
        self.inverted_dict = {}
        self.doc_id = -1
        self.text_preprocessing = text_preprocessing
        self.word_preprocessing = word_preprocessing
        self.texts = {}

    def add_doc(self, doc_path):
        self.doc_id += 1

        if doc_path.endswith(".pdf"):
            reader = PyPDF2.PdfReader(doc_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
                text += "\n"
        elif doc_path.endswith(".doc") or doc_path.endswith(".docx"):
            document = Document()
            document.LoadFromFile(doc_path)
            text = document.GetText()
            text = text.removeprefix("Evaluation Warning: The document was created with Spire.Doc for Python.\r\n")
        elif doc_path.endswith(".txt"):
            with open(doc_path) as f:
                text = f.read()

        self.texts[os.path.basename(doc_path)] = text
        words = text_preprocessing(text)

        for word in words:
            if word not in self.inverted_dict:
                self.inverted_dict[word] = [self.doc_id]
                continue
            
            if self.doc_id not in self.inverted_dict[word]:
                self.inverted_dict[word].append(self.doc_id)