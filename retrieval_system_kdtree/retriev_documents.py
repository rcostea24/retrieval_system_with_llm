from xml.dom.minidom import Document
import PyPDF2
from sklearn.neighbors import KDTree
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_docs(doc_path):
    texts = []

    files = os.listdir(doc_path)
    for file in files:
        doc = os.path.join(doc_path, file)
        if doc.endswith(".pdf"):
            reader = PyPDF2.PdfReader(doc)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
                text += "\n"
        elif doc.endswith(".doc") or doc.endswith(".docx"):
            document = Document()
            document.LoadFromFile(doc)
            text = document.GetText()
            text = text.removeprefix("Evaluation Warning: The document was created with Spire.Doc for Python.\r\n")
        elif doc.endswith(".txt"):
            with open(doc_path) as f:
                text = f.read()
        texts.append(text)

    return texts

def retriev(query, doc_path, embedding_model):
    documents = load_docs(doc_path)

    embeddings = embedding_model.encode(documents)
    index = KDTree(embeddings, leaf_size=40)

    query_embedding = embedding_model.encode([query])[0]
    
    _, indices = index.query(query_embedding.reshape(1, -1), k=2)
    relevant_docs = [documents[i] for i in indices[0]]

    return relevant_docs
    