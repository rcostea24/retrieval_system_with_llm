import argparse
import os
import numpy as np
from retrieval_system_base.indexer import Indexer
from retrieval_system_base.searcher import Searcher

def retriev(query, doc_path):
    indexer = Indexer()

    files = os.listdir(doc_path)
    for file in files:
        indexer.add_doc(os.path.join(doc_path, file))

    searcher = Searcher(indexer.inverted_dict)
    doc_ids, num_doc = searcher.search(query)

    relevant_doc_names = np.array(files)[doc_ids]
    relevant_docs_context = []
    for doc, text in indexer.texts.items():
        if doc in relevant_doc_names:
            relevant_docs_context.append(text)

    return relevant_docs_context, relevant_doc_names, doc_ids, num_doc
        
    
