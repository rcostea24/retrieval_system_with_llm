# Retrieval-Augmented Generation using RoLlama2

## 1 Introduction

In this project we had to build RAG system on top of a base retrieval one.
The scope of it is to observe what system works the best for retrieving data
from a given query. The comparison is between the basic retrieval system, the
Large Language Model alone and the RAG, which is a combination of both. The
langauge given is romanian and the language model is OpenLLM-Ro/RoLlama2-
7b-Base and sentence-transformers/paraphrase-multilingual-mpnet-base-v2 for
encoding whole documents.

## 2 Description of each system

### 2.1 Information Retrieval System

Those are basic systems that retrieve documents based on a given query and
some dictionary. In this project we have two retrieval systems, one based on in-
verted dictionaries and one based on KDTrees with sentence-transformers/paraphrase-
multilingual-mpnet-base-v2 as embedding model.

### 2.2 Large Language Model

The main Large Language Model used for both question answearing and
document retrieval is OpenLLM-Ro/RoLlama2-7b-Base. It is used alone for
both tasks.

## 3 Retrieval Augmented Generation

The model from the previous subsection is also used in pair with an IR
system for both document retrieval and question answearing.

## 4 Method

### 4.1 Data

Data used are pdf files in romanian collected from wikipedia from different
subjects. The titles are: ”Batalia din dealul Spirii”, ”Indicele calitatii vietii”,
”Oceanul Atlantic”, ”Revolutia Americana”, ”Romania”. For computational
reasons, the texts are summarized for RAG experiments.

### 4.2 Retrieval

For retrieval we use inverted dictionaries, KDTrees, and Large Language
Models.
Inverted dictionaries are used to search in what documents are the words
that are present in the query.
KDTree is a data structure used for organizing points in a k-dimensional
space. It is used Nearest Neighbor Search and Efficient Indexing.
The LLM is used as an embedding model that encode the query and the
documents. The cosine similarity is used to find the document with the highest
match with the query.

### 4.3 Question Answearing

For QA we have two methods, LLM alone and LLM with context from
retrieval systems.
For the LLM alone, we just pass the question to it and the model reponds
based on it’s knowledge. The problem with it is that the model might halluci-
nate.
In theory, a RAG system is more precise in answearing a question, because
it uses the given context as additional knowledge. A drawback for this system
is the accuracy of the retrieval system that provides context.

## 5 Comparison

### 5.1 Retrieval

In this section we compare all retrieval systems, the inverted dictionary and
KDTree.
The conclusion from the Table 1 is that KDTree retrieval is superior to
inverted dictionary, therefore we continue with it as the IR system. The table
bellow shows the retrieval tasks perfomed with the LLM alone and with LLM
combined with KDTree.
In the Table 2 it is shown the retrieval of documents using LLM model alone,
LLM paired with KDTree IR system and KDTree alone.

### 5.2 Question Answearing

In this subsection we will compare the LLM alone and the LLM with KDTree
on the questions answearing task. The comparison table is listed bellow



It can be observed that the RAG is superior. The LLM often hallucinates
and tend to generate a specific answear that has nothing to do with the query.
Using additional knowledge about the query the LLM can respond with high
accuracy, therefore the theory is proved in this experiment. Also, it is proved
that the quality of the retrieval system is very important (for example, if we
used the inverted dicitonary and we get back an empty list as in the examples
from above, the context will be empty too, so the RAG will behave like the
LLM without retrieval)

## 6 Conclusion

This project compared different approaches for information retrieval (IR)
and question answering (QA), focusing on a base IR system, a large language
model (LLM), and a Retrieval-Augmented Generation (RAG) system using Ro-
manian data. Our results show that the KDTree-based retrieval system out-
performs the inverted dictionary system, providing more accurate document
retrieval. When combined with the LLM in the RAG system, the results im-
prove further, reducing hallucinations and enhancing accuracy by leveraging the
retrieved documents. Overall, the RAG system proves to be a powerful solution
for improving both retrieval and QA tasks.

