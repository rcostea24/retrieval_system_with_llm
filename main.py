import argparse
import os
import numpy as np
from retireval_system.retrieve_documents import retriev

from sklearn.neighbors import KDTree
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2').to(device)



def create_kdtree_index(embeddings):
    return KDTree(embeddings, leaf_size=40)

def load_model():
    # Set model name and device
    model_name = "OpenLLM-Ro/RoLlama2-7b-Base"
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization for better performance
        bnb_4bit_quant_type="nf4",  # Use "nf4" (normal float 4) quantization type
        llm_int8_enable_fp32_cpu_offload=True  # Offload layers to CPU if GPU memory is full
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute layers between CPU and GPU
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Model loaded successfully with 4-bit quantization!")
    return model, tokenizer

def summarize_context(relevant_docs, tokenizer, max_tokens=512):
    combined_text = " ".join(relevant_docs)
    tokenized = tokenizer(combined_text, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)

def query_system(query, documents, embedding_model, index, model, tokenizer):
    # Encode query
    query_embedding = embedding_model.encode([query])[0]
    
    # Search vector index
    distances, indices = index.query(query_embedding.reshape(1, -1), k=5)
    relevant_docs = [documents[i] for i in indices[0]]
    summarized_context = summarize_context(relevant_docs, tokenizer, max_tokens=1024)

    # Pass the most relevant documents as context to the model
    input_text = f"<context> {summarized_context}\n <intrebare> {query}\n <raspuns> "
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", default=r"C:\Users\razva\Master1\An2\IRTM\Project2\docs")
    parser.add_argument("--query", default="Cand a fost revolutia americana")
    args = parser.parse_args()

    relevant_docs, docs, doc_ids, _ = retriev(args.query, args.doc_path)

    # Generate embeddings
    embeddings = embedding_model.encode(relevant_docs)
    index = create_kdtree_index(embeddings)
    
    llm_model, tokenizer = load_model()

    answer = query_system(args.query, relevant_docs, embedding_model, index, llm_model, tokenizer)
    print(answer)


        
    
