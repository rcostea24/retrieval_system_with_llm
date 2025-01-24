import argparse
import importlib
import os
import numpy as np

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda"

def load_model():
    model_name = "OpenLLM-Ro/RoLlama2-7b-Base"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_quant_type="nf4", 
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Model loaded successfully with 4-bit quantization!")
    return model, tokenizer

def summarize_context(relevant_docs, tokenizer, max_tokens):
    combined_text = " ".join(relevant_docs)
    tokenized = tokenizer(combined_text, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)

def query_system(query, relevant_docs, llm_model, tokenizer):
    summarized_context = summarize_context(relevant_docs, tokenizer, max_tokens=2048)
    input_text = f"<context> {summarized_context}\n <intrebare> {query}\n <raspuns> "
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=100
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def query_without_context(query, llm_model, tokenizer):
    input_text = f"<intrebare> {query}\n <raspuns> "
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=100
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", default=r"C:\Users\razva\Master1\An2\IRTM\Project2\docs")
    parser.add_argument("--ir_system", default="base") # base or kdtree
    parser.add_argument("--type", default="ir") # ir or llm or rag
    args = parser.parse_args()

    retriev_module = importlib.import_module(f"retrieval_system_{args.ir_system}.retriev_documents")

    query_file = "query.txt"
    with open(query_file, "r", encoding="utf-8") as file:
        queries = file.readlines()

    for query in queries:
        if args.type in ["ir", "rag"]:
            if args.ir_system == "base":
                relevant_docs, relevant_doc_names, _, _ = retriev_module.retriev(query, args.doc_path)
            elif args.ir_system == "kdtree":
                embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2").to(device)
                relevant_docs, relevant_doc_names = retriev_module.retriev(query, args.doc_path, embedding_model)

        if args.type == "ir":
            print(relevant_doc_names)
            continue

        llm_model, tokenizer = load_model()

        if args.type == "rag":
            answear = query_system(query, relevant_docs, llm_model, tokenizer)
        elif args.type == "llm":
            answear = query_without_context(query, llm_model, tokenizer)

        answear = answear.split("<intrebare>")[1]
        answear += "\n\n"
        print(answear)
        output_file = f"{args.ir_system}_{args.type}.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(answear)



        
    
