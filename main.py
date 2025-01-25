import argparse
import importlib
import os
from xml.dom.minidom import Document
import PyPDF2
import numpy as np

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda"

def load_model():
    model_name = "OpenLLM-Ro/RoLlama2-7b-Base"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
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
        output_hidden_states=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Model loaded successfully with 4-bit quantization!")
    return model, tokenizer

def summarize_context(relevant_docs, tokenizer, max_tokens):
    combined_text = " ".join(relevant_docs)
    tokenized = tokenizer(combined_text, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)

def query_system(query, relevant_docs, llm_model, tokenizer):
    summarized_context = summarize_context(relevant_docs, tokenizer, max_tokens=1024)
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

def get_embeddings(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states 
    
    last_hidden_state = hidden_states[-1]
    
    embeddings = last_hidden_state.mean(dim=1)
    return embeddings.cpu()

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

    return texts, files

def get_best_document_id(query, relevant_docs, llm_model):
    query_embed = get_embeddings(query, llm_model)
    best_sim = -2
    best_doc = None
    
    for id, doc in enumerate(relevant_docs):
        doc_embed = get_embeddings(doc, llm_model)
        cos_sim = F.cosine_similarity(query_embed, doc_embed)

        if cos_sim > best_sim:
            best_sim = cos_sim
            best_doc = id

    return best_doc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", default=r"C:\Users\razva\Master1\An2\IRTM\Project2\docs")
    parser.add_argument("--ir_system", default="kdtree") # base or kdtree
    parser.add_argument("--type", default="rag") # ir or llm or rag
    parser.add_argument("--qa", default="False")
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
            print(query)
            print(relevant_doc_names)
            output_file = f"{args.ir_system}_{args.type}.txt"
            with open(output_file, "a", encoding="utf-8") as file:
                file.write(f"{query} \n {str(relevant_doc_names)}")
            continue

        llm_model, tokenizer = load_model()

        if args.qa == "True":
            if args.type == "rag":
                answear = query_system(query, relevant_docs, llm_model, tokenizer)
            elif args.type == "llm":
                answear = query_without_context(query, llm_model, tokenizer)

            answear = answear.split("<intrebare>")[1]
            answear += "\n\n"
            print(answear)
            output_file = f"{args.ir_system}_{args.type}.txt"
            with open(output_file, "a", encoding="utf-8") as file:
                file.write(answear)
        elif args.qa == "False":
            if args.type == "rag":
                best_doc_id = get_best_document_id(query, relevant_docs, llm_model)
            elif args.type == "llm":
                texts, relevant_doc_names = load_docs(args.doc_path)
                best_doc_id = get_best_document_id(query, texts, llm_model)

            best_doc = relevant_doc_names[best_doc_id]

            print(query)
            print(best_doc)
            output_file = f"{args.ir_system}_{args.type}_{args.qa}.txt"
            with open(output_file, "a", encoding="utf-8") as file:
                file.write(f"{query} \n {str(best_doc)}")
            



        
    
