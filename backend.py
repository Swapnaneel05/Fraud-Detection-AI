import os
import re
import pandas as pd
import torch
import transformers
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

# --- CONSTANTS FROM YOUR NOTEBOOK ---
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
PERSIST_DIRECTORY = 'docs/chroma_rag/'
DATA_PATH = 'SMS fraud Dataset.csv'

# --- PRE-DEPLOYMENT DATA LOGIC ---
def initialize_vector_store():
    # If the database doesn't exist, create it exactly using your notebook logic
    if not os.path.exists(PERSIST_DIRECTORY):
        print("Creating Chroma DB from CSV...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        
        df = pd.read_csv(DATA_PATH)
        
        # Your exact clean_text logic
        def clean_text(sms):
            sms = sms.encode('ascii', 'ignore').decode()
            sms = re.sub(r'[^\w\s]', '', sms)
            sms = re.sub(r'\d+', '', sms)
            sms = sms.lower()
            tokens = word_tokenize(sms)
            stop_words = set(stopwords.words('english'))
            return [word for word in tokens if word not in stop_words]

        df['cleaned_sms'] = df['sms'].apply(clean_text)
        df['label'] = df['label'].replace({0: 'non-fraud', 1: 'fraud'})
        
        documents = []
        for i, row in df.iterrows():
            # Your exact document structure
            doc_content = f"id:{i}\\Fillings:{row['cleaned_sms']}\\Fraud_Status:{row['label']}"
            documents.append(Document(page_content=doc_content))
        
        hg_embeddings = HuggingFaceEmbeddings()
        return Chroma.from_documents(
            documents=documents,
            collection_name="fraud_data",
            embedding=hg_embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
    else:
        return Chroma(persist_directory=PERSIST_DIRECTORY, 
                      embedding_function=HuggingFaceEmbeddings(), 
                      collection_name="fraud_data")

# Initialize Model & Logic
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, max_new_tokens=1024)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=model_config, quantization_config=bnb_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

query_pipeline = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, 
    max_length=6000, max_new_tokens=500, device_map='auto'
)
llm = HuggingFacePipeline(pipeline=query_pipeline)

# Setup RAG Chain
vector_store = initialize_vector_store()
template = """
You are a Fraud Detector. Analyse them and Predict is the Given Statement is Fraud or not? If you don't know return"I am Sorry, I don't know the answer! Reply with "Fraud" and "Non-Fraud",no hallucination,no extra words"
Question:{input}
Context:{context}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "input"])
qa_chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 1}), 
                                   create_stuff_documents_chain(llm, prompt))

class SMSRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: SMSRequest):
    result = qa_chain.invoke({"input": request.text})
    # This returns the raw output for your frontend to handle
    return {"answer": result["answer"]}