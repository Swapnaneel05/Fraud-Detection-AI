import streamlit as st
import pandas as pd
import re
import os
import nltk
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Investigator Pro", page_icon="🛡️")

# --- INITIALIZE SYSTEM (Cached for efficiency) ---
@st.cache_resource
def initialize_rag():
    # Download NLTK resources exactly as in your notebook
    nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'], quiet=True)
    
    # 1. Your Exact Cleaning Logic
    def clean_text(sms):
        sms = str(sms).encode('ascii', 'ignore').decode()
        sms = re.sub(r'[^\w\s]', '', sms)
        sms = re.sub(r'\d+', '', sms)
        sms = sms.lower()
        tokens = word_tokenize(sms)
        stop_words = set(stopwords.words('english'))
        return " ".join([word for word in tokens if word not in stop_words])

    # 2. Build Vector Store from your uploaded CSV
    if os.path.exists('SMS fraud Dataset.csv'):
        df = pd.read_csv('SMS fraud Dataset.csv')
        df['cleaned'] = df['sms'].apply(clean_text)
        
        # Structure documents based on your notebook's format
        documents = []
        for i, row in df.iterrows():
            content = f"id:{i}\\Fillings:{row['cleaned']}\\Fraud_Status:{row['label']}"
            documents.append(Document(page_content=content))
        
        # Use lightweight embeddings for speed
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory="./chroma_db"
        )
        return vector_db, clean_text
    else:
        st.error("Missing file: SMS fraud Dataset.csv")
        return None, None

# --- LLAMA 3 INFERENCE LOGIC ---
def call_llama_api(prompt_text):
    # Meta-Llama-3-8B-Instruct is safer and more reliable than Zephyr-7B
    API_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    payload = {
    "inputs": prompt_text,
    "parameters": {
        "max_new_tokens": 100, 
        "temperature": 0.01,
        "return_full_text": False
    },
    "options": {
        "wait_for_model": True  # This prevents the 503 error
    }
}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error {response.status_code}: Model loading or API limit."

# --- MAIN INTERFACE ---
st.title("🛡️ Fraud SMS Investigator")
st.write("Using custom RAG knowledge and Llama 3 intelligence.")

vector_db, clean_func = initialize_rag()

if vector_db:
    user_input = st.text_area("Enter suspicious SMS text:", height=100)

    if st.button("Detect Fraud"):
        if user_input:
            with st.spinner("Analyzing message..."):
                # 1. RAG step: Retrieve top 3 relevant examples (k=3 for better context)
                results = vector_db.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in results])
                
                # 2. Optimized Prompt for Llama 3
                llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a professional Fraud Detector. Use the provided context to analyze the message.
                If the message is fraud, reply with "Verdict: Fraud". 
                If the message is safe, reply with "Verdict: Non-Fraud".
                Context: {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
                Is this message fraud? {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                # 3. Get Verdict
                raw_response = call_llama_api(llama_prompt)
                verdict = raw_response.lower()

                # 4. Display Results with robust keyword checking
                st.subheader("Verdict:")
                if "non-fraud" in verdict:
                    st.success(f"✅ SAFE: {raw_response}")
                elif "fraud" in verdict:
                    st.error(f"⚠️ FRAUD: {raw_response}")
                else:
                    st.warning(f"Ambiguous response: {raw_response}")
                
                with st.expander("View Retrieved Evidence"):
                    st.write(context)

