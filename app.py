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
st.set_page_config(page_title="Llama 3 Fraud Detector", page_icon="🛡️")

@st.cache_resource
def initialize_system():
    # Setup NLTK
    nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'], quiet=True)
    stop_words = set(stopwords.words('english'))

    def clean_text(sms):
        sms = str(sms).encode('ascii', 'ignore').decode()
        sms = re.sub(r'[^\w\s]', '', sms)
        sms = re.sub(r'\d+', '', sms)
        sms = sms.lower()
        tokens = word_tokenize(sms)
        return " ".join([word for word in tokens if word not in stop_words])

    # Build Vector Store (Ensure 'SMS fraud Dataset.csv' is in your GitHub)
    if os.path.exists('SMS fraud Dataset.csv'):
        df = pd.read_csv('SMS fraud Dataset.csv')
        df['cleaned'] = df['sms'].apply(clean_text)
        
        documents = []
        for i, row in df.iterrows():
            content = f"id:{i}\\Fillings:{row['cleaned']}\\Fraud_Status:{row['label']}"
            documents.append(Document(page_content=content))
        
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory="./chroma_db"
        )
        return vector_db, clean_text
    else:
        st.error("Dataset not found! Please upload 'SMS fraud Dataset.csv'.")
        return None, None

# --- UPDATED LLAMA 3 ROUTER API ---
def call_llama_api(prompt_text):
    # This is the NEW stable endpoint for Llama 3
    API_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 100, 
            "temperature": 0.01,
            "return_full_text": False
        },
        "options": {"wait_for_model": True}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    elif response.status_code == 404:
        return "ERROR_404: The model endpoint was not found. Please check if you have accepted the Llama 3 terms on Hugging Face."
    else:
        return f"Error {response.status_code}: {response.text}"

# --- UI ---
st.title("🛡️ SMS Fraud Investigation (Llama 3)")
vector_db, clean_func = initialize_system()

if vector_db:
    user_input = st.text_area("Analyze SMS:", placeholder="Paste message here...")

    if st.button("Start AI Investigation"):
        if user_input:
            with st.spinner("Consulting Llama 3..."):
                results = vector_db.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in results])
                
                # Native Llama 3 Prompt Format
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a Fraud Detector. Use the context to determine if the message is fraud.
                Reply ONLY with "Verdict: Fraud" or "Verdict: Non-Fraud".
                Context: {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
                Analyze this message: {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                raw_response = call_llama_api(prompt)
                verdict = raw_response.lower()

                st.subheader("Verdict:")
                if "non-fraud" in verdict:
                    st.success(f"✅ SAFE: {raw_response}")
                elif "fraud" in verdict:
                    st.error(f"⚠️ FRAUD: {raw_response}")
                else:
                    st.warning(f"Ambiguous: {raw_response}")
