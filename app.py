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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Llama 3 Fraud Detector", page_icon="🛡️")

# --- CACHED INITIALIZATION (Prevents crashes and slow starts) ---
@st.cache_resource
def setup_system():
    # Ensuring NLTK data is downloaded on the server
    nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'], quiet=True)
    stop_words = set(stopwords.words('english'))

    # Your exact cleaning logic from the notebook
    def clean_text(sms):
        sms = str(sms).encode('ascii', 'ignore').decode()
        sms = re.sub(r'[^\w\s]', '', sms)
        sms = re.sub(r'\d+', '', sms)
        sms = sms.lower()
        tokens = word_tokenize(sms)
        return " ".join([word for word in tokens if word not in stop_words])

    # Building Vector Store from your CSV
    if os.path.exists('SMS fraud Dataset.csv'):
        df = pd.read_csv('SMS fraud Dataset.csv')
        df['cleaned'] = df['sms'].apply(clean_text)
        df['label'] = df['label'].replace({0: 'non-fraud', 1: 'fraud'})
        
        documents = []
        for i, row in df.iterrows():
            # Your exact document structure: id, Fillings, Fraud_Status
            content = f"id:{i}\\Fillings:{row['cleaned']}\\Fraud_Status:{row['label']}"
            documents.append(Document(page_content=content))
        
        # Using lightweight embeddings to stay within 1GB RAM limit
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory="./chroma_db"
        )
        return vector_db, clean_text
    else:
        st.error("Error: 'SMS fraud Dataset.csv' not found in repository.")
        return None, None

# --- LLAMA 3 INFERENCE API (Bypasses Memory Limit) ---
def query_llama3(prompt_text):
    # Using the new Hugging Face Router endpoint
    API_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 150, 
            "temperature": 0.01, # Set low for consistent fraud detection
            "return_full_text": False
        },
        "options": {"wait_for_model": True} # Prevents 503 errors on startup
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Model Error: {response.status_code}. AI is currently busy or token is invalid."

# --- INTERACTIVE UI ---
st.title("🛡️ SMS Fraud Investigation (Llama 3)")
st.info("This system uses custom RAG context and Meta Llama 3 intelligence.")

vector_db, clean_func = setup_system()

if vector_db:
    user_input = st.text_area("Enter suspicious SMS text to analyze:", height=150)

    if st.button("Start AI Investigation"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Searching database and consulting Llama 3..."):
                # 1. RAG retrieval (k=3 for better context accuracy)
                results = vector_db.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in results])
                
                # 2. Llama 3 Prompt Formatting (Native Instruct format)
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a Financial Fraud Investigator. Use the provided Context to analyze the Message.
                If it is fraud, reply with "Verdict: Fraud". If safe, reply with "Verdict: Non-Fraud".
                Context: {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
                Message to Analyze: {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                # 3. Get AI Verdict
                raw_response = query_llama3(prompt)
                verdict = raw_response.lower()

                # 4. Results Block
                st.subheader("Investigation Results")
                if "non-fraud" in verdict:
                    st.success(f"✅ SAFE: {raw_response}")
                elif "fraud" in verdict:
                    st.error(f"⚠️ POTENTIAL FRAUD: {raw_response}")
                else:
                    st.warning(f"Ambiguous response from AI: {raw_response}")
                
                with st.expander("Show Evidence (Retrieved Data)"):
                    st.write(context)
