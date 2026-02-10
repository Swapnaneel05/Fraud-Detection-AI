import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'], quiet=True)
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
st.set_page_config(page_title="Fraud Guard AI", page_icon="🛡️")

# --- INITIALIZE RESOURCES (Cached to run only once) ---
@st.cache_resource
def initialize_system():
    # 1. Setup NLTK exactly as in your notebook
    nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'], quiet=True)
    stop_words = set(stopwords.words('english'))

    # 2. Your Exact Cleaning Logic
    def clean_text(sms):
        sms = str(sms).encode('ascii', 'ignore').decode()
        sms = re.sub(r'[^\w\s]', '', sms)
        sms = re.sub(r'\d+', '', sms)
        sms = sms.lower()
        tokens = word_tokenize(sms)
        return [word for word in tokens if word not in stop_words]

    # 3. Build Vector Store from CSV
    # Ensure 'SMS fraud Dataset.csv' is in your GitHub repo
    if os.path.exists('SMS fraud Dataset.csv'):
        df = pd.read_csv('SMS fraud Dataset.csv')
        df['cleaned_sms'] = df['sms'].apply(clean_text)
        df['label'] = df['label'].replace({0: 'non-fraud', 1: 'fraud'})
        
        documents = []
        for i, row in df.iterrows():
            # Maintaining your exact notebook document format
            doc_content = f"id:{i}\\Fillings:{row['cleaned_sms']}\\Fraud_Status:{row['label']}"
            documents.append(Document(page_content=doc_content))
        
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(),
            persist_directory="./chroma_db"
        )
        return vector_db, clean_text
    else:
        st.error("Dataset not found! Please upload 'SMS fraud Dataset.csv' to your repository.")
        return None, None

# --- AI INFERENCE LOGIC (Serverless for 24/7 Uptime) ---
def call_zephyr_api(prompt_text):
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    # Access token from Streamlit Secrets
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    payload = {
        "inputs": prompt_text,
        "parameters": {"max_new_tokens": 100, "temperature": 0.001}
    }
    retriever = langchain_chroma.as_retriever(search_kwargs={"k": 5})
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return "Error: Could not connect to the AI model."

# --- MAIN APP INTERFACE ---
st.title("🛡️ SMS Fraud Investigator")
st.markdown("Analyze messages using your custom RAG pipeline and Zephyr-7B.")

vector_db, clean_text_func = initialize_system()

if vector_db:
    user_input = st.text_area("Paste the message content:", height=150)

    if st.button("Start Investigation"):
        if not user_input.strip():
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Searching database and analyzing..."):
                # 1. RAG Step: Find most relevant context
                # Clean and search exactly as per your notebook logic
                results = vector_db.similarity_search(user_input, k=1)
                context = results[0].page_content
                
                # 2. Construct Prompt exactly as in your notebook
                template = f"""
                <|system|>
                You are a Fraud Detector. Use the context to answer. 
                Example 1: Win a free trip! -> Answer: Fraud
                Example 2: Meet me at 5pm. -> Answer: Non-Fraud
                </s>
                <|user|>
                Context: {context}
                Analyze: {user_input}
                Answer:</s>
                <|assistant|>"""
                
                # 3. Get AI Verdict
                raw_response = call_zephyr_api(template)
                
                # Extract the final answer part
                final_answer = raw_response.lower()
                if "non-fraud" in final_answer:
                    st.success("✅ SAFE MESSAGE: Non-Fraud")
                elif "fraud" in final_answer:
                    st.error("⚠️ POTENTIAL FRAUD DETECTED: Fraud")
                else:
                    st.warning(f"AI could not determine clearly. Full response: {raw_response}")

st.divider()
st.caption("Powered by LangChain, ChromaDB, and Zephyr-7B-beta.")





