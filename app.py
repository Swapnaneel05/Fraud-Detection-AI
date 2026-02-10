import streamlit as st
import requests

st.title("🛡️ Fraud Detection System")
sms_input = st.text_area("Enter SMS to analyze:")

if st.button("Check for Fraud"):
    if sms_input:
        with st.spinner("Analyzing..."):
            # REPLACE with your deployed Render URL
            URL = "https://your-backend-name.onrender.com/predict" 
            response = requests.post(URL, json={"text": sms_input})
            
            if response.status_code == 200:
                answer = response.json().get("answer")
                # Add block to display output
                st.subheader("Result:")
                if "Fraud" in answer and "Non-Fraud" not in answer:
                    st.error(answer)
                else:
                    st.success(answer)
            else:
                st.error("Connection to AI engine failed.")