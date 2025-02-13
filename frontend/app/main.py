import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://backend:8000"

st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload-pdf/",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
        )
        if response.status_code == 200:
            st.success("PDF uploaded successfully!")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Exception: {str(e)}")


# Chat interface
session_id = "abc"
user_input = st.text_input("Your question:")

if user_input:
    response = requests.post(
        f"{BACKEND_URL}/ask/",
        json={"session_id": session_id, "question": user_input}
    )
    if response.status_code == 200:
        data = response.json()
        st.write("Assistant:", data['answer'])
        # st.write("Chat History:")
        # for message in data['chat_history']:
        #     st.write(f"{message.type}: {message.content}")
    else:
        st.error("Failed to get a response from the chatbot.")
