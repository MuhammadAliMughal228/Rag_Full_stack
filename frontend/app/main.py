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

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Your question:")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Send question to backend
    response = requests.post(
        f"{BACKEND_URL}/ask/",
        json={"session_id": "default_session", "question": user_input}
    )

    if response.status_code == 200:
        data = response.json()

        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": data['answer']})

        # Display assistant message
        with st.chat_message("assistant"):
            st.write(data['answer'])
    else:
        st.error("Failed to get a response from the chatbot.")


