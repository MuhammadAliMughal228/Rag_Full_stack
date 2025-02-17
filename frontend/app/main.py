import streamlit as st
import requests
# FastAPI backend URL
BACKEND_URL = "http://backend:8000"

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }
    /* Chat message styling */
    .stChatMessage {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 75%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    /* Hover effect on chat messages */
    .stChatMessage:hover {
        transform: scale(1.02);
    }
    /* User message styling */
    .stChatMessage[data-role="user"] {
        background-color: #0078d4;
        color: white;
        margin-left: auto;
        margin-right: 0;
    }
    /* Assistant message styling */
    .stChatMessage[data-role="assistant"] {
        background-color: #ffffff;
        color: black;
        margin-left: 0;
        margin-right: auto;
        border: 1px solid #ddd;
    }
    /* Chat input styling */
    .stTextInput>div>div>input {
        border-radius: 25px;
        padding: 12px;
        font-size: 16px;
        border: 2px solid #0078d4;
        transition: border-color 0.3s ease-in-out;
    }
    /* Focus effect on input */
    .stTextInput>div>div>input:focus {
        border-color: #0056b3;
        box-shadow: 0 0 8px rgba(0, 120, 212, 0.3);
    }
    /* Title styling */
    .stTitle {
        font-size: 36px;
        font-weight: bold;
        color: #0078d4;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* Upload button styling */
    .stFileUploader>div>div>button {
        background: linear-gradient(135deg, #0078d4, #0056b3);
        color: white;
        border-radius: 25px;
        padding: 12px 25px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: transform 0.2s ease-in-out;
    }
    /* Hover effect on upload button */
    .stFileUploader>div>div>button:hover {
        transform: scale(1.05);
    }
    /* Success message styling */
    .stSuccess {
        color: #28a745;
        font-weight: bold;
        font-size: 18px;
        animation: fadeIn 1s ease-in-out;
    }
    /* Error message styling */
    .stError {
        color: #dc3545;
        font-weight: bold;
        font-size: 18px;
        animation: shake 0.5s ease-in-out;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        50% { transform: translateX(5px); }
        75% { transform: translateX(-5px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="stTitle">Conversational RAG With PDF Uploads and Chat History</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=False)
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
user_input = st.chat_input("Ask your question...")
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
