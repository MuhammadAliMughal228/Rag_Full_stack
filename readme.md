# Conversational RAG with PDF Uploads and Chat History

## 📌 Overview
This project is a **Conversational Retrieval-Augmented Generation (RAG) chatbot** that allows users to:
- **Upload PDF documents** and extract their content into a vector database.
- **Ask questions** about the uploaded PDFs using AI-powered retrieval and response generation.
- **Store chat history** for maintaining conversational context.

The system is built using **FastAPI (backend)**, **Streamlit (frontend)**, **Pinecone (vector storage)**, and **OpenAI embeddings & LLMs**.

---

## 🏗️ Project Structure
```plaintext
📂 Conversational-RAG-Chatbot
│── 📂 backend          # FastAPI Backend (Handles AI interactions, vector storage, PDF processing)
│   ├── app
│   │   ├── main.py      # Main FastAPI application
│   │   ├── settings.py  # API keys and configurations
│   │   ├── models.py    # Data models (chat history, queries)
│   │   ├── utils.py     # Helper functions for processing data
│── 📂 frontend         # Streamlit UI (User interaction)
│   ├── app.py          # Main Streamlit application
│── 📂 data             # Stores uploaded PDFs (temporary storage)
│── 📂 docker           # Docker setup files
│── .env               # Environment variables (API Keys, Database info)
│── docker-compose.yml  # Docker Compose configuration
│── README.md           # This guide
```

---

## 🔧 Prerequisites
Before you begin, make sure you have the following installed:

- **Docker & Docker Compose**
- **Python 3.9+** (If running outside Docker)
- **Pinecone, OpenAI, Groq API Keys**
- **Poetry** (For dependency management)

---

## 🚀 Setup & Installation
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/ahadnaeem785/Conversational-RAG-Chatbot.git
cd Conversational-RAG-Chatbot
```

### **2️⃣ Set Up Environment Variables**
Create a **`.env`** file in the root directory and add:
```plaintext
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
GROQ_API_KEY=your_groq_api_key
```

### **3️⃣ Run the Application (Using Docker)**
```sh
docker-compose up --build
```
This will:
- Start **FastAPI (Backend) on port `8000`**
- Start **Streamlit UI (Frontend) on port `8501`**

**Access the UI:** `http://localhost:8501`
**API Docs:** `http://localhost:8000/docs`


---

## 📌 Usage Guide
### **🔹 Upload a PDF**
1. Open `http://localhost:8501`
2. Click **"Choose a PDF File"** and upload a document.
3. Once uploaded, the document content is processed and stored in Pinecone.

### **🔹 Ask Questions About the PDF**
1. Enter your **Session ID** (default: `default_session`).
2. Type a **question related to the uploaded PDF**.
3. The chatbot will retrieve relevant information and respond based on the document content.
4. Chat history will be displayed on the UI.

### **🔹 API Endpoints (For Developers)**
| Method | Endpoint          | Description |
|--------|-----------------|-------------|
| `POST` | `/upload-pdf/`   | Uploads and processes a PDF file |
| `POST` | `/ask/`          | Asks a question about the uploaded PDF |
| `GET`  | `/docs`          | Access API documentation |


---

## 🎯 Troubleshooting
### 🔹 **Backend Not Starting?**
- Ensure `.env` file is correctly configured.
- Run `docker-compose logs backend` to check logs.

### 🔹 **Frontend Not Working?**
- Ensure the backend is running before starting Streamlit.
- Run `docker-compose logs frontend` to check logs.

### 🔹 **API Not Responding?**
- Run a manual test: `curl -X POST "http://localhost:8000/ask/" -H "Content-Type: application/json" -d '{"session_id": "default", "question": "What is AI?"}'`

---

## 📌 Future Improvements
- ✅ Support for multiple PDFs
- ✅ UI enhancements for better user experience
- ✅ Deploy on cloud (AWS/GCP/Azure)

---
.

🚀 **Happy Coding!**

