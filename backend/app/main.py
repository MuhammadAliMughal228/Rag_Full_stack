# from fastapi import FastAPI, File, UploadFile, HTTPException
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_groq import ChatGroq
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import os
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from typing import Dict

# load_dotenv()

# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
# os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# app = FastAPI()

# # Global state for storing vectorstores and chat histories
# vectorstore = None
# store: Dict[str, BaseChatMessageHistory] = {}

# # Initialize LLM
# llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Gemma2-9b-It")

# class QuestionRequest(BaseModel):
#     session_id: str
#     question: str

# @app.post("/upload-pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     global vectorstore
#     try:
#         # Save the uploaded file temporarily
#         file_path = f"./temp_{file.filename}"
#         with open(file_path, "wb") as f:
#             f.write(file.file.read())

#         # Load and process the PDF
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#         splits = text_splitter.split_documents(documents)
#         vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

#         return {"message": "PDF processed successfully", "filename": file.filename}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ask/")
# async def ask_question(request: QuestionRequest):
#     global vectorstore, store
#     if not vectorstore:
#         raise HTTPException(status_code=400, detail="No PDF uploaded yet")

#     retriever = vectorstore.as_retriever()

#     # Contextualize question prompt
#     contextualize_q_system_prompt = (
#         "Given a chat history and the latest user question "
#         "which might reference context in the chat history, "
#         "formulate a standalone question which can be understood "
#         "without the chat history. Do NOT answer the question, "
#         "just reformulate it if needed and otherwise return it as is."
#     )
#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#     # Answer question prompt
#     system_prompt = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. Only answer if the context is given. If you don't get any context, say 'I don't know.' "
#         "\n\n"
#         "{context}"
#     )
#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     # Get or create session history
#     if request.session_id not in store:
#         store[request.session_id] = ChatMessageHistory()
#     session_history = store[request.session_id]

#     # Invoke the RAG chain
#     response = rag_chain.invoke(
#         {"input": request.question},
#         config={"configurable": {"session_id": request.session_id}},
#     )

#     # Update chat history
#     session_history.add_user_message(request.question)
#     session_history.add_ai_message(response['answer'])

#     return {"answer": response['answer'], "chat_history": session_history.messages}






from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict
import pinecone
from pinecone import ServerlessSpec
from app.settings import GROQ_API_KEY,OPENAI_API_KEY,PINECONE_API_KEY

import logging

# Create a logger
logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index"
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"Deleted index: {index_name}")

# Check if the index exists, if not, create one
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = Pinecone(index, embeddings, "text")

app = FastAPI()

# Global state for storing chat histories
store: Dict[str, BaseChatMessageHistory] = {}

# Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

class QuestionRequest(BaseModel):
    session_id: str
    question: str

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    logger.info(f"File content type: {file.content_type}")
    logger.info(f"File size: {file.size}")

    try:
        # Save the uploaded file temporarily
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        # Load and process the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Insert into Pinecone
        vectorstore.add_documents(splits)

        return {"message": "PDF processed successfully", "filename": file.filename}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    global vectorstore, store

    if not vectorstore:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")

    retriever = vectorstore.as_retriever()

    # Contextualizing user question for better retrieval
    contextualize_q_system_prompt = (
        "Your task is to rewrite the user's question so it is self-contained, "
        "without relying on previous chat history. "
        "Do NOT answer the question. Simply return a well-formed version of the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 🔴 STRICT SYSTEM PROMPT (Prevents out-of-context answers)
    system_prompt = (
        "You are an AI assistant that answers questions **only using the provided retrieved context**. "
        "If the context does not contain relevant information, respond with:\n\n"
        "'I don’t know. The provided documents do not contain relevant information.'\n\n"
        "Strictly **do not generate responses based on prior knowledge or external information**.\n"
        "Never attempt to answer a question unless supporting evidence is found in the context.\n\n"
        "🔹 **Retrieved Context**:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Get or create session history
    if request.session_id not in store:
        store[request.session_id] = ChatMessageHistory()
    session_history = store[request.session_id]

    # Prepare input for RAG pipeline
    input_data = {
        "input": request.question,
        "chat_history": session_history.messages,  # Maintain conversation memory
    }

    # 🔥 **Invoke the RAG pipeline**
    response = rag_chain.invoke(
        input_data,
        config={"configurable": {"session_id": request.session_id}},
    )

    # 🔴 **Check if response is out-of-context**
    if not response['answer'].strip() or response['answer'].lower() in ["i don’t know", "i don't know"]:
        final_answer = "I don’t know. The provided documents do not contain relevant information."
    else:
        final_answer = response['answer']

    # Update chat history
    session_history.add_user_message(request.question)
    session_history.add_ai_message(final_answer)

    return {"answer": final_answer, "chat_history": session_history.messages}

