import streamlit as st
import tempfile
import os
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="PDF Notes Q&A Assistant", page_icon="📚")
st.title("📚 Personal PDF / Notes Q&A Bot")

# 1. Grab Gemini API Key safely
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("Please add your GEMINI_API_KEY in Streamlit Secrets!")
    st.stop()

# Initialize official Google GenAI client directly
client = genai.Client(api_key=api_key)

# 2. File Uploader Interface
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Processing PDF and indexing text..."):
        # Load & chunk PDF text
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Local embeddings (bypasses all Google embedding errors)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    st.success("PDF processed successfully! Ask your questions below:")

    # 3. Chat Interface
    user_query = st.text_input("Ask a question about your PDF:")
    if user_query:
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks from the PDF
            relevant_docs = retriever.invoke(user_query)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Construct the prompt
            prompt = f"""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. Keep the answer concise.

Context:
{context_text}

Question: {user_query}"""

            # Call Gemini directly using the official SDK with the correct current model
            response = client.models.generate_content(
                model="gemini-3.8-flash-lite",
                contents=prompt,
            )
            
            st.write("**Answer:**")
            st.write(response.text)
