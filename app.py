import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(page_title="AskSmart.ai", layout="wide")
st.title("📄 AskSmart.ai - Document Q&A System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
api_key = st.text_input("Enter your OpenAI API Key", type="password")

if uploaded_file and api_key:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded and saved successfully!")
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load_and_split()
    
    os.environ["OPENAI_API_KEY"] = api_key
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about the document:")
    if query:
        result = qa_chain.run(query)
        st.write("### Answer:", result)
