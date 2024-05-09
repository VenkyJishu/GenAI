import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
import time

load_dotenv() # imports .env variables to current environment


#loading Groq API Key and OpenAI API Key

groq_api_key = os.getenv('GROQ_API_KEY') # generate key from https://console.groq.com/keys portal
#os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

st.title("Groq Demo with Llama3 ")
llm = ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provode most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}

        """
    )

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        #Ingesting "pdf" directory files
        st.session_state.loader = PyPDFDirectoryLoader("./HuggingFace/pdf")
        #Loading documents which are in above "pdf" directory
        st.session_state.docs = st.session_state.loader.load()
        #Creating chunks 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        #Splitting documents
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:])
        #Creating Vector Store using OpenAI embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) 


prompt_new = st.text_input("Enter your query on existed PDF Doc")
if st.button("Doc/PDF Embeddig "):
    vector_embedding()
    st.write("Vector Store DB is ready ")


import time
print(" prompt_new is ",prompt_new)
if prompt_new:
    
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt_new})
    print("Response Time ",time.process_time()-start)
    st.write(response['answer'])


    with st.expander("Docoments Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write(".................")