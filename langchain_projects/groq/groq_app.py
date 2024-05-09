'This code is used read files from local directory and answer to our questions using Llama3 Model along with stremlit'

#streamlit: For building web apps with Python.
#os: For interacting with the operating system.
#langchain_groq: Specific to Groq's language understanding capabilities.
#langchain_openai: For using OpenAI's embeddings.
#langchain.text_splitter, langchain.chains, langchain.vectorstores: Modules for various language processing tasks.
#dotenv: For loading environment variables from a .env file.
#time: For timing operations.


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
#Initializing the Groq model with the provided API key and model.
llm = ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")
#Defining a prompt template for providing context and questions.
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
    '''
    This function prepares the document embeddings:
    It checks if the embeddings are already loaded in the session state.
    If not, it initializes OpenAI embeddings, loads PDF documents, splits them into chunks, 
    and creates document embeddings using FAISS.
    
    '''
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        #Ingesting "pdf" directory files
        st.session_state.loader = PyPDFDirectoryLoader("./HuggingFace/pdf")
        #Loading documents which are in above "pdf" directory
        st.session_state.docs = st.session_state.loader.load()
        # Splitting documents into chunks 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
       
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:])
        #Creating Vector Store using OpenAI embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



if st.button("Doc/PDF Embedding "):
    vector_embedding()
    st.write("Vector Store DB is ready ")

prompt_new = st.text_input("Enter your query on existed PDF Doc")
import time
print(" prompt_new is ",prompt_new)
if prompt_new:
    
    #Below code checks if there's a user query (prompt_new).
    #If there is, it creates a retrieval chain using Groq and the document vectors stored in the session state.
    #Then, it invokes the retrieval chain with the user's query and stores the response.
    #Finally, it displays the answer to the user's query using st.write()
    
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt_new})
    print("Response Time ",time.process_time()-start)
    st.write(response['answer'])

    
    #Below code creates an expander widget titled "Documents Similarity Search". When expanded, 
    #it iterates over the documents returned in the response and displays their page content along with a separator.
    

    with st.expander("Docoments Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write(".................")
