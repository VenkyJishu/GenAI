import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import PyPDF2  # Use PyPDF2 for more explicit page manipulation

# Load environment variables
load_dotenv()

# Initialize the Groq API key for language model
groq_api_key = os.getenv("GROQ_API_KEY")
st.title("Custom PDF Chatbot")

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Function to extract text from specific pages using PyPDF2
def extract_text_from_pdf(file_path, page_limit=1, words_per_page=100):
    # Read the PDF file and extract text from the first `page_limit` pages
    pdf_reader = PyPDF2.PdfReader(file_path)
    num_pages = len(pdf_reader.pages)

    # Limit the number of pages to `page_limit`
    limited_text = ""
    for i in range(min(page_limit, num_pages)):  # Ensure we don't go beyond available pages
        page = pdf_reader.pages[i]
        text = page.extract_text()
        
        # Split the text into words and limit the number of words per page
        words = text.split()
        limited_words = words[:words_per_page]
        limited_text += " ".join(limited_words) + "\n"  # Add the content of the page
    
    return limited_text

# Function to load PDFs and process them into embeddings
def process_pdfs(limit_pages=1, words_per_page=100):
    # Choose a folder to load PDFs from
    base_path = os.path.join(os.getcwd(), "..", "..")  # Move 2 steps up
    pdf_folder = os.path.join(base_path, "pdf_files")  # Now access the "pdf_files" directory
    
    # Load PDFs
    pdf_loader = PyPDFDirectoryLoader(pdf_folder)
    documents = pdf_loader.load()

    if not documents:
        st.error("No PDFs were loaded.")
        return None

    # Limit the document to the first few pages and words per page
    limited_documents = []
    for doc in documents:
        # Use PyPDF2 to extract text from the first `limit_pages` pages and `words_per_page` words per page
        pdf_file_path = doc.metadata["source"]
        limited_text = extract_text_from_pdf(pdf_file_path, limit_pages, words_per_page)
        print(f" limited_text is {limited_text}")
        
        # Create a new Document object with the updated content
        limited_document = doc.__class__(page_content=limited_text, metadata=doc.metadata)
        limited_documents.append(limited_document)
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=1)
    split_docs = text_splitter.split_documents(limited_documents)
    
    # Create a vector store from the documents and embeddings
    faiss_vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    return faiss_vectorstore

# Create vector store when embedding the documents
if st.button("Embed Documents"):
    faiss_vectorstore = process_pdfs(limit_pages=1, words_per_page=100)  # Process first 3 pages with 200 words per page
    
    if faiss_vectorstore:
        st.success("Vector store initialized and documents are embedded.")

# Function to handle query and return an answer
def get_answer(query, faiss_vectorstore):
    # Retrieve the documents related to the query
    retriever = faiss_vectorstore.as_retriever()
    
    # Create a retrieval chain for the RAG system
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    # Get the response from the model
    response = chain.run(query)
    
    return response

# Input for user query
query = st.text_input("Ask a question:")

if query:
    if 'faiss_vectorstore' not in locals():
        st.error("Please embed the documents first by clicking the 'Embed Documents' button.")
    else:
        # Fetch the answer for the query
        response = get_answer(query, faiss_vectorstore)
        st.write("Answer:", response)


