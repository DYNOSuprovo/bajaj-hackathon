# build_index.py
# This script is responsible for the one-time processing of PDF documents.
# It loads PDFs from a specified directory, splits them into manageable chunks,
# generates vector embeddings for each chunk, and stores them in a local
# ChromaDB vector database for efficient retrieval.

import os
import fitz  # PyMuPDF library for PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# --- Configuration ---
# Directory where your PDF policy documents are stored.
# Ensure you have a folder named 'data' with the 5 PDFs inside.
DATA_PATH = "data" 
# Directory where the local vector database will be saved.
DB_PATH = "chroma_db" 

def load_documents():
    """
    Loads all PDF documents from the specified data directory.
    Extracts text content from each page of each PDF.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains the 
              page content and metadata (source filename) for a document.
    """
    documents = []
    print("Starting to load documents...")
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_PATH, filename)
            try:
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                
                # We store the full text and the source filename for later reference.
                documents.append({
                    "page_content": text,
                    "metadata": {"source": filename}
                })
                print(f"Successfully loaded and extracted text from: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    print(f"\nLoaded a total of {len(documents)} documents.")
    return documents

def chunk_documents(documents):
    """
    Splits the loaded documents into smaller, semantically meaningful chunks.
    This is crucial for the retrieval model to find the most relevant context.
    
    Args:
        documents (list): A list of document dictionaries from load_documents.
        
    Returns:
        list: A list of LangChain Document objects, each representing a chunk.
    """
    print("Chunking documents...")
    # This text splitter is designed to break up large texts while trying to
    # maintain semantic context by splitting on paragraphs, sentences, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,       # The maximum size of each chunk.
        chunk_overlap=200,     # The overlap between consecutive chunks to preserve context.
        length_function=len
    )
    
    # LangChain's vector stores expect input in the form of `Document` objects.
    langchain_docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]
    
    chunks = text_splitter.split_documents(langchain_docs)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def create_and_store_embeddings(chunks):
    """
    Generates vector embeddings for each chunk and stores them in a persistent
    Chroma vector database on your local disk.
    
    Args:
        chunks (list): A list of LangChain Document chunks.
        
    Returns:
        Chroma: The created vector store object.
    """
    print("Generating embeddings and creating vector store...")
    # We use a high-quality, open-source sentence-transformer model to create the embeddings.
    # This model runs entirely on your local machine.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the vector store from the document chunks.
    # This process will automatically generate embeddings for each chunk.
    # The `persist_directory` argument tells Chroma to save the database to disk.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Vector store created successfully.")
    return vector_store

if __name__ == "__main__":
    print("--- Starting Document Indexing Pipeline ---")
    
    # Step 1: Load the PDF documents.
    raw_documents = load_documents()
    document_chunks = chunk_documents(raw_documents)
    create_and_store_embeddings(document_chunks)
    
    print("\n--- Indexing Complete ---")
    print(f"The searchable vector database has been created and is stored in the '{DB_PATH}' directory.")
    print("You can now run main.py to start the query API.")
