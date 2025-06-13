import streamlit as st
import time
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
# Page config
from model import embeddings, llm, load_qa_chain,api_key 
import os

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history and uploaded PDFs
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = []
    
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

if "document_search" not in st.session_state:
    st.session_state.document_search = None

# Path to pre-built FAISS index directory
FAISS_INDEX_PATH = "faiss_index"

# Load FAISS index from disk if available
if "document_search" not in st.session_state or st.session_state.document_search is None:
    if os.path.exists(FAISS_INDEX_PATH):
        try:
           
            st.session_state.document_search = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
            st.info("Loaded pre-built FAISS index from disk.")
        except Exception as e:
            st.warning(f"Could not load FAISS index: {e}")
    else:
        st.session_state.document_search = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    raw_text = ''
    pdfreader = PyPDF2.PdfReader(pdf_file)
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to extract text from TXT file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Initialize QA chain
chain = load_qa_chain(llm)

# App title
st.title("🤖 AI Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            # Placeholder for actual AI integration
            # This is where you would call your AI model
            if st.session_state.document_search:
                docs = st.session_state.document_search.similarity_search(user_input)
                response = chain.run(input_documents=docs, question=user_input)
            else:
                   response = chain.run(input_documents=[], question=user_input)
            
            # Simulate typing effect
            full_response = ""
            for word in response.split():
                full_response += word + " "
                message_placeholder.write(full_response)
                time.sleep(0.05)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with instructions and PDF/text upload
with st.sidebar:
    st.subheader("About")
    if api_key=="":
        st.write("No api key provided.")
    else:
        st.write("api key provided.")
    st.write("This is a simple chatbot UI template.")
    st.write("Ask questions to see the interface in action!")
    
    # PDF and TXT Upload section
    st.subheader("Upload PDF or Text Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files to chat about their content", 
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        all_texts = []
        for file in uploaded_files:
            # Check if this file has already been processed
            if file.name not in st.session_state.pdf_names:
                try:
                    # Extract text based on file type
                    if file.type == "application/pdf":
                        raw_text = extract_text_from_pdf(file)
                    elif file.type == "text/plain":
                        raw_text = extract_text_from_txt(file)
                    else:
                        st.warning(f"Unsupported file type: {file.name}")
                        continue

                    # Store the file text and name
                    st.session_state.pdf_texts.append(raw_text)
                    st.session_state.pdf_names.append(file.name)
                    all_texts.append(raw_text)
                    
                    st.success(f"Processed: {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
        
        if all_texts:
            # Process all texts together
            text_splitter = CharacterTextSplitter(
                separator = "\n",
                chunk_size = 800,
                chunk_overlap = 200,
                length_function = len,
            )
            
            combined_text = " ".join(st.session_state.pdf_texts)
            texts = text_splitter.split_text(combined_text)
            st.session_state.document_search = FAISS.from_texts(texts, embeddings)
    
    # Display currently loaded files
    if st.session_state.pdf_names:
        st.subheader("Loaded Files")
        for file_name in st.session_state.pdf_names:
            st.write(f"- {file_name}")
        
        # Option to clear loaded files
        if st.button("Clear Files"):
            st.session_state.pdf_texts = []
            st.session_state.pdf_names = []
            st.session_state.document_search = None
            st.rerun()
    
    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
