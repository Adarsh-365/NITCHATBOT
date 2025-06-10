import streamlit as st
import time
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
# Page config
from model import embeddings, llm, load_qa_chain

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

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    raw_text = ''
    pdfreader = PyPDF2.PdfReader(pdf_file)
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

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
                response = f"This is a demo response to: {user_input}"
            
            # Simulate typing effect
            full_response = ""
            for word in response.split():
                full_response += word + " "
                message_placeholder.write(full_response)
                time.sleep(0.05)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with instructions and PDF upload
with st.sidebar:
    st.subheader("About")
    st.write("This is a simple chatbot UI template.")
    st.write("Ask questions to see the interface in action!")
    
    # PDF Upload section
    st.subheader("Upload PDF Documents")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF files to chat about their content", 
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_pdfs:
        all_texts = []
        for pdf in uploaded_pdfs:
            # Check if this PDF has already been processed
            if pdf.name not in st.session_state.pdf_names:
                try:
                    # Extract text from PDF
                    raw_text = extract_text_from_pdf(pdf)
                    
                    # Store the PDF text and name
                    st.session_state.pdf_texts.append(raw_text)
                    st.session_state.pdf_names.append(pdf.name)
                    all_texts.append(raw_text)
                    
                    st.success(f"Processed: {pdf.name}")
                except Exception as e:
                    st.error(f"Error processing {pdf.name}: {str(e)}")
        
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
    
    # Display currently loaded PDFs
    if st.session_state.pdf_names:
        st.subheader("Loaded PDFs")
        for pdf_name in st.session_state.pdf_names:
            st.write(f"- {pdf_name}")
        
        # Option to clear loaded PDFs
        if st.button("Clear PDFs"):
            st.session_state.pdf_texts = []
            st.session_state.pdf_names = []
            st.session_state.document_search = None
            st.rerun()
    
    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()