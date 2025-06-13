# NIT Warangal Chatbot (NItChat Bot)

A Streamlit application that allows users to upload PDF or TXT documents related to NIT Warangal and ask questions about their content via a conversational chat interface.

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Groq API key to the `.env` file
4. Run the application:
   ```
   streamlit run index.py
   ```

## Required API Keys

- **Groq API Key**: Get your API key from [Groq Console](https://console.groq.com/)

## Features

- Upload and process PDF or TXT documents about NIT Warangal
- Chat interface for asking questions about uploaded documents or general queries
- Document similarity search using embeddings (FAISS)
- Response generation using LLM
- Option to clear chat history and uploaded files
