# AI Chatbot with Document Q&A

A Streamlit application that allows users to upload PDF documents and ask questions about their content.

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

- Upload and process PDF documents
- Chat interface for asking questions
- Document similarity search using embeddings
- Response generation using LLM
