from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import dotenv
import os
dotenv.load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
from langchain.chains.question_answering import load_qa_chain as original_load_qa_chain
from groq import Groq
from langchain_groq import ChatGroq
llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.3-70b-versatile",  # You can choose different models
    temperature=0,
    max_tokens=1024
)

# Export the load_qa_chain function to be used in index.py
def load_qa_chain(llm_model):
    return original_load_qa_chain(llm_model)
