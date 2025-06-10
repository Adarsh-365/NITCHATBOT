import os
from dotenv import load_dotenv
# loading variables from .env file
load_dotenv()
from groq import Groq
import sys

# Check if API key exists
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY environment variable not found. Make sure it's in your .env file.")
    sys.exit(1)

try:
    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    print(chat_completion.choices[0].message.content)
except Exception as e:
    print(f"Error occurred: {e}")