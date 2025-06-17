import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults

# --- API Key Setup ---
# Set your API keys as environment variables for security
# os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY"

# Optional: To hide LangSmith warnings if you're not using it.
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# --- Agent Setup (Unchanged) ---

# 1. Initialize the LLM (using the more capable model)
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# 2. Define the Tools
tools = [TavilySearchResults(max_results=3)]

# 3. Create the Agent Prompt
prompt = hub.pull("hwchase17/react")

# 4. Create the Agent
agent = create_react_agent(llm, tools, prompt)

# 5. Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# --- NEW: The Guardrail Function ---
def run_nitw_agent(question: str):
    """
    Runs the agent only if the question is about NIT Warangal.
    """
    print(f"--- Processing Question: '{question}' ---")

    # Check for keywords in a case-insensitive way
    keywords = ["nitw", "nit warangal"]
    if any(keyword in question.lower() for keyword in keywords):
        print("Question is on-topic. Running agent...")
        response = agent_executor.invoke({"input": question})
        print("\n--- Final Answer ---")
        print(response["output"])
    else:
        # If the question is off-topic, refuse to answer.
        print("\n--- Final Answer ---")
        print("I'm sorry, I am a specialized agent and can only answer questions about NIT Warangal.")
    
    print("-" * 30)


# # --- Run with Examples ---

# # Example 1: On-topic question (will run the agent)
# on_topic_question = "Who is the Head of the Department for CSE at NITw?"
# run_nitw_agent(on_topic_question)


# # Example 2: Off-topic question (will be rejected by the guardrail)
# off_topic_question = "What is the capital of France?"
# run_nitw_agent(off_topic_question)
