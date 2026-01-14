python
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import tool

# Load .env file
load_dotenv()

# Define web search tool as @tool
@tool
def web_search_tool(query: str):
    """Performs a web search and returns content and source (URL, if available)."""
    search = TavilySearch(max_results=3)
    web_search_results = search.invoke(query)
    if "results" in web_search_results and len(web_search_results["results"]) > 0:
        result = web_search_results["results"][0]
        content = result.get("content", "")
        source = result.get("source") or result.get("url") or "Source unknown"
        return content, source
    else:
        return "No results found.", "Source unknown"

# Initialize model
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Function to save answer and source
def store_answer_and_source(question, answer, source):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("answers_and_sources.txt", "a", encoding="utf-8") as file:
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer: {answer}\n")
        file.write(f"Source: {source}\n")
        file.write("\n" + "-"*50 + "\n")

# Create agent
research_agent = create_react_agent(
    model=llm,
    tools=[web_search_tool],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text.\n"
        "- After you find the answer, store the answer along with the source."
    ),
)
research_agent.name = "research_agent"

# Example: Ask question and save answer with source
def ask_question_and_save_answer(question):
    # Execute web search
    answer, source = web_search_tool.invoke(question)
    
    # Save the answer
    store_answer_and_source(question, answer, source)

    # Return the answer
    return answer, source

# User interaction: Allow multiple questions
# while True:
#     question = input("Please enter your question (or 'exit' to quit): ")
#     if question.strip().lower() == 'exit':
#         print("Terminated.")
#         break
#     answer, source = ask_question_and_save_answer(question)
#     print("Answer:", answer)
#     print("Source:", source)
