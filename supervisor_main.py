from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from rag_agnet_ganzneu import create_agent, setup_tools, load_existing_vectorstore
from web_such_agent import research_agent, ask_question_and_save_answer
from qa_ethics_agent import qa_ethics_agent
from data_analysis_agent import agent as data_analysis_agent
import os
from dotenv import load_dotenv
import re
import time

 === Loading environment variables (e.g., API keys) ===
load_dotenv()

 === Initialization of the language model (Google Gemini Flash) ===
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

 === Preparing the RAG agent (Document-based QA) ===
vectorstore = load_existing_vectorstore()   Loading the existing vector database (e.g., financial reports)
tools = setup_tools(vectorstore)   Setting up the tools for the RAG agent
rag_agent = create_agent(tools)   Creating the ReAct agent with the tools
rag_agent.name = "rag_agent"
research_agent.name = "research_agent"   Naming the web search agent
data_analysis_agent.name = "data_analysis_agent"   Naming the data analysis agent

 === Creating and configuring the supervisor ===
supervisor = create_supervisor(
    model=llm,
    agents=[rag_agent, research_agent, data_analysis_agent],
    prompt=(
        "You are a supervisor managing three agents:\n"
        "- 'rag_agent': Handles document-based and structured data questions.\n"
        "- 'research_agent': Handles real-time web search questions.\n"
        "- 'data_analysis_agent': Handles data analysis, statistics, CSV/Excel, plotting, and advanced comparisons.\n"
        "Always assign one task to one agent. Never do the work yourself.\n"
        "If the user asks for analysis, statistics, plotting, or comparison of data, use the data_analysis_agent.\n"
        "After each agent responds, hand back the results."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

 === Smalltalk detection (friendly greetings, etc.) ===
smalltalk_keywords = [
    "hello", "hi", "how are", "good morning", "good evening", "servus",
    "greetings", "moin", "hey", "what's up", "how's it going", "all right",
    "what are you doing", "who are you", "what can you do"
]

def is_smalltalk(question: str) -> bool:
    return any(kw in question.lower() for kw in smalltalk_keywords)

 === Answer validation: insufficient, empty, no numbers, etc. ===
def is_insufficient(answer: str, user_input: str = "") -> bool:
    if not answer or not isinstance(answer, str):
        return True
    if any(phrase in answer.lower() for phrase in [
        "no data", "not available", "unknown", "i don't know"
    ]):
        return True
    if any(kw in user_input.lower() for kw in ["how much", "revenue", "profit", "current", "numbers", "amount", "revenue"]):
        if not re.search(r"\d{4}|\d+[\.,]?\d*", answer):
            return True
    return len(answer.strip()) < 10

 === Year checking: add a note if year < current year ===
def adjust_temporal_phrasing(user_input: str) -> str:
    from datetime import datetime
    current_year = datetime.now().year
    import re
    match = re.search(r"(?:revenue|profit|cash)[^0-9]*(\d{4})", user_input.lower())
    if match:
        year = int(match.group(1))
        if year < current_year:
            return f"{user_input} (Note: We are in the year {current_year}, the figures for {year} should be published.)"
    return user_input

 === Logging: save question, answer, source, timestamp ===
def log_to_file(user_input, answer, source):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insuff_flag = "❗" if is_insufficient(answer, user_input) else "✅"
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n⏰ {timestamp}\n{insuff_flag} Question: {user_input}\nAnswer: {answer}\nSource: {source}\n")
        f.write("-" * 60 + "\n")

 === Function to check if the question contains a recent year ===
def contains_recent_year(user_input: str, min_year: int = 2024) -> bool:
    years = re.findall(r"\b(20\d{2})\b", user_input)
    return any(int(y) >= min_year for y in years)

 === Only when the file is run directly (not on import) ===
if __name__ == "__main__":
    print("\nSupervisor is ready. Enter a question (or 'exit' to quit):")
    history = []   History for RAG context

    while True:
        user_input = input("\nQuestion: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        if is_smalltalk(user_input):
            general_chat_tool = tools[0]
            answer_text = general_chat_tool.run(user_input)
            source = "RAG-Agent (general_chat)"
        elif contains_recent_year(user_input, 2024):
            print("\n[Note] Question contains year 2024 or later → Using Web Agent...")
            answer_text, source = ask_question_and_save_answer(user_input)
        else:
            try:
                rag_result = rag_agent.invoke({"input": user_input, "history": history})
                answer_text = rag_result.get("output") if isinstance(rag_result, dict) else str(rag_result)
                source = "RAG-Agent"

                 NEW: If answer is empty, None, or too short → Use Web Agent
                if not answer_text or not isinstance(answer_text, str) or len(answer_text.strip()) < 5:
                    print("\n[Note] RAG-Agent provided no answer → Using Web Agent...")
                    answer_text, source = ask_question_and_save_answer(user_input)
                elif is_insufficient(answer_text, user_input):
                    print("\n[Note] RAG answer incomplete → Using Web Agent...")
                    answer_text, source = ask_question_and_save_answer(user_input)

            except Exception as e:
                print("\n[Error] RAG-Agent failed → Using Web Agent...")
                answer_text, source = ask_question_and_save_answer(user_input)

        print("\nAnswer:")
        print(answer_text)
        print(f"Source: {source}")

         Check if relevant numbers are present in the answer
        number_keywords = ["how much", "revenue", "profit", "current", "numbers", "amount", "revenue"]
        if any(kw in user_input.lower() for kw in number_keywords):
            if not re.search(r"\d{4}|\d+[\.,]?\d*", answer_text):
                print("\n⚠️ No current revenue figures could be found. Please check the official financial reports or the investor relations page of the company.")

         QA/Ethics check of the answer
        warnings = qa_ethics_agent.run(answer_text, [source])
        print("\n⚖️ QA/Ethics Check:")
        print(warnings)

         Update history for the next iteration
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer_text})

 === Export for Gradio or external use ===
__all__ = [
    "rag_agent", "tools", "ask_question_and_save_answer",
    "qa_ethics_agent", "is_smalltalk", "is_insufficient",
    "adjust_temporal_phrasing", "log_to_file"
]

