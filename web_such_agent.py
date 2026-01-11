from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import tool

# .env-Datei laden
load_dotenv()

# Web-Suchtool als @tool definieren
@tool
def web_search_tool(query: str):
    """Führt eine Websuche durch und gibt Inhalt und Quelle (URL, falls vorhanden) zurück."""
    search = TavilySearch(max_results=3)
    web_search_results = search.invoke(query)
    if "results" in web_search_results and len(web_search_results["results"]) > 0:
        result = web_search_results["results"][0]
        content = result.get("content", "")
        source = result.get("source") or result.get("url") or "Quelle unbekannt"
        return content, source
    else:
        return "Keine Ergebnisse gefunden.", "Quelle unbekannt"

# Modell initialisieren
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Funktion zum Speichern von Antwort und Quelle
def store_answer_and_source(question, answer, source):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("answers_and_sources.txt", "a") as file:
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer: {answer}\n")
        file.write(f"Source: {source}\n")
        file.write("\n" + "-"*50 + "\n")

# Agent erstellen
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

# Beispiel: Abfrage stellen und Antwort mit Quelle speichern
def ask_question_and_save_answer(question):
    # Web-Suche ausführen
    answer, source = web_search_tool.invoke(question)
    
    # Die Antwort speichern
    store_answer_and_source(question, answer, source)

    # Die Antwort zurückgeben
    return answer, source

# Benutzerinteraktion: Mehrere Fragen zulassen
# while True:
#     question = input("Bitte geben Sie Ihre frage ein (oder 'exit' zum Beenden): ")
#     if question.strip().lower() == 'exit':
#         print("Beendet.")
#         break
#     answer, source = ask_question_and_save_answer(question)
#     print("Antwort:", answer)
#     print("Quelle:", source)
