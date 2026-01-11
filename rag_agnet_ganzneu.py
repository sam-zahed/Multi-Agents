from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

# === 1. Bestehende Vektor-Datenbank laden (z. B. Chroma mit HuggingFace-Embeddings) ===
def load_existing_vectorstore():
    # Verwende HuggingFace-Embedding-Modell zur semantischen Repräsentation von Text
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # Lade die bestehende Chroma-Datenbank mit den Embeddings
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

# === 2. Tools vorbereiten: Dokumentensuche und allgemeiner Chat ===
def setup_tools(vectorstore: Optional[Chroma] = None):
    # Initialisiere das LLM (Google Gemini) für Tool-Logik
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    tools = []  # Liste aller verfügbaren Tools für den Agent

    if vectorstore:
        # Baue eine Retrieval-QA-Kette zur Nutzung der Vektor-Datenbank
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            verbose=True
        )

        # Wrapper-Funktion zur Ausgabe von QA-Ergebnissen mit Debug-Ausgabe
        def debug_qa_chain(query):
            result = qa_chain.run(query)
            print("[DEBUG] RetrievalQA result:", result)
            return result

        # Füge Tool für die Dokumentensuche hinzu
        tools.append(
            Tool(
                name="document_search",
                func=debug_qa_chain,
                description=(
                    "Suche nach konkreten Fakten, Zahlen oder Tabellen aus Finanzberichten. "
                    "Bevorzugt bei Umsatz-, Gewinn-, oder anderen dokumentenbasierten Fragen."
                )
            )
        )

    # Tool für allgemeine Fragen, die nicht dokumentenbasiert sind
    tools.append(
        Tool(
            name="general_chat",
            func=lambda q: llm.invoke(f"Antworte natürlich auf: {q}").content,
            description="Für Smalltalk oder Fragen ohne Dokumentenbezug."
        )
    )

    return tools

# === 3. ReAct-Agent definieren, der Tools intelligent nutzt und per Prompt gesteuert wird ===
def create_agent(tools: list):
    # Prompt mit Anweisungen, wann welches Tool verwendet werden soll
    prompt = ChatPromptTemplate.from_template("""
Du bist ein ReAct-Agent für Unternehmensdaten. Befolge folgende Regeln strikt:

1. Nutze IMMER zuerst das Tool **document_search**, wenn die Frage sich auf:
   - Umsatz, Gewinn, Einnahmen
   - Jahre (z. B. 2021, 2022, 2023)
   - Inhalte aus Berichten, Tabellen oder Dokumenten
   bezieht.

2. Wenn das Jahr in der Frage **< aktuelles Jahr liegt**, gehe davon aus, dass die Daten veröffentlicht sind.
   Lass dich nicht mit 'nicht verfügbar' zufrieden geben.

3. Falls document_search keine gute Antwort bringt oder keine Zahl enthält,
   leite zur Websuche weiter (Tool: research_agent), sofern vorhanden.

4. Nutze general_chat **nur** für Smalltalk oder allgemeine Fragen.

Format:
Verfügbare Tools: {tools}
Tool-Namen: {tool_names}
Verlauf: {history}
Frage: {input}

Nutze folgenden Ablauf:
Thought: ...
Action: <Tool-Name oder Final Answer>
Action Input: <Text>
Observation: <Tool-Ergebnis>
...
Thought: Ich habe genug Information.
Final Answer: <Antwort>

{agent_scratchpad}
""")

    # Nochmals das LLM initialisieren für den Agenten selbst
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    # Erstelle den ReAct-Agent basierend auf Prompt und Tools
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Verpacke den Agenten in einen AgentExecutor für Laufzeitsteuerung
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Ausgabe der Zwischenschritte
        handle_parsing_errors=True  # Toleranz bei Parsing-Problemen
    )
    executor.name = "rag_agent"
    return executor
