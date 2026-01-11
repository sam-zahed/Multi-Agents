
import gradio as gr
import os
import re
from pathlib import Path
from datetime import datetime

from supervisor_main import (
    rag_agent, tools, ask_question_and_save_answer,
    qa_ethics_agent, is_smalltalk, is_insufficient,
    adjust_temporal_phrasing, log_to_file
)
from data_analysis_agent import agent as data_analysis_agent


history = []

# === Funktion: Erkenne Analyseaufträge ===
def is_data_analysis_request(user_input: str) -> bool:
    chart_keywords = [
        "analysiere", "analyse", "plot", "diagramm", "visualisiere",
        "statistik", "vergleich", "vergleiche", "csv", "datenanalyse",
        "korrelation", "trend", "zeitreihe", "daten", "tabelle"
    ]
    finance_keywords = [
        "umsatz", "gewinn", "einnahmen", "ausgaben", "cash", "kapital",
        "verbindlichkeit", "kosten", "aktien", "bilanz", "umsätze"
    ]
    return any(a in user_input.lower() for a in chart_keywords) and any(f in user_input.lower() for f in finance_keywords)


# === Neu: Suche nach neuestem Diagramm im Ordner ===
def get_latest_figure():
    figures_path = Path("figures")
    if not figures_path.exists():
        return None
    figures = list(figures_path.glob("*.png"))
    return str(max(figures, key=os.path.getctime)) if figures else None


# === Hauptlogik ===
def chat_supervisor(message, chat_history):
    global history
    user_input = message.strip()
    adjusted_input = adjust_temporal_phrasing(user_input)
    history.append({"role": "user", "content": user_input})

    image_path = None

    if is_data_analysis_request(user_input):
        try:
            answer = data_analysis_agent.run(user_input)
            source = "Data-Analysis-Agent"
            image_path = get_latest_figure()
        except Exception as e:
            answer = f"Fehler beim Data-Analysis-Agent: {e}"
            source = "Data-Analysis-Agent"

    elif is_smalltalk(user_input):
        general_chat_tool = tools[0]
        answer = general_chat_tool.run(user_input)
        source = "RAG-Agent (general_chat)"

    else:
        try:
            result = rag_agent.invoke({"input": adjusted_input, "history": history})
            answer = result.get("output") if isinstance(result, dict) else str(result)
            source = "RAG-Agent"

            if is_insufficient(answer, adjusted_input):
                answer, source = ask_question_and_save_answer(user_input)

        except Exception:
            answer, source = ask_question_and_save_answer(user_input)

    history.append({"role": "assistant", "content": answer})
    qa = qa_ethics_agent.run(answer, [source])
    annotated = f"{answer}\n\n📚 Quelle: {source}\n⚖️ QA/Ethikprüfung: {qa}"
    log_to_file(user_input, answer, source)

    return (annotated, image_path)


# === Gradio UI ===
demo = gr.ChatInterface(
    fn=chat_supervisor,
    title="📊 Supervisor Multi-Agent Chat",
    description="Kombinierte KI mit RAG + Websuche + Statistik + QA. Stelle Fragen zu Umsatz, Firmen, Trends & mehr.",
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Frage z.B. 'Zeige Apple Umsatz als Diagramm'", label="Frage"),
    additional_outputs=[gr.Image(label="📈 Diagramm", visible=True)]
)


if __name__ == "__main__":
    demo.launch()

