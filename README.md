
# 📊 Multimodales KI-Agentensystem für Finanzanalyse

Ein intelligentes Analyse-Framework, das multimodale Unternehmensdaten aus Investor-Relations-Dokumenten strukturiert verarbeitet, analysiert und in Echtzeit anreichert – bereitgestellt über eine Gradio-Weboberfläche.

---

## 🧠 Agentenarchitektur

Das System basiert auf spezialisierten KI-Agenten, die über ein Supervisor-Modul orchestriert werden:

### 🔍 RAG-Agent (Document QA)
- Verwendet Vektor-Datenbank (Chroma) mit SentenceTransformers.
- Antwortet auf inhaltliche Fragen zu Finanzdaten auf Basis von PDF-Dokumenten (IR-Berichte).
- LLM: Google Gemini
- Tools: `document_search`, `general_chat`

### 📈 Datenanalyse-Agent
- Führt Analysen, Statistiken und Visualisierungen durch (z. B. Gewinnverläufe, Zeitreihen).
- Erkennt automatisch Analyseaufträge.
- Tools: Pandas, Matplotlib, seaborn, smol-ai Agent

### 🌐 Websuche-Agent
- Führt aktuelle Marktsuchen aus (via Tavily).
- Gibt Quelle & Inhalt zurück.
- Speichert Ergebnisse in Logdatei.

### 🧭 Koordinations-Supervisor
- Nutzt LangGraph Supervisor-Modul.
- Zuweisung der Nutzeranfrage an passenden Agenten (RAG, Analyse, Web).
- Integration aller Ergebnisse inkl. Verlaufsspeicherung.

### ✅ QA & Ethik-Agent
- Prüft Antwort auf:
  - Unvollständigkeit
  - Fehlende Quellen
  - Möglichen Bias
- Gibt visuelles Feedback direkt aus.

---

## 🔄 Datenpipeline

1. `data_extrahieren.py`: Extrahiert Tabellen und Text aus IR-PDFs.
2. `data_chunkieren.py`: Teilt Inhalte in semantische Chunks und speichert in Chroma-DB.
3. `rag_agnet_ganzneu.py`: Lädt Vektorstore & Tools.
4. `supervisor_main.py`: Führt Agenten zusammen & regelt Workflows.
5. `app.py`: Gradio-basierte Benutzeroberfläche.

---

## ⚙️ Technologiestack

| Komponente              | Technologie/Modell                   |
|-------------------------|--------------------------------------|
| Vektor-Datenbank        | ChromaDB + SentenceTransformers      |
| LLMs                    | Google Gemini 2.0 Flash              |
| Analyse-Agent           | smol-ai `CodeAgent`, Llama 3         |
| Websuche                | TavilySearch                         |
| GUI                     | Gradio                               |

---

## 🧪 Eingabedaten

- **Quellen:** Investor Relations-Dokumente (2020–2024) von Apple, Google, Meta, Microsoft, NVIDIA
- **Formate:** PDF (Berichte, Präsentationen, Transkripte)
- **Ziel:** Dokumentbasierte QA, Diagrammerzeugung, Prognosen, Trendanalysen

---

