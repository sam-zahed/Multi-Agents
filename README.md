📊 Multimodal AI Agent System for Financial Analysis
An intelligent analysis framework designed to process, analyze, and enrich multimodal corporate data from Investor Relations (IR) documents in real-time—delivered through an interactive Gradio web interface.
________________________________________
🧠 Agent Architecture
The system is built on a multi-agent orchestration pattern, managed by a central Supervisor module:
🔍 RAG Agent (Document QA)
•	Function: Answers content-specific financial questions based on PDF documents (IR reports).
•	Engine: Utilizes a ChromaDB vector database with SentenceTransformers.
•	LLM: Google Gemini.
•	Tools: document_search, general_chat.
📈 Data Analysis Agent
•	Function: Performs statistical analysis and creates visualizations (e.g., profit trends, time series).
•	Intelligence: Automatically detects and executes analysis tasks.
•	Tools: Pandas, Matplotlib, Seaborn, smol-ai Agent.
🌐 Web Search Agent
•	Function: Conducts real-time market research.
•	Engine: Powered by Tavily Search.
•	Output: Returns sources and summarized content, saving results to a dedicated log file.
🧭 Coordination Supervisor
•	Logic: Uses the LangGraph Supervisor module.
•	Workflow: Routes user queries to the most relevant agent (RAG, Analysis, or Web) and consolidates results while maintaining conversation history.
✅ QA & Ethics Agent
•	Function: Reviews final responses for:
o	Incompleteness.
o	Missing citations/sources.
o	Potential algorithmic bias.
•	Feedback: Provides immediate visual quality feedback in the UI.
________________________________________
🔄 Data Pipeline
1.	data_extraction.py: Extracts raw text and tables from IR PDF files.
2.	data_chunking.py: Splits content into semantic chunks and embeds them into ChromaDB.
3.	rag_agent_new.py: Initializes the vector store and specialized tools.
4.	supervisor_main.py: Orchestrates the agent communication and workflow logic.
5.	app.py: The entry point for the Gradio-based web user interface.
________________________________________
⚙️ Technology Stack
Component	Technology / Model
Vector Database	ChromaDB + SentenceTransformers
LLMs	Google Gemini 2.0 Flash
Analysis Agent	smol-ai CodeAgent (Llama 3)
Web Search	TavilySearch API
GUI	Gradio
________________________________________
🧪 Input Data & Objectives
•	Sources: Investor Relations documents (2020–2024) for Apple, Google, Meta, Microsoft, and NVIDIA.
•	Formats: PDF (Annual reports, quarterly presentations, and earnings transcripts).
•	Goal: Enable document-based QA, automated chart generation, financial forecasting, and trend analysis.
________________________________________

