python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

# === 1. Load existing vector database (e.g., Chroma with HuggingFace Embeddings) ===
def load_existing_vectorstore():
    # Use HuggingFace embedding model for semantic text representation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # Load existing Chroma database with embeddings
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

# === 2. Prepare tools: document search and general chat ===
def setup_tools(vectorstore: Optional[Chroma] = None):
    # Initialize LLM (Google Gemini) for tool logic
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    tools = []  # List of all available tools for the agent

    if vectorstore:
        # Build a RetrievalQA chain to use the vector database
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            verbose=True
        )

        # Wrapper function to output QA results with debug output
        def debug_qa_chain(query):
            result = qa_chain.run(query)
            print("[DEBUG] RetrievalQA result:", result)
            return result

        # Add tool for document search
        tools.append(
            Tool(
                name="document_search",
                func=debug_qa_chain,
                description=(
                    "Search for specific facts, numbers, or tables from financial reports. "
                    "Preferred for revenue, profit, or other document-based questions."
                )
            )
        )

    # Tool for general questions not related to documents
    tools.append(
        Tool(
            name="general_chat",
            func=lambda q: llm.invoke(f"Answer naturally to: {q}").content,
            description="For small talk or questions without document reference."
        )
    )

    return tools

# === 3. Define ReAct agent that intelligently uses tools and is controlled by prompt ===
def create_agent(tools: list):
    # Prompt with instructions on when to use which tool
    prompt = ChatPromptTemplate.from_template("""
You are a ReAct agent for corporate data. Strictly follow these rules:

1. ALWAYS use the **document_search** tool first if the question relates to:
   - Revenue, profit, earnings
   - Years (e.g., 2021, 2022, 2023)
   - Content from reports, tables, or documents.

2. If the year in the question **is < current year**, assume the data has been published.
   Do not settle for 'not available'.

3. If document_search does not yield a good answer or contains no numbers,
   forward to web search (tool: research_agent), if available.

4. Use general_chat **only** for small talk or general questions.

Format:
Available tools: {tools}
Tool names: {tool_names}
History: {history}
Question: {input}

Use the following flow:
Thought: ...
Action: <Tool name or Final Answer>
Action Input: <Text>
Observation: <Tool result>
...
Thought: I have enough information.
Final Answer: <Answer>

{agent_scratchpad}
""")

    # Initialize LLM again for the agent itself
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    # Create the ReAct agent based on prompt and tools
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Wrap the agent in an AgentExecutor for runtime control
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Output intermediate steps
        handle_parsing_errors=True  # Tolerance for parsing problems
    )
    executor.name = "rag_agent"
    return executor
