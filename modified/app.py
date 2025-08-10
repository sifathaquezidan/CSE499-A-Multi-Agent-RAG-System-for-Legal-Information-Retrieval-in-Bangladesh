import streamlit as st
import os
import fitz  # PyMuPDF
from typing import List, Optional, TypedDict as TypingTypedDict
from typing_extensions import TypedDict 
from operator import itemgetter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum
import functools
import traceback
import sys
import math 

load_dotenv()

# LangChain Imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder 
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document, HumanMessage, AIMessage 
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph Imports
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver # For potential state saving if needed later

# New LLM Provider System
from llm_providers import LLMConfig, LLMManager, LLMProvider
from config_manager import (
    AppConfig, create_configuration_ui, get_legacy_config, 
    create_legacy_compatibility_wrapper, show_configuration_status,
    show_provider_help
)


# --- Constants ---
DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "Legal_data"
RAG_GRAPH_IMAGE_PATH = "rag_agent_graph.png"
CLARIFICATION_GRAPH_IMAGE_PATH = "clarification_agent_graph.png"
EMBEDDING_BATCH_SIZE = 200
MAX_DOCS_PER_QUERY = 5 
MINIMUN_RETRIVAL_SCORE = 0.1
MAX_QUERY_CLARIFICATION_TURNS = 5

# --- PDF Processing Functions ---
# ... (Keep the existing extract_text_from_pdf and process_pdfs_in_directory functions) ...
def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {e}")
        return None

def process_pdfs_in_directory(directory_path):
    """Processes all PDFs in a directory, extracts text, and splits into documents."""
    all_docs = []
    if not os.path.exists(directory_path):
        st.warning(f"Directory not found: {directory_path}")
        return []

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.warning(f"No PDF files found in {directory_path}")
        return []

    # Check if running in Streamlit context before using st elements
    streamlit_running = 'streamlit' in sys.modules

    if streamlit_running:
        progress_bar = st.progress(0, text="Processing PDFs...")
        total_files = len(pdf_files)
    else:
        total_files = len(pdf_files)
        print(f"Processing {total_files} PDF files...")


    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(directory_path, filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            docs = text_splitter.create_documents([pdf_text], metadatas=[{"source": filename}])
            all_docs.extend(docs)

        if streamlit_running:
            progress_bar.progress((i + 1) / total_files, text=f"Processing: {filename} ({i+1}/{total_files})")
        else:
            print(f"Processed: {filename} ({i+1}/{total_files})")

    if streamlit_running:
        progress_bar.empty()
        if not all_docs:
            st.warning("No text could be extracted from the PDFs.")
        else:
            st.info(f"Processed {total_files} PDF files, extracted {len(all_docs)} document chunks.")
    else:
        print(f"Processed {total_files} PDF files, extracted {len(all_docs)} document chunks.")
        if not all_docs:
            print("Warning: No text could be extracted from the PDFs.")

    return all_docs


# --- Caching Functions for Streamlit (Updated for multi-provider support) ---
@st.cache_resource
def get_embedding_model_legacy(openai_api_key):
    """Legacy function for backward compatibility"""
    os.environ['OPENAI_API_KEY'] = openai_api_key
    try:
        st.write("Initializing Embeddings Model...")
        model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
        st.write("Embeddings Model Initialized.")
        return model
    except Exception as e:
        st.error(f"Failed to initialize OpenAI Embeddings: {e}")
        return None

@st.cache_resource(ttl="1h")
def load_or_create_vector_db(_embeddings_model):
    if not _embeddings_model:
        st.error("Embeddings model not available. Cannot load/create Vector DB.")
        return None

    # Check if DB exists and seems valid
    db_exists = os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR)
    vector_db = None

    if db_exists:
        st.info(f"Attempting to load existing vector database from {PERSIST_DIR}...")
        try:
            vector_db = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR,
                embedding_function=_embeddings_model
            )
            # Try a simple operation to confirm it's loaded correctly
            count = vector_db._collection.count()
            st.success(f"Vector database loaded successfully with {count} documents.")
            return vector_db
        except Exception as e:
            st.error(f"Error loading existing database: {e}. Will try to recreate.")
            st.warning("Manual deletion of the './chroma_db' directory might be required if recreation fails.")
            vector_db = None # Ensure it's None so we proceed to creation

    # If DB doesn't exist or loading failed, create it
    st.info(f"Creating new vector database in {PERSIST_DIR}...")
    if not os.path.exists(DATA_DIR):
         os.makedirs(DATA_DIR)
         st.warning(f"Created data directory: {DATA_DIR}. Please add PDF files there and rerun.")
         return None

    docs = process_pdfs_in_directory(DATA_DIR) # This function now uses Streamlit context check
    if not docs:
        st.error("No documents processed from PDF files. Cannot create vector database.")
        return None

    total_docs = len(docs)
    st.info(f"Embedding {total_docs} document chunks in batches...")

    # Initialize Chroma collection first, without documents
    try:
        vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings_model,
            persist_directory=PERSIST_DIR
        )
    except Exception as e:
        st.error(f"Failed to initialize empty Chroma collection: {e}")
        return None

    # Add documents in batches
    num_batches = math.ceil(total_docs / EMBEDDING_BATCH_SIZE)
    progress_bar_embed = st.progress(0, text=f"Embedding batch 1 of {num_batches}...")

    try:
        for i in range(0, total_docs, EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_docs = docs[i : i + EMBEDDING_BATCH_SIZE]

            if not batch_docs: # Should not happen with the loop logic, but safety check
                continue

            # Update progress text
            progress_text = f"Embedding batch {batch_num} of {num_batches} ({len(batch_docs)} docs)..."
            progress_value = (i + len(batch_docs)) / total_docs
            progress_bar_embed.progress(progress_value, text=progress_text)

            # Add the batch to the collection
            vector_db.add_documents(documents=batch_docs)
            st.write(f"Batch {batch_num}/{num_batches} embedded.") # Optional: more verbose logging

        # Ensure persistence after adding all batches
        # vector_db.persist() # Chroma with persist_directory usually handles this, but explicit call can ensure it. Test if needed.
        progress_bar_embed.progress(1.0, text="Embedding complete. Persisting...")
        progress_bar_embed.empty() # Remove progress bar
        st.success(f"New vector database created and persisted with {vector_db._collection.count()} documents.")
        return vector_db

    except Exception as e:
        progress_bar_embed.empty() # Remove progress bar on error
        st.error(f"Failed to create vector database during batch embedding: {e}")
        st.error(f"Error details: {traceback.format_exc()}") # More detailed error
        # Consider cleanup: os.rmdir or similar if creation failed midway? Risky.
        st.warning(f"Database creation failed. You might need to manually delete the '{PERSIST_DIR}' directory before retrying.")
        return None

# ==============================================================================
# --- NEW: Conversational Clarification Agent ---
# ==============================================================================

# --- State for Clarification Graph ---
class ClarificationState(TypedDict):
    initial_query: str              # The user's very first query
    conversation_history: List     # List of HumanMessage/AIMessage for the clarification phase
    max_turns: int                  # Max clarification attempts
    current_turn: int               # Current attempt number
    clarified_query: Optional[str]  # The final synthesized query if successful
    ask_user_question: Optional[str] # The question to ask the user next (if any)
    reasoning_for_question: Optional[str]
# --- Pydantic Model for Clarity Assessment in Conversation ---
class ClarityStatus(str, Enum):
    CLEAR = "CLEAR"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    MAX_TURNS_REACHED = "MAX_TURNS_REACHED" # Added state

class ConversationalClarityAssessment(BaseModel):
    """Assess if the conversation provides enough detail for a Bangladeshi Legal Assistant."""
    status: ClarityStatus = Field(description="Enum: 'CLEAR' if sufficient details are present, 'NEEDS_CLARIFICATION' if more info needed, 'MAX_TURNS_REACHED' if stuck.")
    reasoning: str = Field(description="Brief reasoning for the status. If NEEDS_CLARIFICATION, explain what's missing. If CLEAR, confirm understanding. If MAX_TURNS_REACHED, explain why stuck.")
    synthesized_query_if_clear: Optional[str] = Field(description="If status is CLEAR, provide a concise, synthesized query representing the user's final need.", default=None)

# --- Cache Clarification LLMs (Updated for multi-provider support) ---
@st.cache_resource
def get_clarification_components_legacy(openai_api_key):
    """Legacy function for clarification components"""
    os.environ['OPENAI_API_KEY'] = openai_api_key
    try:
        st.write("Initializing Clarification Agent LLMs...")
        assessment_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key).with_structured_output(ConversationalClarityAssessment)
        question_gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
        st.write("Clarification Agent LLMs Initialized.")
        return assessment_llm, question_gen_llm
    except Exception as e:
        st.error(f"Failed to initialize Clarification LLMs: {e}")
        return None, None

@st.cache_resource
def get_clarification_components_with_provider(llm_manager_dict):
    """Get clarification components using LLM provider system"""
    try:
        # Recreate LLM manager from dict (since it's not directly cacheable)
        from llm_providers import LLMConfig, LLMManager
        config = LLMConfig(**llm_manager_dict)
        llm_manager = LLMManager(config)
        
        st.write("Initializing Clarification Agent LLMs...")
        
        # Get base chat model
        base_model = llm_manager.get_chat_model()
        
        # Try to get structured output model
        try:
            assessment_llm = llm_manager.get_structured_output_model(ConversationalClarityAssessment)
        except Exception as e:
            st.warning(f"Structured output not supported by {config.provider.value}. Using base model with output parser.")
            assessment_llm = base_model
        
        # Question generation model with slightly higher temperature
        question_config = config.copy()
        question_config.temperature = 0.3
        question_manager = LLMManager(question_config)
        question_gen_llm = question_manager.get_chat_model()
        
        st.write("Clarification Agent LLMs Initialized.")
        return assessment_llm, question_gen_llm
        
    except Exception as e:
        st.error(f"Failed to initialize Clarification LLMs: {e}")
        return None, None

# --- Clarification Graph Nodes (Ensure signatures match partial binding) ---
# Note: The node functions themselves should still accept the LLM argument
def assess_clarity_node(state: ClarificationState, assessment_llm):
    """Assesses if the current conversation history provides enough clarity."""
    print("--- CLARIFICATION NODE: Assess Clarity ---")
    # ... (rest of the node function as corrected before) ...
    history = state['conversation_history']
    current_turn = state['current_turn']
    max_turns = state['max_turns']

    if current_turn >= max_turns:
        print("--- Max clarification turns reached. ---")
        return {
            "ask_user_question": None,
            "clarified_query": f"Could not fully clarify after {max_turns} attempts. Proceeding with best guess based on: {state['initial_query']}",
            "reasoning_for_question": None
        }

    ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are evaluating an ongoing conversation with a user asking about Bangladeshi Law.
            The user started with an initial query, and you may have asked clarifying questions.
            Your goal is to determine if the *entire conversation history* now provides enough specific detail (laws, facts, context, desired outcome related to Bangladesh) to be answered effectively by a legal RAG system.

            Analyze the conversation provided in the 'history'.

            Output your assessment using the 'ConversationalClarityAssessment' tool.
            - If CLEAR: Synthesize the user's core need into a concise 'synthesized_query_if_clear'.
            - If NEEDS_CLARIFICATION: Explain *specifically* what information is still missing in the 'reasoning'.
            - If MAX_TURNS_REACHED (use this only if explicitly told max turns reached by the system state): Explain why you couldn't get clarity.

            Initial User Query: {initial_query}
            Conversation History:
            """),
        MessagesPlaceholder(variable_name="history"),
        ("system", "Based *only* on the conversation above, assess the clarity and determine the next step.")
    ])

    assessment_chain = ASSESSMENT_PROMPT | assessment_llm
    try:
        assessment_result = assessment_chain.invoke({
            "initial_query": state['initial_query'],
            "history": history
        })
    except Exception as e:
        print(f"--- ERROR during assessment LLM call: {e} ---")
        # Handle error gracefully, maybe ask generic question or end
        return {
            "ask_user_question": "Sorry, I encountered an error trying to assess the conversation. Could you please restate your query?",
            "clarified_query": None,
            "reasoning_for_question": None
        }


    print(f"Clarity Assessment: {assessment_result.status}, Reason: {assessment_result.reasoning}")

    if assessment_result.status == ClarityStatus.CLEAR:
        return {
            "clarified_query": assessment_result.synthesized_query_if_clear,
            "ask_user_question": None,
            "reasoning_for_question": None
        }
    elif assessment_result.status == ClarityStatus.NEEDS_CLARIFICATION:
        return {
            "reasoning_for_question": assessment_result.reasoning,
            "ask_user_question": None,
            "clarified_query": None
        }
    else: # MAX_TURNS_REACHED handled by LLM or other unexpected status
            return {
                "ask_user_question": None,
                "clarified_query": f"Could not fully clarify. Reason: {assessment_result.reasoning}. Proceeding with best guess based on: {state['initial_query']}",
                "reasoning_for_question": None
            }


def generate_question_node(state: ClarificationState, question_gen_llm):
    """Generates the next clarifying question based on the assessment."""
    print("--- CLARIFICATION NODE: Generate Question ---")
    # Access reasoning safely using .get()
    reasoning = state.get("reasoning_for_question")
    if not reasoning:
         print("Error/Warning: Reasoning for question missing in state for generate_question_node.")
         # Ask a generic question or fallback
         return {
             "ask_user_question": "Sorry, I got a bit lost. Could you please provide more details about what you'd like to know?",
             "current_turn": state.get("current_turn", 0) + 1
             }

    history = state['conversation_history']

    QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
       # ... (prompt remains the same) ...
        ("system", """You are part of a conversational assistant helping a user clarify their legal question about Bangladesh.
        A previous analysis determined the conversation is not yet clear enough. The reason given was:
        '{reasoning}'

        Based on this reason and the conversation history so far, generate a *single, concise, specific follow-up question* to ask the user.
        Focus *only* on eliciting the missing information identified in the reasoning. Do NOT try to answer the original query.

        Conversation History:
        """),
        MessagesPlaceholder(variable_name="history"),
        ("system", "Generate the next clarifying question based on the history and the reasoning provided above.")
    ])

    question_chain = QUESTION_GEN_PROMPT | question_gen_llm | StrOutputParser()
    try:
        next_question = question_chain.invoke({
            "reasoning": reasoning,
            "history": history
        })
    except Exception as e:
        print(f"--- ERROR during question generation LLM call: {e} ---")
        # Handle error gracefully
        next_question = "Sorry, I encountered an error trying to formulate a follow-up question. Could you perhaps rephrase or add more detail?"


    print(f"Generated Question: {next_question}")
    return {
        "ask_user_question": next_question,
        "current_turn": state.get("current_turn", 0) + 1
    }


# --- Build Clarification Graph (Updated for multi-provider support) ---
@st.cache_resource
def get_clarification_agent_legacy(openai_api_key):
    """Legacy function for clarification agent"""
    assessment_llm, question_gen_llm = get_clarification_components_legacy(openai_api_key)
    if not assessment_llm or not question_gen_llm:
        st.error("Cannot build clarification agent: LLMs not available.")
        return None

    st.write("Building Clarification Agent Graph...")
    clarif_graph = StateGraph(ClarificationState)

    # Use functools.partial for cleaner binding
    bound_assess_clarity = functools.partial(assess_clarity_node, assessment_llm=assessment_llm)
    bound_generate_question = functools.partial(generate_question_node, question_gen_llm=question_gen_llm)

    clarif_graph.add_node("assess_clarity", bound_assess_clarity)
    clarif_graph.add_node("generate_question", bound_generate_question)
    clarif_graph.add_node("ask_user", lambda state: {"ask_user_question": state.get("ask_user_question")})

    clarif_graph.set_entry_point("assess_clarity")

    clarif_graph.add_conditional_edges(
        "assess_clarity",
        lambda state: "generate_question" if state.get("reasoning_for_question") else END,
        {
            "generate_question": "generate_question",
             END: END
        }
    )

    clarif_graph.add_edge("generate_question", "ask_user")

    try:
        st.write("Compiling Clarification Agent Graph...")
        clarification_agent = clarif_graph.compile()
        st.success("Clarification agent graph compiled successfully.")
        
        # Try to save graph image
        try:
            st.write(f"Attempting to save clarification graph image to {CLARIFICATION_GRAPH_IMAGE_PATH}...")
            if hasattr(clarification_agent, 'get_graph') and hasattr(clarification_agent.get_graph(), 'draw_mermaid_png'):
                image_bytes = clarification_agent.get_graph().draw_mermaid_png()
                with open(CLARIFICATION_GRAPH_IMAGE_PATH, "wb") as f:
                    f.write(image_bytes)
                st.info(f"Clarification agent graph visualization saved as {CLARIFICATION_GRAPH_IMAGE_PATH}")
            else:
                 st.warning("Could not generate clarification graph image: Method not found or dependencies missing.")
        except Exception as img_e:
            st.warning(f"Could not save clarification agent graph image: {img_e}")
        
        return clarification_agent
    except Exception as e:
        st.error(f"Failed to compile clarification agent graph: {e}")
        traceback.print_exc()
        return None

@st.cache_resource
def get_clarification_agent_with_provider(llm_manager_dict):
    """Build clarification agent using LLM provider system"""
    assessment_llm, question_gen_llm = get_clarification_components_with_provider(llm_manager_dict)
    if not assessment_llm or not question_gen_llm:
        st.error("Cannot build clarification agent: LLMs not available.")
        return None

    st.write("Building Clarification Agent Graph...")
    clarif_graph = StateGraph(ClarificationState)

    # Use functools.partial for cleaner binding
    bound_assess_clarity = functools.partial(assess_clarity_node, assessment_llm=assessment_llm)
    bound_generate_question = functools.partial(generate_question_node, question_gen_llm=question_gen_llm)

    clarif_graph.add_node("assess_clarity", bound_assess_clarity)
    clarif_graph.add_node("generate_question", bound_generate_question)
    clarif_graph.add_node("ask_user", lambda state: {"ask_user_question": state.get("ask_user_question")})

    clarif_graph.set_entry_point("assess_clarity")

    clarif_graph.add_conditional_edges(
        "assess_clarity",
        lambda state: "generate_question" if state.get("reasoning_for_question") else END,
        {
            "generate_question": "generate_question",
             END: END
        }
    )

    clarif_graph.add_edge("generate_question", "ask_user")

    try:
        st.write("Compiling Clarification Agent Graph...")
        clarification_agent = clarif_graph.compile()
        st.success("Clarification agent graph compiled successfully.")
        
        # Try to save graph image
        try:
            st.write(f"Attempting to save clarification graph image to {CLARIFICATION_GRAPH_IMAGE_PATH}...")
            if hasattr(clarification_agent, 'get_graph') and hasattr(clarification_agent.get_graph(), 'draw_mermaid_png'):
                image_bytes = clarification_agent.get_graph().draw_mermaid_png()
                with open(CLARIFICATION_GRAPH_IMAGE_PATH, "wb") as f:
                    f.write(image_bytes)
                st.info(f"Clarification agent graph visualization saved as {CLARIFICATION_GRAPH_IMAGE_PATH}")
            else:
                 st.warning("Could not generate clarification graph image: Method not found or dependencies missing.")
        except Exception as img_e:
            st.warning(f"Could not save clarification agent graph image: {img_e}")
        
        return clarification_agent
    except Exception as e:
        st.error(f"Failed to compile clarification agent graph: {e}")
        traceback.print_exc()
        return None

# ==============================================================================
# --- Agentic RAG Components (Main RAG Graph) ---
# ==============================================================================

# --- Pydantic model for grader ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# --- Cache the agent graph compilation (Updated for multi-provider support) ---
@st.cache_resource(ttl="1h")
def get_agentic_rag_app_legacy(_retriever, openai_api_key, tavily_api_key):
    """Legacy RAG app for backward compatibility"""
    if not _retriever:
       st.error("Retriever not available. Cannot build RAG agent.")
       return None
    if not openai_api_key or not tavily_api_key:
        st.error("API keys not set. Cannot build RAG agent.")
        return None

    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['TAVILY_API_KEY'] = tavily_api_key

    try:
        st.write("Initializing RAG Agent LLMs and Tools...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
        tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000, tavily_api_key=tavily_api_key)
        st.write("RAG Agent LLMs and Tools Initialized.")
    except Exception as e:
        st.error(f"Failed to initialize RAG Agent LLMs or Tools: {e}")
        return None

    return _build_rag_graph(_retriever, llm, structured_llm_grader, rewriter_llm, tv_search)

@st.cache_resource(ttl="1h")
def get_agentic_rag_app_with_provider(_retriever, llm_manager_dict, tavily_api_key):
    """RAG app using LLM provider system"""
    if not _retriever:
       st.error("Retriever not available. Cannot build RAG agent.")
       return None
    if not llm_manager_dict or not tavily_api_key:
        st.error("LLM manager or Tavily API key not provided. Cannot build RAG agent.")
        return None

    os.environ['TAVILY_API_KEY'] = tavily_api_key

    try:
        # Recreate LLM manager from dict
        from llm_providers import LLMConfig, LLMManager
        config = LLMConfig(**llm_manager_dict)
        llm_manager = LLMManager(config)
        
        st.write("Initializing RAG Agent LLMs and Tools...")
        
        # Get base chat model
        llm = llm_manager.get_chat_model()
        
        # Try to get structured output model for grading
        try:
            structured_llm_grader = llm_manager.get_structured_output_model(GradeDocuments)
        except Exception as e:
            st.warning(f"Structured output not fully supported by {config.provider.value}. Using base model.")
            structured_llm_grader = llm
        
        # Get rewriter model (same as base for now)
        rewriter_llm = llm_manager.get_chat_model()
        
        # Initialize Tavily search
        tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000, tavily_api_key=tavily_api_key)
        
        st.write("RAG Agent LLMs and Tools Initialized.")
        
        return _build_rag_graph(_retriever, llm, structured_llm_grader, rewriter_llm, tv_search)
        
    except Exception as e:
        st.error(f"Failed to initialize RAG Agent LLMs or Tools: {e}")
        return None

def _build_rag_graph(_retriever, llm, structured_llm_grader, rewriter_llm, tv_search):
    """Build the RAG graph with provided components"""
    # --- Prompts ---
    # Grader Prompt
    SYS_PROMPT_GRADER = """You are an expert grader assessing relevance of a retrieved document to a user question.
                 Follow these instructions for grading:
                   - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                   - Your grade should be strictly 'yes' or 'no' to indicate whether the document is relevant to the question or not."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT_GRADER),
        ("human", "Retrieved document:\n{document}\nUser question:\n{question}"),
    ])

    # RAG Prompt
    SYS_PROMPT_RAG = """You are an assistant for question-answering Bangladeshi Legal tasks.
             Use the following pieces of retrieved context to answer the question.
             If no context is present or if you don't know the answer based on the provided documents, state that clearly.
             Do not make up an answer.
             Provide a detailed and specific answer based *only* on the context provided.
             If context includes filenames, cite them like this: (Source: filename.pdf). If the context is from a web search, cite it as (Source: Web Search).
             Question:
             {question}
             Context:
             {context}
             Answer:"""
    rag_prompt_template = ChatPromptTemplate.from_template(SYS_PROMPT_RAG)

    # Web Search Re-writer Prompt
    SYS_WEB_SEARCH_PROMPT = """You are an expert question re-writer. Your task is to convert the user's question into an effective query optimized for a web search engine like Google or Tavily.
                   Focus on keywords and clear phrasing. Remove conversational elements. Maintain the core legal intent.
                   Example Input: "What are the specific rules for registering a private limited company in Dhaka, Bangladesh, including required documents and fees?"
                   Example Output: "private limited company registration Dhaka Bangladesh required documents fees"
                   """
    re_write_prompt_web = ChatPromptTemplate.from_messages([
        ("system", SYS_WEB_SEARCH_PROMPT),
        ("human", "Here is the initial question:\n{question}\nFormulate an improved web search query."),
    ])

    # Initial Query Re-writer Prompt
    SYS_INITIAL_QUERY_PROMPT = """You are a query optimizer for a vector database containing Bangladeshi legal documents.
                Analyze the input query, which has ALREADY been clarified through conversation.
                Your task is to refine this query slightly for optimal retrieval performance.
                Focus on extracting key legal terms, specific entities (laws, locations if relevant), and actions relevant to Bangladesh.
                Keep the query concise and targeted for semantic search. It should remain a single query.

                Example Input (Already Clarified): "Procedure for child custody application under Muslim Family Laws Ordinance in Dhaka Family Court"
                Example Output: "child custody application Muslim Family Laws Ordinance Dhaka Family Court procedure"

                Example Input (Already Clarified): "Requirements for bail in non-bailable offense under NDPS Act Bangladesh"
                Example Output: "bail requirements non-bailable offense NDPS Act Bangladesh"

                Refine the following clarified query for vector DB retrieval:
                Input Query:
                {question}
                """
    re_write_initial_prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_INITIAL_QUERY_PROMPT),
        ("human", "Refine the query based on the input provided above."),
    ])

    # --- Chains ---
    doc_grader = (grade_prompt | structured_llm_grader)
    question_rewriter_web = (re_write_prompt_web | rewriter_llm | StrOutputParser())
    initial_query_rewriter = (re_write_initial_prompt | rewriter_llm | StrOutputParser())

    def format_docs_with_sources(docs: List[Document]) -> str:
        """Formats documents for the RAG prompt, including sources."""
        if not docs:
            return "No relevant context found."
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            content = doc.page_content
            formatted.append(f"--- Context from: {source} ---\n{content}")
        return "\n\n".join(formatted)

    # QA RAG Chain
    qa_rag_chain = (
        {
            "context": itemgetter('context') | RunnableLambda(format_docs_with_sources),
            "question": itemgetter('question')
        }
        | rag_prompt_template
        | llm 
        | StrOutputParser()
    )

    # --- RAG Graph State ---
    class RAGGraphState(TypedDict):
        question: str           # The query being processed (initially clarified, then rewritten for DB, then for web)
        original_clarified_question: str  # Store the input from the clarification agent
        generation: str         # The final generated answer
        web_search_needed: str  # "Yes" or "No"
        documents: List[Document] # List of relevant documents

    # --- RAG Graph Nodes ---
    def rewrite_clarified_query_for_db(state: RAGGraphState):
        """Rewrites the CLARIFIED query for potentially better DB retrieval."""
        print("--- RAG NODE: Initial Query Rewriter (DB) ---")
        clarified_input = state["question"]
        print(f"Clarified Query Input for Rewriting: {clarified_input}")
        better_question_for_db = initial_query_rewriter.invoke({"question": clarified_input})
        print(f"Rewritten for DB: {better_question_for_db}")
        return {"question": better_question_for_db, "original_clarified_question": clarified_input}

    def retrieve_rag(state: RAGGraphState):
        """Retrieves documents from the vector database based on the DB-rewritten question."""
        print("--- RAG NODE: Retrieve from Vector DB ---")
        question_for_db = state["question"]
        original_clarified = state["original_clarified_question"]
        print(f"Retrieving for: {question_for_db}")
        try:
            documents = _retriever.invoke(question_for_db)
            print(f"Retrieved {len(documents)} documents.")
            return {"documents": documents, "original_clarified_question": original_clarified, "question": question_for_db}
        except Exception as e:
            print(f"---ERROR during RAG retrieval: {e}---")
            st.warning(f"Error during document retrieval: {e}")
            return {"documents": [], "original_clarified_question": original_clarified, "question": question_for_db}

    def grade_documents_rag_node(state: RAGGraphState):
        """Grades retrieved documents for relevance against the ORIGINAL CLARIFIED query."""
        print("--- RAG NODE: Grade Documents ---")
        question_to_grade_against = state["original_clarified_question"]
        documents = state["documents"]
        current_question = state["question"]
        print(f"Grading {len(documents)} documents against original clarified intent: {question_to_grade_against}")
        filtered_docs = []
        web_search_needed = "No"

        if not documents:
             print("---No documents retrieved, definitely needs web search.---")
             web_search_needed = "Yes"
        else:
            all_irrelevant = True
            for d in documents:
                doc_content = d.page_content
                doc_source = d.metadata.get("source", "Unknown Source")
                try:
                    score = doc_grader.invoke({"question": question_to_grade_against, "document": doc_content})
                    # Handle both structured and non-structured output
                    if hasattr(score, 'binary_score'):
                        grade = score.binary_score.lower().strip()
                    else:
                        # For models that don't support structured output, parse the response
                        grade_text = str(score).lower().strip()
                        grade = "yes" if "yes" in grade_text else "no"

                    if grade == "yes":
                        print(f"---GRADE: Relevant (Source: {doc_source})---")
                        filtered_docs.append(d)
                        all_irrelevant = False
                    else:
                        print(f"---GRADE: Not Relevant (Source: {doc_source})---")
                except Exception as e:
                    print(f"---ERROR Grading Document (Source: {doc_source}): {e}---")
                    web_search_needed = "Yes"

            if all_irrelevant and documents:
                print("---All retrieved documents were graded irrelevant.---")
                web_search_needed = "Yes"

        print(f"---Web Search Decision: {web_search_needed}---")
        return {"documents": filtered_docs, "original_clarified_question": question_to_grade_against, "question": current_question, "web_search_needed": web_search_needed}

    def rewrite_query_for_web_rag(state: RAGGraphState):
        """Rewrites the original clarified query for effective web search."""
        print("--- RAG NODE: Rewrite Query for Web Search ---")
        original_clarified = state["original_clarified_question"]
        print(f"Original clarified query for Web Rewrite: {original_clarified}")
        web_query = question_rewriter_web.invoke({"question": original_clarified})
        print(f"Rewritten for Web: {web_query}")
        return {"documents": state["documents"], "original_clarified_question": original_clarified, "question": web_query}

    def web_search_rag(state: RAGGraphState):
        """Performs web search using Tavily."""
        print("--- RAG NODE: Web Search ---")
        web_query = state["question"]
        original_documents = state["documents"]
        original_clarified = state["original_clarified_question"]
        print(f"Searching web for: {web_query}")

        try:
            docs_dict_list = tv_search.invoke(web_query)

            web_results_docs = []
            if docs_dict_list and isinstance(docs_dict_list, list):
                 content_limit = 3000
                 for doc_dict in docs_dict_list:
                     content = doc_dict.get("content", "")[:content_limit]
                     if content:
                         metadata = {"source": "Web Search", "url": doc_dict.get("url", "N/A")}
                         web_results_docs.append(Document(page_content=content, metadata=metadata))

            if web_results_docs:
                print(f"---Web Search added {len(web_results_docs)} results.---")
                combined_documents = original_documents + web_results_docs
                return {"documents": combined_documents, "original_clarified_question": original_clarified, "question": original_clarified}
            else:
                print("---Web Search returned no usable results.---")
                return {"documents": original_documents, "original_clarified_question": original_clarified, "question": original_clarified}

        except Exception as e:
            print(f"---ERROR During Web Search: {e}---")
            st.warning(f"Web search failed: {e}")
            return {"documents": original_documents, "original_clarified_question": original_clarified, "question": original_clarified}

    def generate_answer_rag(state: RAGGraphState):
        """Generates the final answer using the gathered context and original clarified query."""
        print("--- RAG NODE: Generate Answer ---")
        question_for_llm = state["original_clarified_question"]
        documents = state["documents"]
        print(f"Generating answer for original clarified intent: {question_for_llm}")
        print(f"Using {len(documents)} documents as context.")

        if not documents:
            print("---No documents available for generation.---")
            generation = f"I could not find relevant information in the local documents or via web search to answer your question: '{question_for_llm}'"
        else:
             try:
                 generation = qa_rag_chain.invoke({"context": documents, "question": question_for_llm})
                 print("---Generation Complete---")
             except Exception as e:
                 print(f"---ERROR During Generation: {e}---")
                 st.error(f"Error during answer generation: {e}")
                 generation = f"Sorry, an error occurred while generating the answer: {e}"

        return {"documents": documents, "original_clarified_question": question_for_llm, "generation": generation}

    # --- RAG Conditional Edges ---
    def decide_to_generate_rag(state: RAGGraphState):
        """Decides whether to proceed to generation or initiate web search."""
        print("--- RAG EDGE: Decide to Generate or Web Search ---")
        web_search_needed = state["web_search_needed"]

        if web_search_needed == "Yes":
            print("---DECISION: Routing to Web Search Path---")
            return "rewrite_query_web"
        else:
            print("---DECISION: Routing to Generate Answer (from DB Docs)---")
            return "generate_answer"

    # --- Build RAG Graph ---
    st.write("Building RAG Agent Graph...")
    rag_graph = StateGraph(RAGGraphState)

    # Add nodes
    rag_graph.add_node("query_rewriter_db", rewrite_clarified_query_for_db)
    rag_graph.add_node("retrieve", retrieve_rag)
    rag_graph.add_node("grade_documents", grade_documents_rag_node)
    rag_graph.add_node("rewrite_query_web", rewrite_query_for_web_rag)
    rag_graph.add_node("web_search", web_search_rag)
    rag_graph.add_node("generate_answer", generate_answer_rag)

    # Set entry point
    rag_graph.set_entry_point("query_rewriter_db")

    # Add edges
    rag_graph.add_edge("query_rewriter_db", "retrieve")
    rag_graph.add_edge("retrieve", "grade_documents")
    rag_graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate_rag,
        {
            "rewrite_query_web": "rewrite_query_web",
            "generate_answer": "generate_answer",
        },
    )
    rag_graph.add_edge("rewrite_query_web", "web_search")
    rag_graph.add_edge("web_search", "generate_answer")
    rag_graph.add_edge("generate_answer", END)

    try:
        st.write("Compiling RAG Agent Graph...")
        agentic_rag_compiled = rag_graph.compile()
        st.success("RAG Agent graph compiled successfully.")

        # --- Save RAG graph image ---
        try:
            st.write(f"Attempting to save RAG graph image to {RAG_GRAPH_IMAGE_PATH}...")
            if hasattr(agentic_rag_compiled, 'get_graph') and hasattr(agentic_rag_compiled.get_graph(), 'draw_mermaid_png'):
                 image_bytes = agentic_rag_compiled.get_graph().draw_mermaid_png()
                 with open(RAG_GRAPH_IMAGE_PATH, "wb") as f:
                     f.write(image_bytes)
                 st.info(f"RAG agent graph visualization saved as {RAG_GRAPH_IMAGE_PATH}")
            else:
                st.warning("Could not generate RAG graph image: Method not found or dependencies missing.")
        except ImportError as img_e:
             st.warning(f"Could not save RAG agent graph image due to missing dependencies: {img_e}.")
        except Exception as img_e:
            st.warning(f"Could not save RAG agent graph image: {img_e}")

        return agentic_rag_compiled
    except Exception as e:
        st.error(f"Failed to compile RAG agent graph: {e}")
        traceback.print_exc()
        return None

# ==============================================================================
# --- Streamlit UI (Updated for Multi-Provider Support) ---
# ==============================================================================

st.set_page_config(page_title="Multi-LLM Legal RAG Chatbot (Bangladesh)", layout="wide")

# --- Guidelines Sidebar ---
with st.sidebar:
    with st.expander("📋 GUIDELINES FOR USING THE CHATBOT", expanded=False):
        st.markdown("""
        ### 📁 About the Data
        
        The chatbot has access to **22+ comprehensive legal documents** covering:
        - **Constitutional Law**: Bangladesh Constitution
        - **Criminal Law**: Penal Code, Criminal Procedure Code, Criminal Law Amendments
        - **Civil Law**: Code of Civil Procedure, Citizenship Act
        - **Land Law**: State Acquisition & Tenancy Act, Land Reforms Ordinance, Agricultural Land Laws
        - **Commercial Law**: Negotiable Instruments Act (Cheque Dishonour cases)
        - **Social Law**: Domestic Violence Laws, Right to Information Act
        - **Administrative Law**: Laws Revision & Declaration Acts
        
        ---
        
        ### 🤖 How the Chatbot Works
        
        **Two-Stage Intelligence System:**
        
        1. **Clarification Agent** 📝
           - Analyzes your query for clarity and completeness
           - Asks follow-up questions if more details are needed
           - Maximum 5 clarification rounds to avoid endless loops
           - Synthesizes your final legal question
        
        2. **RAG (Retrieval-Augmented Generation) Agent** 🧠
           - Searches local legal database using semantic similarity
           - Grades retrieved documents for relevance
           - Falls back to web search if local documents insufficient
           - Combines multiple sources for comprehensive answers
           - Cites sources (PDF files or web search results)
        
        ---
        
        ### 💬 Query Types Supported
        
        **✅ Excellent for:**
        - Legal procedure questions (*"How to file a bail application?"*)
        - Specific law interpretations (*"Section 138 cheque dishonour penalties"*)
        - Rights and obligations (*"Property inheritance rights in Bangladesh"*)
        - Court procedures (*"Steps for filing a civil suit"*)
        - Constitutional matters (*"Fundamental rights under Bangladesh Constitution"*)
        
        **⚠️ Limited support for:**
        - Very recent legal changes (post-2023)
        - Highly specialized commercial law
        - International law matters
        - Personal legal advice
        
        ---
        
        ### 🎯 Best Practices
        
        **For Best Results:**
        - **Be Specific**: Include relevant details (location, law type, circumstances)
        - **Mention Context**: Personal, business, criminal, civil case?
        - **Ask Follow-ups**: The chatbot learns from conversation
        - **Verify Important Info**: Use responses as guidance, not definitive legal advice
        
        **Example Good Queries:**
        - *"What documents are needed to file a domestic violence case in Dhaka Family Court?"*
        - *"Bail procedure for non-bailable offenses under Section 497 CrPC"*
        - *"Land registration process under State Acquisition and Tenancy Act"*
        
        **Limitations:**
        - Responses are informational, not professional legal advice
        - Always consult qualified lawyers for actual cases
        - Database reflects laws as of 2023 and earlier
        """)

st.title("⚖️ Multi-LLM Bangladeshi Legal Assistant")

# --- Configuration UI ---
with st.sidebar:
    # Show provider help
    show_provider_help()
    
    # Configuration mode selection
    config_mode = st.radio(
        "Configuration Mode",
        ["🚀 Multi-Provider (New)", "🔧 Legacy (OpenAI Only)"],
        help="Choose between the new multi-provider system or legacy OpenAI-only mode"
    )
    
    if config_mode == "🚀 Multi-Provider (New)":
        # Use new configuration system
        app_config = create_configuration_ui()
        show_configuration_status(app_config)
        
        if not app_config.is_valid():
            st.error("⚠️ Please complete the configuration above to proceed.")
            st.stop()
            
    else:
        # Legacy mode
        st.subheader("🔧 Legacy Configuration")
        openai_api_key_input, tavily_api_key_input = get_legacy_config()
        
        if not (openai_api_key_input and tavily_api_key_input):
            st.warning("Please enter both OpenAI and Tavily API keys to proceed.")
            st.stop()
        
        # Create app config from legacy inputs
        app_config = create_legacy_compatibility_wrapper(openai_api_key_input, tavily_api_key_input)
        
        if not app_config.is_valid():
            st.error("Failed to initialize with legacy configuration.")
            st.stop()

# --- Resource Initialization ---
status_placeholder = st.empty()
status_placeholder.info("Initializing resources...")

# Initialize vector database
vector_db = load_or_create_vector_db(app_config.embedding_model)
if not vector_db:
    st.error("Vector Database could not be initialized.")
    st.stop()

# Initialize retriever
try:
    similarity_retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": MAX_DOCS_PER_QUERY, "score_threshold": MINIMUN_RETRIVAL_SCORE}
    )
    st.success("✅ Retriever initialized")
except Exception as e:
    st.error(f"Error setting up retriever: {e}")
    st.stop()

# Initialize clarification agent
if config_mode == "🚀 Multi-Provider (New)":
    clarification_agent = get_clarification_agent_with_provider(app_config.llm_config.dict())
else:
    clarification_agent = get_clarification_agent_legacy(app_config.llm_config.api_key)

if not clarification_agent:
    st.error("Clarification Agent failed to initialize.")
    st.stop()

# Initialize RAG agent
if config_mode == "🚀 Multi-Provider (New)":
    rag_app = get_agentic_rag_app_with_provider(similarity_retriever, app_config.llm_config.dict(), app_config.tavily_api_key)
else:
    rag_app = get_agentic_rag_app_legacy(similarity_retriever, app_config.llm_config.api_key, app_config.tavily_api_key)

if not rag_app:
    st.error("RAG Agent application failed to compile.")
    st.stop()

status_placeholder.success("🎉 All resources initialized successfully!")

# --- Display current configuration ---
with st.sidebar:
    st.markdown("---")
    st.markdown("**Current Configuration:**")
    st.markdown(f"**LLM Provider:** {app_config.llm_config.provider.value.title()}")
    st.markdown(f"**Model:** {app_config.llm_config.model}")
    st.markdown(f"**Temperature:** {app_config.llm_config.temperature}")
    
    st.markdown("---")
    st.markdown("Powered by LangChain, LangGraph, Multi-LLM Providers, ChromaDB, Tavily & Streamlit")
    
    # Show graph images if available
    if os.path.exists(CLARIFICATION_GRAPH_IMAGE_PATH):
        st.subheader("Clarification Agent Workflow")
        st.image(CLARIFICATION_GRAPH_IMAGE_PATH, use_column_width=True)
    if os.path.exists(RAG_GRAPH_IMAGE_PATH):
        st.subheader("RAG Agent Workflow")
        st.image(RAG_GRAPH_IMAGE_PATH, use_column_width=True)


# --- Chat Interface ---
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me any question about the Bangladeshi laws. I'll ask for clarification if needed."}]
# NEW state variables for clarification flow
if "clarification_active" not in st.session_state:
    st.session_state.clarification_active = False # Is the clarification agent running?
if "clarification_history" not in st.session_state:
    st.session_state.clarification_history = [] # Stores HumanMessage/AIMessage during clarification
if "initial_query" not in st.session_state:
    st.session_state.initial_query = None # Stores the very first user query of a conversation turn


# --- Helper Function to Run Clarification Step ---
def run_clarification_step(user_input: Optional[str] = None):
    """
    Runs one step of the clarification agent.
    Takes optional user_input if this is a response to a question.
    Updates session state and returns the agent's response (question or final status).
    """
    if not st.session_state.clarification_active: # First turn
        st.session_state.initial_query = user_input # Store the initial query
        current_history = [HumanMessage(content=user_input)]
        st.session_state.clarification_history = current_history
        st.session_state.clarification_active = True
        current_turn = 0
    else: # Subsequent turn
        if user_input: # Should always have user input here
            st.session_state.clarification_history.append(HumanMessage(content=user_input))
        current_history = st.session_state.clarification_history
        # Retrieve current turn from state (assuming it was stored by the agent's output)
        # This needs careful handling - the agent node needs to output the incremented turn.
        # For simplicity here, let's just count messages. A bit less robust.
        current_turn = len([msg for msg in current_history if isinstance(msg, AIMessage)])


    if not clarification_agent:
         st.error("Clarification Agent is not available.")
         st.session_state.clarification_active = False # Stop the loop
         return "Sorry, the clarification component is not working.", None

    try:
        # Define the input state for the clarification agent
        input_state = ClarificationState(
            initial_query=st.session_state.initial_query,
            conversation_history=current_history,
            max_turns=MAX_QUERY_CLARIFICATION_TURNS,
            current_turn=current_turn,
            clarified_query=None,
            ask_user_question=None # Agent will fill this
        )

        # Invoke the clarification agent
        # The agent will run until it hits END or the 'ask_user' state
        final_clarification_state = None
        for event in clarification_agent.stream(input_state, config=RunnableConfig(recursion_limit=10)):
            # In LangGraph >= 0.2.0, stream returns events with keys being node names
             last_event_key = list(event.keys())[-1]
             final_clarification_state = event[last_event_key]
             print(f"--- Clarification Event: {last_event_key} ---")
             # print(f"State: {final_clarification_state}") # Debugging

             # Check if the agent decided to ask a question
             if last_event_key == 'ask_user' and final_clarification_state.get("ask_user_question"):
                 # Stop streaming, we need user input
                 print("--- Stopping stream to ask user ---")
                 break
             # Check if the agent reached the end (clarified or max turns)
             elif last_event_key == END:
                 print("--- Clarification agent reached END ---")
                 break


        if not final_clarification_state:
             raise ValueError("Clarification agent did not return a final state.")


        # Process the final state of this step
        next_question = final_clarification_state.get("ask_user_question")
        clarified_query = final_clarification_state.get("clarified_query")

        if next_question:
            # Agent needs more info - ask the user
            st.session_state.clarification_history.append(AIMessage(content=next_question)) # Add AI question to history
            return next_question, None # Return question to display
        elif clarified_query:
            # Clarity achieved or max turns reached
            st.session_state.clarification_active = False # End clarification phase
            st.session_state.clarification_history = [] # Clear history for next interaction
            st.session_state.initial_query = None
            print(f"--- Clarification Complete. Final Query: {clarified_query} ---")
            return "Okay, I have enough details now. Processing your request...", clarified_query # Return confirmation and the final query
        else:
            # Should not happen if agent logic is correct, but handle defensively
            st.warning("Clarification agent finished unexpectedly. Proceeding with initial query.")
            st.session_state.clarification_active = False
            st.session_state.clarification_history = []
            final_q = st.session_state.initial_query
            st.session_state.initial_query = None
            return "Something went wrong during clarification. I'll try with your original query.", final_q

    except Exception as e:
        st.error(f"Error during clarification step: {e}")
        traceback.print_exc()
        st.session_state.clarification_active = False # Stop loop on error
        st.session_state.clarification_history = []
        final_q = st.session_state.initial_query
        st.session_state.initial_query = None
        return f"An error occurred during clarification: {e}. Trying with the original query.", final_q


# --- Main Chat Logic ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your legal question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Clarification Phase ---
    if st.session_state.clarification_active:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            agent_response, clarified_query = run_clarification_step(user_input=prompt)
            message_placeholder.markdown(agent_response) # Display question or confirmation
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

            # If clarification finished AND we got a query, run RAG
            if not st.session_state.clarification_active and clarified_query:
                 message_placeholder_rag = st.empty() # New placeholder for RAG result
                 message_placeholder_rag.markdown("Now running the main legal analysis... 🧠")
                 if rag_app:
                     try:
                         inputs = {"question": clarified_query} # Use the final clarified query
                         final_rag_state = rag_app.invoke(inputs, config=RunnableConfig(recursion_limit=25)) # Increased limit
                         rag_response = final_rag_state.get("generation", "Sorry, I couldn't generate a final response after analysis.")
                         message_placeholder_rag.markdown(rag_response)
                         st.session_state.messages.append({"role": "assistant", "content": rag_response})
                     except Exception as e:
                         st.error(f"An error occurred during RAG agent execution: {e}")
                         error_message = f"Sorry, I encountered an error during the final analysis. Error: {e}"
                         message_placeholder_rag.markdown(error_message)
                         st.session_state.messages.append({"role": "assistant", "content": error_message})
                 else:
                     err_msg = "The main RAG agent is not ready."
                     message_placeholder_rag.markdown(err_msg)
                     st.session_state.messages.append({"role": "assistant", "content": err_msg})

    # --- Initial Query - Start Clarification ---
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing your query... 🤔")
            # Start the clarification process
            agent_response, clarified_query = run_clarification_step(user_input=prompt)
            message_placeholder.markdown(agent_response) # Display first question or confirmation
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

            # If clarification finished immediately (query was clear) AND we got a query, run RAG
            if not st.session_state.clarification_active and clarified_query:
                 message_placeholder_rag = st.empty()
                 message_placeholder_rag.markdown("Now running the main legal analysis... 🧠")
                 if rag_app:
                     try:
                         inputs = {"question": clarified_query}
                         final_rag_state = rag_app.invoke(inputs, config=RunnableConfig(recursion_limit=25))
                         rag_response = final_rag_state.get("generation", "Sorry, I couldn't generate a final response.")
                         message_placeholder_rag.markdown(rag_response)
                         st.session_state.messages.append({"role": "assistant", "content": rag_response})
                     except Exception as e:
                         st.error(f"An error occurred during RAG agent execution: {e}")
                         error_message = f"Sorry, I encountered an error during the final analysis. Error: {e}"
                         message_placeholder_rag.markdown(error_message)
                         st.session_state.messages.append({"role": "assistant", "content": error_message})
                 else:
                     err_msg = "The main RAG agent is not ready."
                     message_placeholder_rag.markdown(err_msg)
                     st.session_state.messages.append({"role": "assistant", "content": err_msg})