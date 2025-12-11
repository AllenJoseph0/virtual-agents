import io
import os
import re
import gc
import json
import time
import math
import shutil
import logging
import random
import asyncio
import aiofiles
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional, TypedDict, Union
from uuid import uuid4
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import pandas as pd
from PIL import Image

# Playwright for Agentic Browser Automation
from playwright.async_api import async_playwright

# FastAPI Imports
from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, Form, BackgroundTasks, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv
import requests
from langdetect import detect, LangDetectException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus

from langchain_qdrant import QdrantVectorStore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    DataFrameLoader,
)
from langchain_unstructured import UnstructuredLoader
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from groq import RateLimitError
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.memory import ConversationSummaryBufferMemory
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import whisper
import pytesseract
import google.generativeai as genai
from google.cloud import texttospeech
import pymysql
from elevenlabs.client import ElevenLabs
from deepgram import DeepgramClient
import uvicorn

# MCP Imports
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types

load_dotenv()

# -----------------------------------------------------------------------------
# Config & paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "ai_server.log")

# Shared directories
UPLOAD_ROOT = os.path.join(BASE_DIR, "data", "uploads")
VECTORSTORE_ROOT = os.path.join(BASE_DIR, "vectorstores")
DB_DIR = os.path.join(BASE_DIR, "db")
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, 'chat_histories')
PERSONAS_FILE = os.path.join(DB_DIR, "personas.json")
PERMISSIONS_FILE = os.path.join(DB_DIR, "permissions.json") # Added for linking agents to personas

# AI-Server specific directories
TEMP_FOLDER = os.path.join(BASE_DIR, "temp")


for d in [LOGS_DIR, UPLOAD_ROOT, VECTORSTORE_ROOT, TEMP_FOLDER, DB_DIR, CHAT_HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)

# Logging
log_fmt = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
fh.setFormatter(log_fmt)
ch = logging.StreamHandler()
ch.setFormatter(log_fmt)
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

# Env vars & DB connection
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_DATABASE"),
    'port': int(os.getenv("DB_PORT", 3306)),
    'cursorclass': pymysql.cursors.DictCursor
}
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "rag-")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
INITIAL_K = int(os.getenv("INITIAL_K", 25))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 8000))
MAX_GRAPH_TURNS = int(os.getenv("MAX_GRAPH_TURNS", 10))
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", 3000))
MAX_QUESTION_GEN_CONTEXT = int(os.getenv("MAX_QUESTION_GEN_CONTEXT", 10000))


CONTENT_PAYLOAD_KEY = "page_content"

# Supported file types for indexing
AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]

# Caches
GOOGLE_VOICES_CACHE = {}
VISION_MODEL_CACHE = {}

# -----------------------------------------------------------------------------
# Model init
# -----------------------------------------------------------------------------
try:
    logger.info("Initializing models and clients…")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    VECTOR_SIZE = len(embeddings.embed_query("probe"))
    logger.info(f"Embeddings ready. vector_size={VECTOR_SIZE}")

    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
    logger.info(f"Whisper '{WHISPER_MODEL_NAME}' loaded")

    qdrant = QdrantClient(url=QDRANT_URL, timeout=180)
    logger.info(f"Qdrant client -> {QDRANT_URL}")

    cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL_NAME)
    logger.info("Cross-encoder model initialized")

    logger.info("All components loaded")
except Exception as e:
    logger.exception(f"Fatal during init: {e}")
    raise

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RETRIEVER_CACHE: Dict[str, Any] = {}
MEMORY_SESSIONS: Dict[str, ConversationSummaryBufferMemory] = {}

# -----------------------------------------------------------------------------
# Helper for MCP Compatibility
# -----------------------------------------------------------------------------
class NoOpResponse(Response):
    """
    A custom response class that does nothing when executed.
    This is required for MCP endpoints where the response has already been
    manually sent via the ASGI 'send' channel by the MCP library, 
    preventing FastAPI from attempting to send a second response.
    """
    async def __call__(self, scope, receive, send):
        pass

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------
class PersonaCreate(BaseModel):
    name: str
    firm_id: int
    prompt: Optional[str] = None
    voice_prompt: Optional[str] = None
    stages: Optional[List[str]] = None

class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    voice_prompt: Optional[str] = None
    stages: Optional[List[str]] = None

class IndexRequest(BaseModel):
    username: str
    category: str
    firm_id: int = 0

class DeleteIndexRequest(BaseModel):
    username: str
    category: str

class BatchStatusCheckRequest(BaseModel):
    categories: List[Dict[str, str]]

class RagChainRequest(BaseModel):
    owner_id: str
    category: str
    question: str
    session_id: str
    persona_id: str
    firm_id: int
    compliance_rules: Optional[str] = None
    rulebook: Optional[Dict[str, Any]] = None
    call_state: Optional[str] = "initial"
    query_source: Optional[str] = "text"

class TtsRequest(BaseModel):
    text: str
    provider: str = "google"
    firm_id: int
    code: Optional[str] = None
    language: Optional[str] = None

class GreetingRequest(BaseModel):
    persona_id: Optional[str] = None
    firmId: Optional[int] = None
    provider: str = "google"
    code: Optional[str] = None
    language: Optional[str] = None

class DemoRequest(BaseModel):
    persona_id: Optional[str] = None
    firmId: Optional[int] = None
    provider: str = "google"
    code: Optional[str] = None
    language: Optional[str] = None

class TestRequest(BaseModel):
    owner_id: str
    category: str
    persona_id: str
    firmId: int
    num_questions: int = 10
    compliance_rules: Optional[str] = None

class BrowserTaskRequest(BaseModel):
    firm_id: int
    url: str
    instruction: str
    task_type: str = "general" # options: general, google_form, google_doc, google_sheet, social_comment


# -----------------------------------------------------------------------------
# LLM and API Key Management
# -----------------------------------------------------------------------------
def get_db():
    """Dependency that opens a database connection."""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

def get_api_key(firm_id: int, provider: str, db_conn) -> Optional[str]:
    """Fetches the active API key for a given firm and provider using the provided db connection."""
    if not firm_id: return None
    try:
        with db_conn.cursor() as cursor:
            sql = """
                SELECT API_KEY FROM LLM_DETAILS
                WHERE FIRMID = %s AND LLM_PROVIDER = %s AND STATUS = 'ACTIVE'
                ORDER BY UPD_DTM DESC LIMIT 1
            """
            cursor.execute(sql, (firm_id, provider))
            result = cursor.fetchone()
            return result['API_KEY'] if result else None
    except pymysql.Error as e:
        logger.error(f"Database error in get_api_key: {e}")
        raise

def get_llm(firm_id: int, preferred_provider: str = 'GROQ', db_conn=None):
    """
    Dynamically initializes and returns an LLM instance based on available keys.
    """
    local_conn = False
    if db_conn is None:
        db_conn = pymysql.connect(**DB_CONFIG)
        local_conn = True

    try:
        api_key = get_api_key(firm_id, preferred_provider, db_conn)
        if not api_key:
            raise ValueError(f"{preferred_provider} API key is not configured for firm {firm_id}.")

        if preferred_provider == 'GROQ':
            return ChatGroq(temperature=0.2, groq_api_key=api_key, model_name="llama-3.1-8b-instant")
        elif preferred_provider == 'GEMINI':
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)
        elif preferred_provider == 'OPENAI_GPT4':
            return ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=api_key, temperature=0.2)

        logger.warning(f"Preferred provider '{preferred_provider}' not supported. Falling back to Groq.")
        groq_api_key = get_api_key(firm_id, 'GROQ', db_conn)
        if not groq_api_key:
            raise ValueError("Fallback Groq API key is not configured for this firm.")
        return ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    finally:
        if local_conn:
            db_conn.close()


# -----------------------------------------------------------------------------
# Smart Browser Agent (New Agentic Capability)
# -----------------------------------------------------------------------------
class SmartBrowserAgent:
    """
    An agentic tool that uses Playwright to perform end-to-end tasks on websites.
    It uses an LLM to "see" the Accessibility Tree of the page and decide which elements to interact with.
    """
    def __init__(self, firm_id: int, db_conn):
        self.llm = get_llm(firm_id, db_conn=db_conn)

    async def run_task(self, url: str, instruction: str, task_type: str):
        logger.info(f"Starting Browser Agent Task: {task_type} on {url}")
        
        async with async_playwright() as p:
            # Launch in headless mode for server environment
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                viewport={"width": 1280, "height": 720},
                locale="en-US"
            )
            page = await context.new_page()

            try:
                await page.goto(url, timeout=60000)
                # Wait for network idle to ensure dynamic content loads
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except:
                    logger.warning("Network idle timeout, proceeding anyway.")

                result_message = ""

                if task_type == "google_form" or task_type == "general":
                    result_message = await self._handle_smart_form(page, instruction)
                elif task_type == "google_doc":
                    result_message = await self._handle_google_doc(page, instruction)
                elif task_type == "google_sheet":
                    result_message = await self._handle_google_sheet(page, instruction)
                elif task_type == "social_comment":
                    result_message = await self._handle_social_comment(page, instruction)
                else:
                    result_message = "Unknown task type."

                # Slight delay to ensure auto-saves (for Google Docs/Sheets)
                await asyncio.sleep(2)
                
                logger.info(f"Browser Agent Task Completed: {result_message}")
                return {"status": "success", "message": result_message, "url": url}

            except Exception as e:
                logger.error(f"Browser Task Failed: {e}")
                screenshot_path = os.path.join(TEMP_FOLDER, f"error_{uuid4().hex}.png")
                await page.screenshot(path=screenshot_path)
                logger.info(f"Error screenshot saved to {screenshot_path}")
                raise HTTPException(status_code=500, detail=f"Browser automation failed: {str(e)}")
            finally:
                await browser.close()

    async def _handle_smart_form(self, page, instruction):
        """
        Extracts the Accessibility Tree to understand the form structure, 
        then uses LLM to map user data to specific fields.
        """
        # Get the accessibility tree (semantic structure of the page)
        snapshot = await page.accessibility.snapshot()
        
        # Optimize snapshot for LLM (reduce token usage)
        snapshot_str = json.dumps(snapshot, indent=2)[:15000] # Limit size

        prompt_template = """
        You are an advanced Browser Automation Agent.
        Your task is to fill a form on a webpage based on the user's instructions.
        
        **User Instruction:** {instruction}
        
        **Page Accessibility Tree (Semantic Structure):**
        {snapshot}
        
        **Goal:** Identify the input fields (Role: 'textbox', 'combobox', 'checkbox', etc.) that match the user's data.
        
        **Return:** A JSON list of actions. Each action must have:
        - "role": The accessibility role to target (e.g., "textbox", "button").
        - "name": The accessibility name/label to target (e.g., "First Name").
        - "action": "fill" or "click".
        - "value": The value to type (for "fill").
        
        Example:
        [
            {{"role": "textbox", "name": "Email", "action": "fill", "value": "test@example.com"}},
            {{"role": "button", "name": "Submit", "action": "click"}}
        ]
        
        Return ONLY valid JSON.
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | JsonOutputParser()
        
        actions = await chain.ainvoke({"instruction": instruction, "snapshot": snapshot_str})
        
        results = []
        for act in actions:
            try:
                # Playwright's "get_by_role" is robust for accessibility-based selection
                locator = page.get_by_role(act["role"], name=act["name"])
                
                if act["action"] == "fill":
                    await locator.fill(act["value"])
                    results.append(f"Filled {act['name']}")
                elif act["action"] == "click":
                    await locator.click()
                    results.append(f"Clicked {act['name']}")
                    
                await asyncio.sleep(0.5) # Human-like delay
            except Exception as e:
                logger.warning(f"Failed to execute action {act}: {e}")
                
        return f"Form actions executed: {', '.join(results)}"

    async def _handle_google_doc(self, page, content):
        """
        Specific logic for Google Docs.
        Google Docs uses a canvas, so standard inputs don't work. We focus the document and simulate typing.
        """
        # Wait for the main document canvas/editable area
        # Google Docs usually exposes an element with role="document" or "textbox" in the accessibility tree
        try:
            # Wait for the editor to load
            await page.wait_for_selector(".kix-appview-editor", timeout=10000)
            
            # Click the editing area to ensure focus
            # We target the specific class Google uses for the page canvas
            await page.click(".kix-appview-editor") 
            
            # Clear existing content (Select All -> Backspace) - Optional, depends on use case
            # await page.keyboard.press("Control+A")
            # await page.keyboard.press("Backspace")
            
            # Type the new content
            # We use type() instead of fill() because it sends keystrokes, which canvas apps require
            await page.keyboard.type(content, delay=50) # 50ms delay mimics human typing
            
            return "Typed content into Google Doc."
        except Exception as e:
            # Fallback: Try accessibility selector
            try:
                await page.get_by_role("textbox", name="Document content").click()
                await page.keyboard.type(content)
                return "Typed content using Accessibility selector."
            except Exception as e2:
                raise RuntimeError(f"Could not write to Google Doc: {e}")

    async def _handle_google_sheet(self, page, data_instruction):
        """
        Specific logic for Google Sheets.
        Navigates the grid using keyboard commands.
        """
        try:
            # Wait for grid to load
            await page.wait_for_selector(".grid-container", timeout=10000)
            
            # Click top-left cell (approximate) or focus grid
            await page.click(".grid-container")
            
            # Use LLM to parse what data goes where?
            # For simplicity in this agent, we append data to the current cell or active selection
            # A more complex agent would parse "Put X in cell A1", but here we assume "Write X"
            
            # Simple interaction: Type data, press Enter
            await page.keyboard.type(data_instruction)
            await page.keyboard.press("Enter")
            
            return "Typed data into Google Sheet active cell."
        except Exception as e:
             raise RuntimeError(f"Could not write to Google Sheet: {e}")

    async def _handle_social_comment(self, page, comment):
        """
        General logic for social media commenting.
        Finds the 'Write a comment' box and 'Post' button.
        """
        snapshot = await page.accessibility.snapshot()
        snapshot_str = json.dumps(snapshot)[:10000]

        prompt_template = """
        You are a social media bot. 
        Analyze the page structure to find the "Comment" or "Reply" input area and the "Post/Reply" button.
        
        **User Comment:** {comment}
        **Page Structure:** {snapshot}
        
        Return JSON actions:
        [
           {{"role": "textbox", "name": "Write a comment...", "action": "fill", "value": "{comment}"}},
           {{"role": "button", "name": "Post", "action": "click"}}
        ]
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | JsonOutputParser()
        actions = await chain.ainvoke({"comment": comment, "snapshot": snapshot_str})
        
        for act in actions:
            try:
                locator = page.get_by_role(act["role"], name=act["name"])
                if act["action"] == "fill":
                    # Social media often requires clicking the box first
                    await locator.click() 
                    await locator.fill(act["value"])
                elif act["action"] == "click":
                    await locator.click()
            except Exception as e:
                logger.warning(f"Social action failed: {e}")
                
        return "Posted comment on social media."

# -----------------------------------------------------------------------------
# Persona Engine
# -----------------------------------------------------------------------------
def _load_personas() -> Dict[str, Dict[str, Any]]:
    """Loads persona configurations from a JSON file."""
    if not os.path.exists(PERSONAS_FILE):
        return {}
    try:
        with open(PERSONAS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.exception("Failed to load personas.json, returning empty dict.")
        return {}

def _save_personas(personas: Dict[str, Dict[str, Any]]):
    """Saves the current personas dictionary to the JSON file."""
    try:
        with open(PERSONAS_FILE, "w") as f:
            json.dump(personas, f, indent=4)
    except IOError:
        logger.error("Failed to save personas.json")

def _load_permissions() -> Dict[str, Dict[str, Any]]:
    """Loads permissions configurations from a JSON file."""
    if not os.path.exists(PERMISSIONS_FILE):
        return {}
    try:
        with open(PERMISSIONS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.exception("Failed to load permissions.json, returning empty dict.")
        return {}


@app.post("/personas", status_code=201)
async def create_persona(payload: PersonaCreate, db=Depends(get_db)):
    name = payload.name
    firm_id = payload.firm_id

    try:
        llm = get_llm(firm_id, 'GROQ', db_conn=db)
    except Exception as e:
        logger.error(f"Failed to initialize LLM for persona generation for firm {firm_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not initialize LLM: {e}")

    generated_prompt = payload.prompt
    generated_voice_prompt = payload.voice_prompt
    generated_stages = payload.stages
    generated_stages_str = ", ".join(generated_stages) if generated_stages else ""

    try:
        # ---- ENHANCED TEXT PROMPT ----
        if not generated_prompt:
            logger.info(f"Generating TEXT persona prompt for '{name}'")
            text_prompt_template = """
You are a system prompt generator. Your task is to create a professional system prompt for a text-based AI assistant.

The persona for the AI assistant is "{name}".

Your output must be a single, complete system prompt that starts with "You are {name}, an AI assistant that..." and then defines the AI's identity, tone, and behavioral boundaries.

Follow these instructions for the content of the prompt you generate:
1.  Identity & Role: The AI should adopt the linguistic style, vocabulary, and reasoning typical of a {name}.
2.  Knowledge Boundaries: The AI must rely only on the documents provided. It must not hallucinate or use external information. When information is missing, it must say: "I don't see that information in the documents. Could you share more details?"
3.  Response Behavior: The AI should communicate clearly with short paragraphs and lists. It must maintain a professional and cooperative tone fitting the {name} persona.
4.  Fallback Handling: If the user's request is unclear, the AI should politely ask for clarification.

Generate ONLY the final system prompt. Do not include instructions, examples, or any markdown formatting (like asterisks or hashtags) in your output.
"""
            prompt_chain = ChatPromptTemplate.from_template(text_prompt_template) | llm | StrOutputParser()
            generated_prompt = prompt_chain.invoke({"name": name})

        # ---- ENHANCED VOICE PROMPT ----
        if not generated_voice_prompt:
            logger.info(f"Generating VOICE persona prompt for '{name}'")
            voice_prompt_template = """
You are an expert in designing voice-first AI personas. Your task is to generate a natural-sounding system prompt for a voice-based assistant with the persona of "{name}".

The generated system prompt must define how the AI speaks and behaves in a spoken conversation.

Your output must be a single, complete system prompt that starts with "You are {name}. Your purpose is to..." and then defines the voice persona's characteristics.

Follow these instructions for the content of the prompt you generate:
1.  Persona Identity: Define the personality, tone, and verbal rhythm of a {name}. The speech should sound human-like.
2.  Conversational Style: Instruct the AI to speak in concise, fluid sentences. It should use natural connectors (e.g., "Alright, let's see..."). Responses should be brief, typically under 2-3 short sentences.
3.  Delivery Guidelines: The AI must avoid robotic phrasing. It should summarize information conversationally and never read lists or URLs verbatim. The tone should be friendly and professional.
4.  Information Boundaries: If details are missing from documents, the AI should say: "I can't seem to find that in the provided information. Could you fill me in?" It must never invent facts.
5.  Voice Nuance: The AI can include small human-like pauses or softeners ("Let me think...") to sound more natural.

Generate ONLY the final system prompt. Do not include quotes, commentary, or any markdown formatting (like asterisks or hashtags) in your output.
"""
            voice_prompt_chain = ChatPromptTemplate.from_template(voice_prompt_template) | llm | StrOutputParser()
            generated_voice_prompt = voice_prompt_chain.invoke({"name": name})

        # ---- ENHANCED STAGES ----
        if not generated_stages_str:
            logger.info(f"Generating conversation stages for '{name}'")
            stages_template = """
You are a conversation design expert.
Define the high-level conversational flow for the persona "{name}".

Output must be a **single, comma-separated list** of concise, snake_case stage identifiers
representing a typical structured conversation for this role.

Guidelines:
- Keep stage names short (2–4 words).
- Include logical flow: greeting → context_building → inquiry → resolution → wrap_up.
- Example output: greeting, context_building, question_analysis, response_generation, follow_up, closing.

Do not include extra explanations, examples, or commentary — output only the list.
"""
            stages_chain = ChatPromptTemplate.from_template(stages_template) | llm | StrOutputParser()
            generated_stages_str = stages_chain.invoke({"name": name})

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM failed to generate persona component: {e}")

    final_stages = [s.strip() for s in generated_stages_str.split(",") if s.strip()]

    personas = _load_personas()
    persona_id = str(uuid4())
    personas[persona_id] = {
        "id": persona_id,
        "name": name,
        "prompt": generated_prompt,
        "voice_prompt": generated_voice_prompt,
        "stages": final_stages,
    }
    _save_personas(personas)

    return personas[persona_id]

@app.get("/personas")
async def get_personas():
    """Endpoint to retrieve all configured personas."""
    return list(_load_personas().values())


@app.put("/personas/{persona_id}")
async def update_persona(persona_id: str, payload: PersonaUpdate):
    """Endpoint to update an existing persona."""
    personas = _load_personas()
    if persona_id not in personas:
        raise HTTPException(status_code=404, detail="Persona not found")

    if payload.name:
        personas[persona_id]["name"] = payload.name
    if payload.prompt:
        personas[persona_id]["prompt"] = payload.prompt
    if payload.voice_prompt:
        personas[persona_id]["voice_prompt"] = payload.voice_prompt
    if payload.stages:
        personas[persona_id]["stages"] = payload.stages
    
    _save_personas(personas)
    return personas[persona_id]

@app.delete("/personas/{persona_id}", status_code=204)
async def delete_persona(persona_id: str):
    """Endpoint to delete a persona."""
    personas = _load_personas()
    if persona_id in personas:
        del personas[persona_id]
        _save_personas(personas)
    return

# -----------------------------------------------------------------------------
# Helpers & Decorators
# -----------------------------------------------------------------------------
def retry_with_backoff(retries=3, delay=1, backoff=2):
    """Decorator for retrying an async function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _delay = delay
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except RateLimitError as e:
                    logger.warning(f"Rate limit hit for {func.__name__}. Retrying in {_delay}s... ({i+1}/{retries})")
                    await asyncio.sleep(_delay)
                    _delay *= backoff
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}. Retrying in {_delay}s... ({i+1}/{retries})")
                    await asyncio.sleep(_delay)
                    _delay *= backoff
            return await func(*args, **kwargs) # Try one last time
        return wrapper
    return decorator

def apply_redaction_rules(text: str, rulebook: Optional[Any]) -> str:
    if not text: return text
    if isinstance(rulebook, dict):
        for pattern in rulebook.get("patterns", []):
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    return text

def get_dynamic_k(question: str) -> (int, int):
    question_len = len(question.split())
    if question_len < 5:
        initial_k, final_k = 30, 7
    elif question_len > 15:
        initial_k, final_k = 20, 4
    else:
        initial_k, final_k = INITIAL_K, 5
    logger.info(f"Dynamic K: q_len={question_len}, initial_k={initial_k}, final_k={final_k}")
    return initial_k, final_k

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "".join(ch for ch in text if ch.isprintable())

def clean_markdown_for_speech(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'^\s*[\*\-]\s+', '. ', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'(!\[[^\]]+\]\([^\)]+\))', r'', text)
    text = re.sub(r'(\*\*|__|~~)', '', text)
    text = re.sub(r'(\*|_|`)', '', text)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    text = text.replace('\n', ' ').strip()
    return text

def get_document_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf": return PyMuPDFLoader(file_path)
    if ext == ".docx": return Docx2txtLoader(file_path)
    if ext == ".pptx": return UnstructuredPowerPointLoader(file_path)
    if ext in [".md", ".txt", ".json"]: return TextLoader(file_path, autodetect_encoding=True)
    if ext in [".html", ".xml", ".eml"]: return UnstructuredLoader(file_path)
    if ext == ".csv": return DataFrameLoader(pd.read_csv(file_path, on_bad_lines="skip"))
    if ext == ".xlsx": return DataFrameLoader(pd.read_excel(file_path))
    return None

def transcribe_audio(path: str) -> str:
    """Transcribes audio using the globally loaded Whisper model."""
    result = whisper_model.transcribe(path, fp16=False)
    return clean_text(result.get("text", ""))

def get_vision_model(firm_id: int, db_conn):
    """Lazily initializes and caches a Gemini vision model for a specific firm."""
    if firm_id in VISION_MODEL_CACHE:
        return VISION_MODEL_CACHE[firm_id]
    
    api_key = get_api_key(firm_id, 'GEMINI', db_conn)
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        VISION_MODEL_CACHE[firm_id] = model
        logger.info(f"Initialized Gemini Vision for firm {firm_id}")
        return model
    return None

def process_image(file_path: str, firm_id: int, db_conn) -> str:
    """Processes an image using Gemini Vision if available, otherwise Tesseract OCR."""
    try:
        image = Image.open(file_path)
        vision_model = get_vision_model(firm_id, db_conn)
        if vision_model:
            prompt = "Extract all visible text from the image. Then, add a brief one-sentence description of the image for semantic context."
            response = vision_model.generate_content([prompt, image])
            return clean_text(response.text or "")
        
        logger.warning(f"No Gemini key for firm {firm_id}, falling back to Tesseract OCR.")
        return clean_text(pytesseract.image_to_string(image))
    except Exception as e:
        logger.error(f"Image processing failed for {file_path}: {e}")
        return ""

def _wait_for_collection_green(collection_name: str, timeout_s: int = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = qdrant.get_collection(collection_name=collection_name).status
            if status == CollectionStatus.GREEN: return True
        except Exception: pass
        time.sleep(0.5)
    logger.warning(f"{collection_name} not GREEN after {timeout_s}s")
    return False

def _vectorstore(collection_name: str) -> QdrantVectorStore:
    """Initializes the connection to a specific Qdrant vector collection."""
    return QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        embedding=embeddings, # Corrected from embeddings to embedding for langchain-qdrant
        content_payload_key=CONTENT_PAYLOAD_KEY,
    )

def _list_user_categories(username: str) -> List[str]:
    user_dir = os.path.join(UPLOAD_ROOT, username)
    if not os.path.isdir(user_dir): return []
    return sorted([d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))])


def google_tts_to_wav(text: str, language_code: str, voice_name: str, api_key: str) -> str:
    """Synthesizes speech from text using Google Cloud TTS and returns the path to a WAV file."""
    try:
        client_options = {"api_key": api_key}
        client = texttospeech.TextToSpeechClient(client_options=client_options)

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16 # WAV format
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        out_path = os.path.join(TEMP_FOLDER, f"tts_google_{uuid4().hex}.wav")
        with open(out_path, "wb") as out:
            out.write(response.audio_content)
        
        logger.info(f"Google TTS audio content written to file {out_path}")
        return out_path

    except Exception as e:
        logger.error(f"Google TTS synthesis failed: {e}")
        raise RuntimeError(f"Google TTS failed: {e}")

def elevenlabs_tts_to_wav(text: str, voice_id: str, api_key: str) -> str:
    """Synthesizes speech using ElevenLabs and saves to a WAV file."""
    try:
        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_16000" # Use PCM 16kHz
        )
        
        out_path = os.path.join(TEMP_FOLDER, f"tts_elevenlabs_{uuid4().hex}.wav")
        
        # Collect PCM data
        pcm_data = b""
        for chunk in audio_stream:
            pcm_data += chunk
            
        # Write WAV file
        import wave
        with wave.open(out_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_data)
                
        logger.info(f"ElevenLabs audio content written to file {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"ElevenLabs TTS synthesis failed: {e}")
        raise RuntimeError(f"ElevenLabs TTS failed: {e}")

def deepgram_tts_to_wav(text: str, model_name: str, api_key: str) -> str:
    """Synthesizes speech using Deepgram and saves to a WAV file by calling the REST API directly."""
    try:
        # The user selects a voice code (e.g., 'aura-luna-en'), which is passed as model_name.
        url = f"https://api.deepgram.com/v1/speak?model={model_name}&encoding=linear16&container=wav"
        
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        out_path = os.path.join(TEMP_FOLDER, f"tts_deepgram_{uuid4().hex}.wav")
        with open(out_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Deepgram TTS audio content written to file {out_path}")
        return out_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Deepgram TTS API request failed: {e}")
        if e.response is not None:
            logger.error(f"Deepgram API Response: {e.response.text}")
        raise RuntimeError(f"Deepgram TTS failed: API request error")
    except Exception as e:
        logger.error(f"Deepgram TTS synthesis failed: {e}")
        raise RuntimeError(f"Deepgram TTS failed: {e}")

def deepgram_stt(audio_path: str, api_key: str) -> str:
    """Transcribes audio using Deepgram SDK v3+."""
    try:
        # Initialize client (v3 syntax)
        client = DeepgramClient(api_key=api_key)

        # Read the audio file as bytes
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        # Transcribe the audio synchronously
        # Using the exact syntax from voice-server.py
        response = client.listen.v1.media.transcribe_file(
            request={"buffer": audio_data},
            model="nova-3",
            smart_format=True,
            language="en"
        )

        # Extract transcript text safely
        transcript = ""
        if hasattr(response, "results") and response.results.channels:
            alt = response.results.channels[0].alternatives
            if alt:
                transcript = alt[0].transcript or ""

        return transcript.strip()

    except Exception as e:
        logger.error(f"Deepgram STT failed: {e}")
        raise RuntimeError(f"Deepgram STT failed: {e}")


# -----------------------------------------------------------------------------
# Routes: structure & indexing
# -----------------------------------------------------------------------------
@app.post("/clear-cache")
async def clear_cache():
    """Endpoint to clear the in-memory retriever cache."""
    global RETRIEVER_CACHE
    count = len(RETRIEVER_CACHE)
    RETRIEVER_CACHE.clear()
    await asyncio.to_thread(gc.collect) # Force garbage collection
    logger.info(f"Cleared {count} items from retriever cache.")
    return {"message": f"Cache cleared. {count} items removed."}

@app.get("/structure/{username}")
async def structure(username: str):
    user_dir = os.path.join(UPLOAD_ROOT, username)
    result = {username: []}
    if os.path.isdir(user_dir):
        for cat in sorted(os.listdir(user_dir)):
            cdir = os.path.join(user_dir, cat)
            if not os.path.isdir(cdir): continue
            collection_name = f"{QDRANT_COLLECTION_PREFIX}{username}-{cat}"
            try:
                # Use asyncio.to_thread for potentially blocking network call
                status_result = await asyncio.to_thread(qdrant.get_collection, collection_name=collection_name)
                status = status_result.status
                index_status = "ACTIVE" if status == CollectionStatus.GREEN else "INACTIVE"
            except Exception:
                index_status = "INACTIVE"
            
            # CHANGED: Gather file metadata (name, size, date) instead of just strings
            files_data = []
            if os.path.isdir(cdir):
                for f in os.listdir(cdir):
                    f_path = os.path.join(cdir, f)
                    if os.path.isfile(f_path):
                        stat = os.stat(f_path)
                        files_data.append({
                            "name": f,
                            "size": stat.st_size,
                            "date": time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))
                        })
            
            # Sort alphabetically
            files_data.sort(key=lambda x: x['name'])

            result[username].append({"name": cat, "files": files_data, "indexStatus": index_status})
    return result

def _iter_category_docs(username: str, category: str, firm_id: int, db_conn):
    cdir = os.path.join(UPLOAD_ROOT, username, category)
    if not os.path.isdir(cdir): return
    for name in os.listdir(cdir):
        path = os.path.join(cdir, name)
        ext = os.path.splitext(name)[1].lower()

        if ext in AUDIO_EXTENSIONS:
            transcript = transcribe_audio(path)
            if transcript: yield Document(page_content=transcript, metadata={"source": name})
            continue
        if ext in IMAGE_EXTENSIONS:
            ocr_text = process_image(path, firm_id, db_conn)
            if ocr_text: yield Document(page_content=ocr_text, metadata={"source": name})
            continue

        loader = get_document_loader(path)
        if not loader: continue
        try:
            for d in loader.load():
                d.page_content = clean_text(getattr(d, "page_content", ""))
                d.metadata.setdefault("source", name)
                yield d
        except Exception as e:
            logger.error(f"Loader failed for {path}: {e}")

def _chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

@app.post("/create-index")
async def create_index(request: IndexRequest, db=Depends(get_db)):
    username, category, firm_id = request.username, request.category, request.firm_id
    if not all([username, category, firm_id]):
        raise HTTPException(status_code=400, detail="username, category, and firm_id are required")

    collection_name = f"{QDRANT_COLLECTION_PREFIX}{username}-{category}"
    
    # Run heavy processing in thread
    def process_index():
        docs = list(_iter_category_docs(username, category, firm_id, db))
        chunks = _chunk_docs(docs)
        logger.info(f"Indexing {len(chunks)} chunks for {username}/{category}.")

        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        vs = _vectorstore(collection_name)
        if chunks: vs.add_documents(chunks)
        return len(chunks)

    try:
        chunk_count = await asyncio.to_thread(process_index)

        # Wait for the collection to become queryable before returning success
        if not await asyncio.to_thread(_wait_for_collection_green, collection_name):
            logger.warning(f"Collection '{collection_name}' did not become active in time after creation.")
            return {"message": f"Indexed {chunk_count} chunks. Index is optimizing and will be active shortly."}

        return {"message": f"Indexed {chunk_count} chunks. Index is now active."}
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create index")

@app.post("/update-index")
async def update_index(request: IndexRequest, db=Depends(get_db)):
    username, category, firm_id = request.username, request.category, request.firm_id
    if not all([username, category, firm_id]):
        raise HTTPException(status_code=400, detail="username, category, and firm_id are required")

    collection_name = f"{QDRANT_COLLECTION_PREFIX}{username}-{category}"

    try:
        def process_update():
            try:
                qdrant.get_collection(collection_name=collection_name)
            except Exception:
                logger.warning(f"Update called on non-existent collection '{collection_name}'. Deferring to create_index.")
                docs = list(_iter_category_docs(username, category, firm_id, db))
                chunks = _chunk_docs(docs)
                logger.info(f"Creating collection and indexing {len(chunks)} chunks for {username}/{category}.")
                qdrant.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                vs = _vectorstore(collection_name)
                if chunks: vs.add_documents(chunks)
                return {"message": f"Created index and added {len(chunks)} chunks."}

            vs = _vectorstore(collection_name)
            cdir = os.path.join(UPLOAD_ROOT, username, category)
            if not os.path.isdir(cdir):
                return {"message": "No new documents to add. Category directory not found."}

            disk_files = set(os.listdir(cdir))

            indexed_sources = set()
            response, _ = qdrant.scroll(
                collection_name=collection_name,
                with_payload=["metadata"],
                limit=10000
            )
            for point in response:
                if point.payload and "metadata" in point.payload and "source" in point.payload["metadata"]:
                    indexed_sources.add(point.payload["metadata"]["source"])

            new_files = disk_files - indexed_sources

            if not new_files:
                return {"message": "Index is already up-to-date. No new documents found."}

            logger.info(f"Found {len(new_files)} new documents to index for {username}/{category}: {list(new_files)}")

            docs_to_add = []
            for file_name in new_files:
                file_path = os.path.join(cdir, file_name)

                ext = os.path.splitext(file_name)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    transcript = transcribe_audio(file_path)
                    if transcript: docs_to_add.append(Document(page_content=transcript, metadata={"source": file_name}))
                    continue
                if ext in IMAGE_EXTENSIONS:
                    ocr_text = process_image(file_path, firm_id, db)
                    if ocr_text: docs_to_add.append(Document(page_content=ocr_text, metadata={"source": file_name}))
                    continue

                loader = get_document_loader(file_path)
                if not loader: continue
                try:
                    for d in loader.load():
                        d.page_content = clean_text(getattr(d, "page_content", ""))
                        d.metadata.setdefault("source", file_name)
                        docs_to_add.append(d)
                except Exception as e:
                    logger.error(f"Loader failed during update for {file_path}: {e}")

            if not docs_to_add:
                return {"message": "Found new files, but they produced no content to index."}

            chunks = _chunk_docs(docs_to_add)
            logger.info(f"Adding {len(chunks)} new chunks to collection '{collection_name}'.")

            if chunks:
                vs.add_documents(chunks)
            
            return {"message": f"Successfully added {len(new_files)} new document(s) ({len(chunks)} chunks) to the index."}

        return await asyncio.to_thread(process_update)

    except Exception as e:
        logger.exception(f"Update index failed for {username}/{category}")
        raise HTTPException(status_code=500, detail="Failed to update index.")


@app.post("/delete-index")
async def delete_index(request: DeleteIndexRequest):
    """Deletes a Qdrant collection (index) but keeps the source files."""
    username, category = request.username, request.category
    if not all([username, category]):
        raise HTTPException(status_code=400, detail="username and category are required")

    collection_name = f"{QDRANT_COLLECTION_PREFIX}{username}-{category}"
    try:
        await asyncio.to_thread(qdrant.delete_collection, collection_name=collection_name)
        logger.info(f"Deleted index (collection): {collection_name}")
        return {"message": f"Index '{category}' deleted successfully."}
    except Exception as e:
        logger.warning(f"Could not delete index {collection_name}: {e}")
        return {"message": f"Index '{category}' was already inactive or could not be deleted."}

@app.post("/delete-category")
async def delete_category(request: DeleteIndexRequest):
    """Deletes an entire category, including its index and all source files."""
    username, category = request.username, request.category
    if not all([username, category]):
        raise HTTPException(status_code=400, detail="username and category are required")

    collection_name = f"{QDRANT_COLLECTION_PREFIX}{username}-{category}"
    category_dir = os.path.join(UPLOAD_ROOT, username, category)

    try:
        def process_delete():
            try:
                qdrant.delete_collection(collection_name=collection_name)
                logger.info(f"Deleted index for category deletion: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete index for category {collection_name} (it may not have existed): {e}")

            if os.path.isdir(category_dir):
                shutil.rmtree(category_dir)
                logger.info(f"Deleted file directory: {category_dir}")

        await asyncio.to_thread(process_delete)
        return {"message": f"Category '{category}' and its index were permanently deleted."}

    except Exception as e:
        logger.error(f"Failed to delete category {username}/{category}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete category.")

@app.post("/batch-status-check")
async def batch_status_check(request: BatchStatusCheckRequest):
    categories = request.categories
    results = []
    
    async def check_single_category(cat_info):
        name, owner = cat_info.get("name"), cat_info.get("owner")
        if not name or not owner: return None
        collection_name = f"{QDRANT_COLLECTION_PREFIX}{owner}-{name}"
        try:
            status_result = await asyncio.to_thread(qdrant.get_collection, collection_name=collection_name)
            status = status_result.status
            index_status = "ACTIVE" if status == CollectionStatus.GREEN else "INACTIVE"
        except Exception:
            index_status = "INACTIVE"
        return {"name": name, "owner": owner, "indexStatus": index_status}

    tasks = [check_single_category(cat) for cat in categories]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# -----------------------------------------------------------------------------
# LangGraph RAG chain
# -----------------------------------------------------------------------------
class GraphState(TypedDict):
    collection_name: str
    question: str
    original_question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    persona_id: str
    response: str
    turns: int
    compliance_rules: Optional[str]
    firm_id: int
    llm: Any
    call_state: str
    query_source: str # Added to distinguish between 'text' and 'voice' queries


def get_llm_node(state: GraphState) -> GraphState:
    """Initializes the LLM for the current firm."""
    logger.info("---LANGGRAPH NODE: get_llm_node---")
    try:
        llm_instance = get_llm(state['firm_id'])
        return {**state, "llm": llm_instance}
    except Exception as e:
        logger.error(f"Failed to initialize LLM for firm {state['firm_id']}: {e}")
        # Return a response indicating the failure
        return {**state, "response": f"LLM Error: {e}", "context": []}


@retry_with_backoff()
async def compliance_check_node(state: GraphState) -> GraphState:
    """Checks if the user's question violates any compliance rules."""
    logger.info("---LANGGRAPH NODE: compliance_check_node---")
    rules = state.get("compliance_rules")
    if not rules:
        logger.info("No compliance rules found, skipping check.")
        return {**state}

    compliance_prompt_template = """
    You are a strict Compliance Officer AI. Your task is to determine if a user's question violates any of the provided compliance rules.

    **Compliance Rules:**
    ---
    {rules}
    ---

    **User's Question:**
    "{question}"

    **Instructions:**
    1.  Carefully read each rule and its corresponding weight (e.g., "Do not give financial advice, 95%").
    2.  Analyze the user's question to see if it asks for information related to any of these rules.
    3.  Consider the weight. A higher weight means the rule is more critical.
    4.  You MUST respond with a single JSON object containing two keys:
        - "decision": A string, either "ALLOWED" or "DENIED".
        - "reason": A brief, one-sentence explanation for your decision, especially if denied.
    """
    
    prompt = ChatPromptTemplate.from_template(compliance_prompt_template)
    chain = prompt | state['llm'] | JsonOutputParser()
    
    response = await chain.ainvoke({
        "rules": rules,
        "question": state["question"]
    })

    decision = response.get("decision", "DENIED").upper()
    reason = response.get("reason", "No reason provided.")
    logger.info(f"Compliance check result: {decision}. Reason: {reason}")

    if decision == "DENIED":
        return {**state, "response": f"I am sorry, but I cannot answer that question. Reason: {reason}", "context": []}
    
    return {**state}

@retry_with_backoff()
async def query_rewrite_node(state: GraphState) -> GraphState:
    """Rewrites the user's question to be a standalone query."""
    logger.info("---LANGGRAPH NODE: query_rewrite_node---")
    if not state["chat_history"]:
        return {**state, "original_question": state["question"]}

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the conversation history and a follow-up question, rephrase the question to be a standalone query."),
        MessagesPlaceholder("chat_history"),
        ("human", "Follow-up question: {question}"),
        ("system", "Standalone question:"),
    ])

    rewriter = rewrite_prompt | state['llm'] | StrOutputParser()
    refined_query = await rewriter.ainvoke({"chat_history": state["chat_history"], "question": state["question"]})
    logger.info(f"Rewritten query: '{refined_query}'")

    return {**state, "question": refined_query, "original_question": state["question"]}

@retry_with_backoff()
async def retrieve_documents_node(state: GraphState) -> GraphState:
    """Retrieves and re-ranks documents from the vector store."""
    logger.info("---LANGGRAPH NODE: retrieve_documents_node---")
    turns = state.get("turns", 0) + 1
    if turns >= MAX_GRAPH_TURNS:
        return {**state, "turns": turns, "context": [], "response": "MAX_TURNS_EXCEEDED"}

    initial_k, final_k = get_dynamic_k(state["question"])
    
    def run_retrieval():
        retriever = _vectorstore(state["collection_name"]).as_retriever(search_kwargs={"k": initial_k})
        docs = retriever.invoke(state["question"])
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=final_k)
        return reranker.compress_documents(docs, query=state["question"])

    ranked_docs = await asyncio.to_thread(run_retrieval)

    return {**state, "context": ranked_docs, "turns": turns}

@retry_with_backoff()
async def generate_response_node(state: GraphState) -> GraphState:
    """Generates the final response using the persona's prompt."""
    logger.info(f"---LANGGRAPH NODE: generate_response_node---")

    personas = await asyncio.to_thread(_load_personas)
    persona_config = personas.get(state["persona_id"])
    if not persona_config:
        return {**state, "response": "Error: Persona configuration not found."}

    query_source = state.get("query_source", "text")
    default_prompt = "You are a helpful assistant. Answer the question based on the context provided."

    if query_source == 'voice' and persona_config.get("voice_prompt"):
        active_prompt = persona_config.get("voice_prompt")
        logger.info("Using VOICE prompt for response generation.")
    else:
        active_prompt = persona_config.get("prompt")
        logger.info("Using TEXT prompt for response generation.")
    
    active_prompt = active_prompt or default_prompt # Fallback

    context_str = "\n\n".join([f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(state["context"])])
    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS]

    prompt = ChatPromptTemplate.from_messages([
        ("system", active_prompt),
        MessagesPlaceholder("chat_history"),
        ("system", "Context Documents:\n---\n{context_str}\n---"),
        ("human", "{question}"),
    ])

    chain = prompt | state['llm'] | StrOutputParser()
    response = await chain.ainvoke({
        "chat_history": state["chat_history"],
        "context_str": context_str or "No context provided.",
        "question": state["original_question"]
    })

    return {**state, "response": response}

@retry_with_backoff()
async def update_call_state_node(state: GraphState) -> GraphState:
    """Analyzes the conversation and updates the call_state."""
    logger.info("---LANGGRAPH NODE: update_call_state_node---")
    personas = await asyncio.to_thread(_load_personas)
    persona_config = personas.get(state["persona_id"], {})
    # Ensure stages has a default value if it's missing, None, or an empty list
    stages = persona_config.get("stages") or ["general_conversation"]

    # If there's no history, we are in the first stage.
    if not state["chat_history"]:
        # Now this is safe because `stages` will always have at least one element.
        return {**state, "call_state": stages[0]}

    state_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are a conversation analyst. Your task is to determine the current stage of the conversation.
The possible stages are: {', '.join(stages)}.
Based on the latest user question and the AI's response, which stage is the conversation in now?
Respond with ONLY the name of the stage (e.g., 'problem_identification').
"""),
        MessagesPlaceholder("chat_history"),
        ("human", "User's latest question: {question}"),
        ("ai", "AI's latest response: {response}"),
        ("system", "Current Stage:"),
    ])

    chain = state_prompt | state['llm'] | StrOutputParser()
    new_state = await chain.ainvoke({
        "chat_history": state["chat_history"],
        "question": state["original_question"],
        "response": state["response"]
    })
    
    # Clean up the response to ensure it's a valid stage
    final_state = new_state.strip().lower().replace(" ", "_")
    if final_state not in stages:
        final_state = state.get("call_state", stages[0]) # Default to old state if invalid
        
    logger.info(f"Updated call state to: '{final_state}'")
    return {**state, "call_state": final_state}


def should_continue(state: GraphState) -> str:
    """Determines the next step after a node."""
    if state.get("response"): # If any node generated a final response (e.g., error)
        return "end"
    return "continue"


workflow = StateGraph(GraphState)

workflow.add_node("get_llm", get_llm_node)
workflow.add_node("compliance_check", compliance_check_node)
workflow.add_node("rewrite_query", query_rewrite_node)
workflow.add_node("retrieve", retrieve_documents_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("update_call_state", update_call_state_node) # <-- ADDED NODE

workflow.set_entry_point("get_llm")

workflow.add_conditional_edges(
    "get_llm",
    should_continue,
    {"continue": "compliance_check", "end": END}
)
workflow.add_conditional_edges(
    "compliance_check",
    should_continue,
    {"continue": "rewrite_query", "end": END}
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("retrieve", "generate_response")
workflow.add_edge("generate_response", "update_call_state") # <-- MODIFIED EDGE
workflow.add_edge("update_call_state", END) # <-- MODIFIED EDGE


# Compile the graph into a runnable application
lang_graph_app = workflow.compile()


# -----------------------------------------------------------------------------
# Main RAG Logic (Extracted for internal reuse)
# -----------------------------------------------------------------------------
async def process_rag_query(
    owner_id: str,
    category: str,
    question: str,
    session_id: str,
    firm_id: int,
    persona_id: str = None,
    compliance_rules: str = None,
    rulebook: dict = None,
    query_source: str = "text",
    db_conn = None
) -> Dict[str, Any]:
    """Reusable RAG logic for both API endpoints and MCP tools."""
    
    # 1. Get Memory (Session history)
    # Note: For MCP, we might want session persistence or start fresh each time.
    # Here we assume session_id is provided by caller to maintain context.
    if session_id not in MEMORY_SESSIONS:
        try:
            llm = get_llm(firm_id, db_conn=db_conn)
            MEMORY_SESSIONS[session_id] = ConversationSummaryBufferMemory(
                llm=llm, max_token_limit=MEMORY_TOKEN_LIMIT, return_messages=True
            )
        except Exception as e:
            logger.error(f"Failed to create memory for session {session_id}: {e}")
            raise

    memory = MEMORY_SESSIONS[session_id]

    # 2. Prepare LangGraph State
    initial_state = {
        "collection_name": f"{QDRANT_COLLECTION_PREFIX}{owner_id}-{category}",
        "question": question.strip(),
        "chat_history": memory.chat_memory.messages,
        "persona_id": persona_id,
        "turns": 0,
        "compliance_rules": compliance_rules,
        "firm_id": firm_id,
        "call_state": "initial",
        "query_source": query_source,
    }

    # 3. Run Graph
    final_state = await lang_graph_app.ainvoke(initial_state)
    final_answer = apply_redaction_rules(final_state.get("response", "An error occurred."), rulebook)

    # 4. Update Memory
    if "LLM Error" not in final_answer:
        memory.save_context({"input": question}, {"output": final_answer})

    return {
        "answer": final_answer,
        "sources": [d.metadata for d in final_state.get("context", [])],
        "call_state": final_state.get("call_state", "unknown")
    }


# -----------------------------------------------------------------------------
# Main RAG Endpoint (Uses extracted logic)
# -----------------------------------------------------------------------------
@app.post("/rag/chain")
async def rag_chain_endpoint(request: RagChainRequest, db=Depends(get_db)):
    try:
        result = await process_rag_query(
            owner_id=request.owner_id,
            category=request.category,
            question=request.question,
            session_id=request.session_id,
            firm_id=request.firm_id,
            persona_id=request.persona_id,
            compliance_rules=request.compliance_rules,
            rulebook=request.rulebook,
            query_source=request.query_source,
            db_conn=db
        )
        return result

    except RateLimitError as e:
        logger.error(f"Rate limit error for firm {request.firm_id}: {e}")
        raise HTTPException(status_code=429, detail="The API rate limit has been reached. Please check your plan or try again later.")
    except Exception as e:
        logger.exception("RAG chain endpoint error")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")


# =============================================================================
# MCP SERVER INTEGRATION
# =============================================================================
mcp_server = Server("v-agents-mcp")

# GLOBAL TRANSPORT: Must be shared between GET and POST endpoints
sse = SseServerTransport("/mcp/messages")

@mcp_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    Dynamically lists a Tool for every Agent found in the UPLOAD_ROOT directory.
    Uses directory structure as source of truth for available agents.
    """
    tools = []
    
    if os.path.exists(UPLOAD_ROOT):
        # Scan all user directories
        for username in os.listdir(UPLOAD_ROOT):
            user_dir = os.path.join(UPLOAD_ROOT, username)
            if not os.path.isdir(user_dir): continue
            
            # Scan all agent categories for this user
            for agent_name in os.listdir(user_dir):
                agent_path = os.path.join(user_dir, agent_name)
                if not os.path.isdir(agent_path): continue
                
                # Sanitize name for tool: "Fitness Agent" -> "ask_fitness_agent"
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', agent_name.lower())
                tool_name = f"ask_{clean_name}"
                
                tools.append(
                    types.Tool(
                        name=tool_name,
                        description=f"Ask the '{agent_name}' agent (Owner: {username}) about its knowledge base.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The specific question to ask the agent."
                                },
                                "user_id": {
                                    "type": "string",
                                    "description": "The User ID identifying the agent owner (required)."
                                },
                                "firm_id": {
                                    "type": "integer",
                                    "description": "The Firm ID for LLM API access (required)."
                                }
                            },
                            "required": ["question", "user_id", "firm_id"]
                        }
                    )
                )
    return tools

@mcp_server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Routes the tool call to the correct Agent logic internally.
    """
    if not arguments:
        raise ValueError("Missing arguments")
    
    question = arguments.get("question")
    user_id = arguments.get("user_id")
    firm_id = arguments.get("firm_id")

    if not all([question, user_id, firm_id]):
        raise ValueError("Missing required arguments: question, user_id, or firm_id")

    # Reverse engineer the tool name to find the agent category
    # name is like "ask_fitness_agent". We need to match it against actual folders.
    target_category = None
    
    # We need to re-scan to find the matching folder name because sanitization is lossy
    if os.path.exists(UPLOAD_ROOT):
        user_dir = os.path.join(UPLOAD_ROOT, user_id)
        if os.path.isdir(user_dir):
            for agent_name in os.listdir(user_dir):
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', agent_name.lower())
                candidate_tool_name = f"ask_{clean_name}"
                if candidate_tool_name == name:
                    target_category = agent_name
                    break
    
    if not target_category:
        return [types.TextContent(type="text", text=f"Error: Agent corresponding to tool '{name}' not found for user '{user_id}'.")]

    # Execute RAG query using the internal logic
    # We use a distinct session ID for MCP calls to avoid polluting web chat history
    try:
        # DB Connection for LLM key retrieval
        conn = pymysql.connect(**DB_CONFIG)
        
        # LINKING LOGIC: Find Persona ID via permissions.json
        # Format in permissions.json is: "USERID-CategoryName" -> { "personaId": "..." }
        permission_key = f"{user_id}-{target_category}"
        permissions_data = _load_permissions()
        
        # Get the persona ID if it exists, otherwise None
        agent_persona_id = None
        if permission_key in permissions_data:
             agent_persona_id = permissions_data[permission_key].get("personaId")

        logger.info(f"MCP Call: Agent='{target_category}', PersonaID='{agent_persona_id}'")

        try:
            result = await process_rag_query(
                owner_id=user_id,
                category=target_category,
                question=question,
                session_id=f"mcp-{user_id}-{target_category}",
                firm_id=int(firm_id),
                persona_id=agent_persona_id, # DYNAMIC PERSONA ID
                query_source="text",
                db_conn=conn
            )
            
            answer = result.get("answer", "No answer.")
            sources = result.get("sources", [])
            
            source_text = ""
            if sources:
                source_list = list(set([s.get("source", "unknown") for s in sources]))
                source_text = "\n\nSources:\n- " + "\n- ".join(source_list)
                
            return [types.TextContent(type="text", text=f"{answer}{source_text}")]
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"MCP Tool Execution Failed: {e}")
        return [types.TextContent(type="text", text=f"Error processing query: {str(e)}")]


@app.get("/mcp/sse")
async def handle_sse(request: Request):
    """
    Exposes the MCP server via Server-Sent Events (SSE).
    This allows external MCP clients (like Claude Desktop) to connect via HTTP.
    """
    # Initialize options (Synchronous call, not a context manager)
    init_options = mcp_server.create_initialization_options()
    
    # SseServerTransport.connect_sse handles the full ASGI response lifecycle for SSE
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(streams[0], streams[1], init_options)
    
    # RETURN NO-OP TO PREVENT DOUBLE-RESPONSE
    return NoOpResponse()

@app.post("/mcp/messages")
async def handle_messages(request: Request):
    """
    Handles JSON-RPC messages for the MCP SSE transport.
    """
    await sse.handle_post_message(request.scope, request.receive, request._send)
    
    # RETURN NO-OP TO PREVENT DOUBLE-RESPONSE
    return NoOpResponse()
# -----------------------------------------------------------------------------
# New Agentic Browser Route
# -----------------------------------------------------------------------------
@app.post("/agent/browser-task")
async def browser_task_endpoint(request: BrowserTaskRequest, db=Depends(get_db)):
    """
    Triggers the end-to-end browser agent.
    Task Type options: 'google_form', 'google_doc', 'google_sheet', 'social_comment', 'general'
    """
    agent = SmartBrowserAgent(request.firm_id, db_conn=db)
    return await agent.run_task(request.url, request.instruction, request.task_type)
# -----------------------------------------------------------------------------
# STT & TTS Routes (Rest of file...)
# -----------------------------------------------------------------------------

@app.post("/voice/stt")
async def stt_endpoint(
    firm_id: int = Form(...),
    provider: str = Form("whisper"),
    audio: UploadFile = File(...),
    db=Depends(get_db)
):
    provider = provider.lower() # 'whisper' or 'deepgram'

    in_path = os.path.join(TEMP_FOLDER, f"stt_{uuid4().hex}")
    
    try:
        async with aiofiles.open(in_path, 'wb') as buffer:
            content = await audio.read()
            await buffer.write(content)

        if provider == 'deepgram':
            api_key = get_api_key(firm_id, 'DEEPGRAM', db)
            if not api_key:
                raise HTTPException(status_code=400, detail="Deepgram API key not configured for this firm.")
            text = await asyncio.to_thread(deepgram_stt, in_path, api_key)
        else: # Default to whisper
            text = await asyncio.to_thread(transcribe_audio, in_path)
        
        return {"text": text, "provider": provider}

    except Exception as e:
        logger.error(f"STT failed with provider {provider}: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
    finally:
        if os.path.exists(in_path):
            try:
                await asyncio.to_thread(os.remove, in_path)
            except Exception:
                pass


@app.get("/voice/list-voices")
async def list_voices():
    # Deprecated endpoint as Piper is removed. Returning empty list.
    return []

@app.get("/voice/list-google-voices")
async def list_google_voices(firm_id: int, db=Depends(get_db)):
    if not firm_id:
        raise HTTPException(status_code=400, detail="firm_id is required")
        
    api_key = get_api_key(firm_id, 'GOOGLE_TTS', db)

    if not api_key:
        raise HTTPException(status_code=400, detail="Google TTS API key is not configured for this firm.")

    approved_voices = {
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Wavenet-D",
        "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-D",
        "ta-IN-Wavenet-A", "ta-IN-Wavenet-B", "te-IN-Wavenet-A", "te-IN-Wavenet-B",
        "kn-IN-Wavenet-A", "kn-IN-Wavenet-B", "ml-IN-Wavenet-A", "ml-IN-Wavenet-B",
        "gu-IN-Wavenet-A", "gu-IN-Wavenet-B", "pa-IN-Wavenet-A", "pa-IN-Wavenet-B",
        "mr-IN-Wavenet-A", "mr-IN-Wavenet-B", "ur-IN-Wavenet-A", "ur-IN-Wavenet-B",
        "en-US-Wavenet-C", "en-US-Wavenet-F", "en-US-Wavenet-G",
        "en-GB-Wavenet-A", "en-GB-Wavenet-C", "en-GB-Wavenet-F",
        "en-AU-Wavenet-A", "en-AU-Wavenet-C"
    }

    api_key_hash = str(hash(api_key))
    if api_key_hash in GOOGLE_VOICES_CACHE:
        logger.info("Returning cached and filtered Google TTS voices.")
        cached_voices = GOOGLE_VOICES_CACHE[api_key_hash]
        filtered_voices = [v for v in cached_voices if v['code'] in approved_voices]
        return filtered_voices

    try:
        logger.info("Fetching and filtering Google TTS voices from API.")
        
        def fetch_voices():
            client_options = {"api_key": api_key}
            client = texttospeech.TextToSpeechClient(client_options=client_options)
            return client.list_voices()

        response = await asyncio.to_thread(fetch_voices)
        
        all_formatted_voices = []
        for voice in response.voices:
            if voice.name in approved_voices:
                lang_code = voice.language_codes[0] if voice.language_codes else "unknown"
                gender = str(voice.ssml_gender).split('.')[-1].capitalize()
                
                lang_name, country_name = (lang_code.split('-') + [None, None])[:2]
                
                name_map = {"hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi", "mr": "Marathi", "ur": "Urdu"}
                country_map = {"IN": "India", "US": "USA", "GB": "UK", "AU": "Australia"}

                lang_display = name_map.get(lang_name, lang_name)
                country_display = country_map.get(country_name, country_name)
                
                display_name = f"{lang_display} ({country_display}) - {voice.name.split('-')[-1]} ({gender})"
                
                all_formatted_voices.append({
                    "code": voice.name, "name": display_name, "language": lang_code, "isGoogle": True,
                })
        
        GOOGLE_VOICES_CACHE[api_key_hash] = all_formatted_voices
        return all_formatted_voices

    except Exception as e:
        logger.error(f"Failed to fetch Google TTS voices: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch voices from Google Cloud.")

@app.get("/voice/list-elevenlabs-voices")
async def list_elevenlabs_voices(firm_id: int, db=Depends(get_db)):
    api_key = get_api_key(firm_id, 'ELEVENLABS', db)

    if not api_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API key is not configured for this firm.")
    
    try:
        def fetch_voices():
            client = ElevenLabs(api_key=api_key)
            return client.voices.get_all()

        voices = await asyncio.to_thread(fetch_voices)
        
        formatted_voices = []
        for voice in voices.voices:
            labels = voice.labels if voice.labels else {}
            formatted_voices.append({
                "code": voice.voice_id,
                "name": voice.name,
                "accent": labels.get('accent'),
                "gender": labels.get('gender'),
                "age": labels.get('age'),
                "isElevenLabs": True
            })
        return formatted_voices
    except Exception as e:
        logger.error(f"Failed to fetch ElevenLabs voices: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch voices from ElevenLabs.")

@app.get("/voice/list-deepgram-voices")
async def list_deepgram_voices():
    """
    Provides a curated list of known-to-work Deepgram Aura models.
    Some models from the initial list were causing "No such model/version" errors.
    """
    dg_voices_final = [
        {"code": "aura-asteria-en", "name": "Asteria", "isDeepgram": True},
        {"code": "aura-luna-en", "name": "Luna", "isDeepgram": True},
        {"code": "aura-stella-en", "name": "Stella", "isDeepgram": True},
        {"code": "aura-athena-en", "name": "Athena", "isDeepgram": True},
        {"code": "aura-hera-en", "name": "Hera", "isDeepgram": True},
        {"code": "aura-orpheus-en", "name": "Orpheus", "isDeepgram": True},
        {"code": "aura-arcas-en", "name": "Arcas", "isDeepgram": True},
        {"code": "aura-zeus-en", "name": "Zeus", "isDeepgram": True},
    ]
    return dg_voices_final


@app.post("/voice/tts")
@retry_with_backoff()
async def tts_endpoint(request: TtsRequest, background_tasks: BackgroundTasks, db=Depends(get_db)):
    text = clean_markdown_for_speech(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    provider = request.provider.lower() # google, elevenlabs, deepgram
    firm_id = request.firm_id
    voice_code = request.code # Voice ID/Name for the provider
    language = request.language # Language code, mainly for Google

    wav_path = None
    audio_buffer = None

    try:
        provider_map = {
            'google': 'GOOGLE_TTS',
            'elevenlabs': 'ELEVENLABS',
            'deepgram': 'DEEPGRAM'
        }
        api_key_provider = provider_map.get(provider, provider.upper())
        api_key = get_api_key(firm_id, api_key_provider, db)
        
        if not api_key:
             raise ValueError(f"{provider.capitalize()} API key is not configured for this firm.")

        if provider == 'elevenlabs':
            if not voice_code: raise ValueError("ElevenLabs TTS requires a 'code' (voice_id).")
            logger.info(f"Attempting TTS with ElevenLabs, voice: {voice_code}")
            wav_path = await asyncio.to_thread(elevenlabs_tts_to_wav, text, voice_code, api_key)

        elif provider == 'deepgram':
            if not voice_code: raise ValueError("Deepgram TTS requires a 'code' (model name).")
            logger.info(f"Attempting TTS with Deepgram, model: {voice_code}")
            wav_path = await asyncio.to_thread(deepgram_tts_to_wav, text, voice_code, api_key)

        elif provider == 'google':
            if not voice_code or not language:
                raise ValueError("Google TTS requires 'code' (voice name) and 'language'.")
            logger.info(f"Attempting TTS with Google Cloud, voice: {voice_code}")
            wav_path = await asyncio.to_thread(google_tts_to_wav, text, language, voice_code, api_key)

        else:
             raise ValueError(f"Unknown or unsupported provider: {provider}")
        
        # Read the generated file into an in-memory buffer
        async with aiofiles.open(wav_path, 'rb') as f:
            audio_data = await f.read()
            audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)
        
        # Clean up the temporary file in background
        if wav_path:
             background_tasks.add_task(os.remove, wav_path)

        return StreamingResponse(audio_buffer, media_type="audio/wav")
        
    except ValueError as ve:
        logger.warning(f"TTS configuration error for provider '{provider}': {ve}")
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except: pass
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"TTS generation failed for provider '{provider}': {e}")
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except: pass
        raise HTTPException(status_code=500, detail="Failed to generate speech")


@app.post("/voice/greeting")
@retry_with_backoff()
async def tts_greeting(request: GreetingRequest, background_tasks: BackgroundTasks, db=Depends(get_db)):
    persona_id = request.persona_id
    firm_id = request.firmId
    
    provider = request.provider.lower()
    code = request.code
    language = request.language

    wav_path = None
    llm = None
    
    try:
        if firm_id:
            llm = get_llm(firm_id, db_conn=db)
        else:
            logger.warning("No firmId provided for greeting generation; will use default text.")
    except ValueError as e:
        logger.warning(f"Could not get LLM for firm {firm_id}: {e}. Using default greeting text.")

    personas = await asyncio.to_thread(_load_personas)
    persona_config = personas.get(persona_id, {})
    persona_name = persona_config.get("name", "the assistant")
    
    lang_key = (language or str(code) or "").split("-")[0].lower()
    lang_map = {"en": "English", "hi": "Hindi", "ml": "Malayalam", "te": "Telugu", "ta": "Tamil", "bn": "Bengali", "mr": "Marathi"}
    lang_name = lang_map.get(lang_key, "English")

    default_greetings = {
        "en": f"Hello, I'm {persona_name}. How can I help you today?",
        "hi": f"नमस्ते, मैं {persona_name} हूँ। मैं आज आपकी कैसे मदद कर सकता हूँ?",
    }
    text_to_speak = default_greetings.get(lang_key)

    if not text_to_speak and llm:
        try:
            persona_context = persona_config.get("voice_prompt", persona_config.get("prompt", "A helpful assistant."))
            generation_prompt = (
                f"You are writing a script for an AI voice assistant named '{persona_name}'. "
                f"Its core personality is: '{persona_context[:250]}...'.\n\n"
                f"Your task is to generate a single, short, welcoming opening greeting IN THE {lang_name.upper()} LANGUAGE. "
                "It should be one sentence and sound natural. "
                "Output ONLY the greeting text in that language. Do not add quotes."
            )
            text_to_speak = await llm.ainvoke(generation_prompt)
            text_to_speak = text_to_speak.content
            logger.info(f"Generated greeting for '{persona_name}' in '{lang_name}': '{text_to_speak}'")

        except Exception as e:
            logger.error(f"Greeting LLM generation failed: {e}")
            text_to_speak = "Hello, how can I help you?"
    elif not text_to_speak:
        text_to_speak = default_greetings.get("en")


    try:
        provider_map = {
            'google': 'GOOGLE_TTS',
            'elevenlabs': 'ELEVENLABS',
            'deepgram': 'DEEPGRAM'
        }
        api_key_provider = provider_map.get(provider, provider.upper())
        api_key = get_api_key(firm_id, api_key_provider, db)

        if not api_key:
             raise ValueError(f"{provider.capitalize()} API key is not configured for this firm.")

        if provider == 'elevenlabs':
            wav_path = await asyncio.to_thread(elevenlabs_tts_to_wav, text_to_speak, code, api_key)
        elif provider == 'deepgram':
            wav_path = await asyncio.to_thread(deepgram_tts_to_wav, text_to_speak, code, api_key)
        elif provider == 'google':
            wav_path = await asyncio.to_thread(google_tts_to_wav, text_to_speak, language, code, api_key)
        else:
            raise ValueError(f"Unknown or unsupported provider: {provider}")
        
        async with aiofiles.open(wav_path, 'rb') as f:
            audio_data = await f.read()
            audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)

        if wav_path:
             background_tasks.add_task(os.remove, wav_path)

        return StreamingResponse(audio_buffer, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Greeting TTS failed: {e}")
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except: pass
        raise HTTPException(status_code=500, detail=f"Failed to generate greeting: {e}")


@app.post("/voice/demo")
@retry_with_backoff()
async def tts_demo(request: DemoRequest, background_tasks: BackgroundTasks, db=Depends(get_db)):
    firm_id = request.firmId
    provider = request.provider.lower()
    code = request.code
    language = request.language

    wav_path = None
    llm = None
    
    try:
        if firm_id: llm = get_llm(firm_id, db_conn=db)
        else: logger.warning("No firmId for demo generation.")
    except ValueError as e:
        logger.warning(f"Could not get LLM for firm {firm_id}: {e}. Using default demo text.")

    persona_id = request.persona_id
    personas = await asyncio.to_thread(_load_personas)
    persona_config = personas.get(persona_id, {})
    persona_name = persona_config.get("name", "the assistant")
    
    lang_key = (language or str(code) or "").split("-")[0].lower()
    lang_map = {"en": "English", "hi": "Hindi", "ml": "Malayalam", "te": "Telugu", "ta": "Tamil", "bn": "Bengali", "mr": "Marathi"}
    lang_name = lang_map.get(lang_key, "English")

    default_demo_sentences = {
        "en": "This is a demonstration of my voice.",
        "hi": "यह मेरी आवाज़ का प्रदर्शन है।",
    }
    text_to_speak = default_demo_sentences.get(lang_key)
    
    if not text_to_speak and llm:
        try:
            persona_context = persona_config.get("voice_prompt", "A helpful assistant.")
            generation_prompt = (
                f"You are an AI voice assistant named '{persona_name}'. "
                f"Your personality: '{persona_context[:250]}...'.\n"
                f"Generate one short sentence in {lang_name.upper()} to demonstrate your voice, reflecting your personality."
                "Output ONLY the sentence. No quotes."
            )
            text_to_speak = await llm.ainvoke(generation_prompt)
            text_to_speak = text_to_speak.content
            logger.info(f"Generated demo text for '{persona_name}' in '{lang_name}': '{text_to_speak}'")
        except Exception as e:
            logger.error(f"Demo text LLM generation failed: {e}")
            text_to_speak = "This is a sample of my voice."
    elif not text_to_speak:
         text_to_speak = default_demo_sentences.get('en')

    try:
        provider_map = {
            'google': 'GOOGLE_TTS',
            'elevenlabs': 'ELEVENLABS',
            'deepgram': 'DEEPGRAM'
        }
        api_key_provider = provider_map.get(provider, provider.upper())
        api_key = get_api_key(firm_id, api_key_provider, db)

        if not api_key:
            raise ValueError(f"{provider.capitalize()} API key is not configured for this firm.")

        if provider == 'elevenlabs':
            wav_path = await asyncio.to_thread(elevenlabs_tts_to_wav, text_to_speak, code, api_key)
        elif provider == 'deepgram':
            wav_path = await asyncio.to_thread(deepgram_tts_to_wav, text_to_speak, code, api_key)
        elif provider == 'google':
            wav_path = await asyncio.to_thread(google_tts_to_wav, text_to_speak, language, code, api_key)
        else:
            raise ValueError(f"Unknown or unsupported provider: {provider}")
        
        async with aiofiles.open(wav_path, 'rb') as f:
            audio_data = await f.read()
            audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)
        
        if wav_path:
             background_tasks.add_task(os.remove, wav_path)

        return StreamingResponse(audio_buffer, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Demo TTS failed for {code}: {e}")
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except: pass
        raise HTTPException(status_code=500, detail="Failed to generate demo speech")


@app.post("/rag/run-test")
@retry_with_backoff(retries=2)
async def run_rag_test(request: TestRequest, db=Depends(get_db)):
    """
    Runs a series of compliance and accuracy tests.
    It automatically generates questions based on the knowledge base content.
    """
    try:
        firm_id = request.firmId
        num_questions = request.num_questions
        collection_name = f"{QDRANT_COLLECTION_PREFIX}{request.owner_id}-{request.category}"
        persona_id = request.persona_id
        compliance_rules = request.compliance_rules
        
        try:
            llm = get_llm(firm_id, db_conn=db)
        except ValueError as e:
             raise HTTPException(status_code=400, detail=str(e))


        # 1. Fetch context from the vector store to seed question generation
        try:
            # Scroll is synchronous in QdrantClient
            scroll_response, _ = await asyncio.to_thread(
                qdrant.scroll,
                collection_name=collection_name,
                limit=50,
                with_payload=True,
                with_vectors=False
            )
            context_docs = [point.payload.get(CONTENT_PAYLOAD_KEY, "") for point in scroll_response if point.payload]
            kb_context = "\n---\n".join(context_docs)
            
            if len(kb_context) > MAX_QUESTION_GEN_CONTEXT:
                kb_context = kb_context[:MAX_QUESTION_GEN_CONTEXT]
            
            if not kb_context.strip():
                 raise HTTPException(status_code=404, detail="Knowledge base is empty or could not be read.")

        except Exception as e:
            logger.error(f"Failed to fetch context for question generation from {collection_name}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve knowledge base context to generate questions.")

        # 2. Generate test questions using an LLM
        question_generation_prompt_template = """
        You are a Quality Assurance AI for a RAG system. Your task is to generate a list of relevant test questions based on the provided document excerpts.
        The goal is to test if the Agent effectively "knows" the content.

        **Knowledge Base Context:**
        ---
        {kb_context}
        ---

        **Instructions:**
        1.  Create exactly {num_questions} diverse questions.
        2.  Mix of specific fact retrieval and summary-based questions.
        3.  Questions must be answerable from the text provided.
        4.  You MUST respond with a single JSON object containing one key: "questions", which is a list of the generated question strings.
        """
        
        prompt = ChatPromptTemplate.from_template(question_generation_prompt_template)
        chain = prompt | llm | JsonOutputParser()
        
        try:
            response = await chain.ainvoke({
                "kb_context": kb_context,
                "num_questions": num_questions
            })
            questions_to_run = response.get("questions", [])
            
            # SANITIZATION FIX: Ensure questions are strings to prevent crashes
            # The LLM sometimes returns objects like {"text": "question"} instead of strings.
            sanitized_questions = []
            for q in questions_to_run:
                if isinstance(q, str):
                    sanitized_questions.append(q)
                elif isinstance(q, dict) and "question" in q:
                    sanitized_questions.append(q["question"])
                elif isinstance(q, dict) and "text" in q:
                    sanitized_questions.append(q["text"])
            
            # Use sanitized list and enforce user limit
            questions_to_run = sanitized_questions[:num_questions]
            
            if not questions_to_run:
                raise ValueError("LLM returned no valid questions.")
                
            logger.info(f"Generated {len(questions_to_run)} questions for test on {collection_name}")
        except Exception as e:
            logger.error(f"Failed to generate test questions using LLM for {collection_name}: {e}")
            raise HTTPException(status_code=500, detail="The AI failed to generate test questions.")

        # 3. Run RAG and EVALUATE responses
        async def run_and_evaluate(question):
            try:
                # A. Run the RAG Chain
                initial_state = {
                    "collection_name": collection_name,
                    "question": question.strip(),
                    "chat_history": [],
                    "persona_id": persona_id,
                    "turns": 0,
                    "compliance_rules": compliance_rules,
                    "firm_id": firm_id,
                    "call_state": "initial",
                    "query_source": "text",
                }
                
                final_state = await lang_graph_app.ainvoke(initial_state)
                answer = final_state.get("response", "No response generated.")
                
                # B. Evaluate the Answer (Grounding Check)
                # IMPROVED PROMPT: Strict JSON enforcement
                eval_prompt_template = """
                You are a strict Grader AI. Rate the following RAG response on a scale of 0 to 100.
                
                **Question:** {question}
                **Agent Answer:** {answer}
                
                **Criteria:**
                - 100: Perfect, accurate, and completely grounded in common sense or likely context.
                - 75: Mostly correct but missed a minor detail.
                - 50: Partially correct or vague.
                - 0: Completely wrong, hallucinated, or "I don't know".
                
                **Output Format:**
                Provide ONLY a raw JSON object. Do not output markdown code blocks (```json).
                {{
                    "score": <integer_0_to_100>,
                    "reason": "<short_explanation>"
                }}
                """
                
                eval_prompt = ChatPromptTemplate.from_template(eval_prompt_template)
                eval_chain = eval_prompt | llm | JsonOutputParser()
                
                try:
                    eval_res = await eval_chain.ainvoke({"question": question, "answer": answer})
                    score = int(eval_res.get("score", 0))
                    reason = eval_res.get("reason", "No reason provided.")
                except Exception as e:
                    logger.warning(f"Evaluation parsing failed: {e}. Defaulting to 0.")
                    score = 0
                    reason = "Could not parse grader response."
                
                return {
                    "question": question, 
                    "answer": answer, 
                    "score": score, 
                    "reason": reason
                }
                
            except Exception as e:
                logger.error(f"Test question failed: {e}")
                return {"question": question, "answer": "Error executing test.", "score": 0, "reason": str(e)}

        results = await asyncio.gather(*(run_and_evaluate(q) for q in questions_to_run))

        # Calculate Aggregate Score
        total_score = sum(r['score'] for r in results)
        avg_score = round(total_score / len(results), 1) if results else 0

        logger.info(f"Completed test run for {collection_name}. Avg Score: {avg_score}%")
        
        return {
            "results": results,
            "overall_score": avg_score,
            "total_questions": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG test run endpoint failed")
        raise HTTPException(status_code=500, detail="A server error occurred while running the test.")

# -----------------------------------------------------------------------------
# App start
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8252)