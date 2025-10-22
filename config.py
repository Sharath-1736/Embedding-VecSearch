"""
Configuration file for ArXiv Vector Search System
Enhanced with Individual Paper Summary capabilities
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "indexes"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, INDEX_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration
ARXIV_DATASET_PATH = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
PROCESSED_DATA_PATH = DATA_DIR / "processed_papers.pkl"
SAMPLE_SIZE = None  # CHANGED: Set to None to process ALL papers (was 1000)

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Correct
# Alternative models:
# "all-mpnet-base-v2"  # More accurate but slower
# "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # Optimized for Q&A

EMBEDDING_DIMENSION = 384  # Dimension for MiniLM-L6-v2
BATCH_SIZE = 32
MAX_TEXT_LENGTH = 512

# FAISS configuration
FAISS_INDEX_TYPE = "IVF"  # Options: "Flat", "IVF", "HNSW"
N_CLUSTERS = 1000  # For IVF index
N_PROBE = 50  # Search parameter for IVF

# Search configuration
TOP_K_RESULTS = 1
SIMILARITY_THRESHOLD = 0.5

# ====== AI SUMMARY CONFIGURATION ======
# Enhanced AI configuration for individual paper summaries

# OpenAI Configuration
USE_OPENAI = True  # Set to True to enable OpenAI summaries
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ADD YOUR API KEY TO .env FILE
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4" for better quality

# Hugging Face Configuration (Fallback)
USE_HUGGINGFACE = False  # Fallback when OpenAI fails
HF_GENERATIVE_MODEL = "microsoft/DialoGPT-medium"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Individual Paper Summary Settings
ENABLE_INDIVIDUAL_SUMMARIES = True  # Enable AI summaries for each paper
INDIVIDUAL_SUMMARY_MAX_TOKENS = 150  # Max tokens per individual summary
INDIVIDUAL_SUMMARY_TEMPERATURE = 0.7  # Creativity level (0.0-1.0)

# Batch Summary Settings
ENABLE_BATCH_SUMMARIES = True  # Enable overall search result summaries
BATCH_SUMMARY_MAX_TOKENS = 300  # Max tokens for batch summaries
BATCH_SUMMARY_TEMPERATURE = 0.8

# Cost Management
MAX_SUMMARIES_PER_SEARCH = 5  # Limit summaries to control costs
ENABLE_SUMMARY_CACHING = True  # Cache summaries to avoid regeneration
SUMMARY_CACHE_DIR = DATA_DIR / "summary_cache"
SUMMARY_CACHE_DIR.mkdir(exist_ok=True)

# Summary Prompts Templates
INDIVIDUAL_SUMMARY_PROMPT = """
Summarize this research paper in 5-6 concise sentences focusing on:
1. Main contribution/finding
2. Methodology used
3. Practical implications

Title: {title}
Abstract: {abstract}

Summary:"""

BATCH_SUMMARY_PROMPT = """
Based on these {count} research papers related to "{query}", provide a brief overview highlighting:
1. Common themes and research directions
2. Key methodologies being used
3. Notable trends or gaps

Papers: {papers_info}

Overview:"""

# Error Handling
AI_SUMMARY_TIMEOUT = 30  # Seconds to wait for AI response
MAX_RETRIES = 2  # Number of retries for failed API calls
FALLBACK_TO_SIMPLE_SUMMARY = True  # Use rule-based summary if AI fails

# Text processing configuration
MIN_ABSTRACT_LENGTH = 50
MAX_ABSTRACT_LENGTH = 2000
LANGUAGE_FILTER = "en"  # Only English papers

# Categories to include (ArXiv categories)
# CHANGED: Empty list means NO category filtering - include ALL categories
INCLUDED_CATEGORIES = []

# Original category filter (for reference if you want to re-enable later):
# INCLUDED_CATEGORIES = [
#     "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO",  # Computer Science
#     "stat.ML",  # Statistics - Machine Learning
#     "math.ST", "math.PR", "math.OC",  # Mathematics
#     "physics.data-an", "physics.comp-ph",  # Physics
#     "q-bio.QM", "q-bio.GN"  # Quantitative Biology
# ]

# Streamlit configuration
APP_TITLE = "Re-Search: AI-Powered Paper Finder"
APP_DESCRIPTION = "Search through thousands of research papers using semantic similarity and get AI-powered summaries"
PAGE_SIZE = 20

# Display Configuration
SHOW_INDIVIDUAL_SUMMARIES = True  # Show AI summaries in results
SHOW_ABSTRACT_BY_DEFAULT = False  # Hide full abstracts when summaries available
SUMMARY_DISPLAY_MODE = "expandable"  # Options: "inline", "expandable", "tooltip"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Validation function
def validate_ai_config():
    """Validate AI configuration settings"""
    issues = []
    
    if USE_OPENAI and not OPENAI_API_KEY:
        issues.append("OpenAI is enabled but OPENAI_API_KEY is not set. Please add it to your .env file.")
    
    if USE_HUGGINGFACE and not HF_API_TOKEN:
        issues.append("Hugging Face is enabled but HF_API_TOKEN is not set. This may limit model access.")
    
    if INDIVIDUAL_SUMMARY_MAX_TOKENS > 500:
        issues.append("INDIVIDUAL_SUMMARY_MAX_TOKENS is very high. This may increase costs significantly.")
    
    if MAX_SUMMARIES_PER_SEARCH > 20:
        issues.append("MAX_SUMMARIES_PER_SEARCH is high. Consider reducing to control API costs.")
    
    return issues

# Cost estimation (rough)
def estimate_cost_per_search(num_results: int = TOP_K_RESULTS):
    """Estimate OpenAI API cost per search"""
    if not USE_OPENAI or not ENABLE_INDIVIDUAL_SUMMARIES:
        return 0.0
    
    # GPT-3.5-turbo pricing: ~$0.002 per 1K tokens
    tokens_per_summary = INDIVIDUAL_SUMMARY_MAX_TOKENS + 200  # Input + output tokens
    total_tokens = min(num_results, MAX_SUMMARIES_PER_SEARCH) * tokens_per_summary
    
    if ENABLE_BATCH_SUMMARIES:
        total_tokens += BATCH_SUMMARY_MAX_TOKENS + 500
    
    cost_per_1k_tokens = 0.002  # USD
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    return estimated_cost