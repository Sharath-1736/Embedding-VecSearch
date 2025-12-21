"""
Configuration file for ArXiv Vector Search System
Enhanced with Advanced Semantic Search and AI Summary capabilities
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

# Processing Options:
# None = Process all papers (recommended for 4GB+ RAM)
# 50000 = Process first 50,000 papers
# 10000 = Conservative batch size for limited RAM
SAMPLE_SIZE = None  # Set to None to process ALL papers

# ====== ENHANCED EMBEDDING CONFIGURATION ======

# RECOMMENDED: Use allenai/specter for scientific papers (best quality)
# This model is specifically trained on scientific literature
EMBEDDING_MODEL = "allenai/specter"  # 768 dims, best for scientific papers

# Alternative embedding models:
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dims, excellent general-purpose
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, fast and lightweight
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"  # 384 dims, better quality than L6

# Embedding dimensions (auto-detected by EmbeddingGenerator)
# SPECTER: 768 dims
# all-mpnet-base-v2: 768 dims
# all-MiniLM-L6-v2: 384 dims
EMBEDDING_DIMENSION = 768  # Update based on your model choice

BATCH_SIZE = 32  # Increase to 64 if you have more RAM
MAX_TEXT_LENGTH = 512

# ====== ENHANCED SEARCH CONFIGURATION ======

# Enable hybrid title+abstract embeddings (RECOMMENDED)
# This weights title (40%) and abstract (60%) separately for better semantic matching
USE_HYBRID_EMBEDDINGS = True

# Enable cross-encoder re-ranking (HIGHLY RECOMMENDED)
# This dramatically improves relevance but adds ~100-200ms per search
# The quality improvement is typically 30-50%!
USE_RERANKING = True

# Cross-encoder model for re-ranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Number of candidates to retrieve for re-ranking
# Higher = better quality but slower
# Lower = faster but may miss relevant papers
RERANK_TOP_K = 50  # Retrieve top 50 candidates, then re-rank to final top_k

# Enable query expansion (RECOMMENDED)
# Automatically expands abbreviations like "ml" -> "machine learning"
USE_QUERY_EXPANSION = True

# Query expansion weight (how much to trust expanded query)
QUERY_EXPANSION_WEIGHT = 0.3  # 70% original, 30% expanded

# FAISS configuration
FAISS_INDEX_TYPE = "IVF"  # Options: "Flat", "IVF", "HNSW"

# For 50,000 papers: sqrt(50000) ≈ 224, use 256-512 clusters
# For 100,000 papers: use 512-1000 clusters
# For 1,000,000+ papers: use 4096-8192 clusters
N_CLUSTERS = 256  # Optimized for ~50K papers
N_PROBE = 32  # Optimized for faster search

# Search configuration
TOP_K_RESULTS = 10  # Default number of search results (increased from 1)

# Minimum similarity threshold (lowered due to better re-ranking)
# With re-ranking, we can use a lower threshold since the cross-encoder
# will filter out irrelevant results
SIMILARITY_THRESHOLD = 0.3  # Lowered from 0.5 for better recall

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

# ====== CACHING SETTINGS ======

# Enable embedding caching (RECOMMENDED)
ENABLE_EMBEDDING_CACHE = True

# Cache directory
EMBEDDING_CACHE_DIR = INDEX_DIR / "cache"
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)

# ====== TEXT PROCESSING CONFIGURATION ======

MIN_ABSTRACT_LENGTH = 50
MAX_ABSTRACT_LENGTH = 2000
LANGUAGE_FILTER = "en"  # Only English papers

# Categories to include (ArXiv categories)
# Empty list means NO category filtering - include ALL categories
INCLUDED_CATEGORIES = []

# Original category filter (for reference if you want to re-enable later):
# INCLUDED_CATEGORIES = [
#     "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO",  # Computer Science
#     "stat.ML",  # Statistics - Machine Learning
#     "math.ST", "math.PR", "math.OC",  # Mathematics
#     "physics.data-an", "physics.comp-ph",  # Physics
#     "q-bio.QM", "q-bio.GN"  # Quantitative Biology
# ]

# ====== STREAMLIT CONFIGURATION ======

APP_TITLE = "Re-Search: AI-Powered Paper Finder"
APP_DESCRIPTION = "Search through thousands of research papers using semantic similarity and get AI-powered summaries"
PAGE_SIZE = 20

# Display Configuration
SHOW_INDIVIDUAL_SUMMARIES = True  # Show AI summaries in results
SHOW_ABSTRACT_BY_DEFAULT = False  # Hide full abstracts when summaries available
SUMMARY_DISPLAY_MODE = "expandable"  # Options: "inline", "expandable", "tooltip"

# ====== LOGGING CONFIGURATION ======

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Enable detailed search logging
ENABLE_SEARCH_LOGGING = True

# Log search quality metrics
LOG_SIMILARITY_SCORES = True

# Save search history for analysis
SAVE_SEARCH_HISTORY = False  # Set to True to save all searches
SEARCH_HISTORY_FILE = DATA_DIR / "logs" / "search_history.json"

# ====== FEATURE FLAGS ======

# Enable/disable specific features
FEATURES = {
    'query_expansion': USE_QUERY_EXPANSION,
    'reranking': USE_RERANKING,
    'hybrid_embeddings': USE_HYBRID_EMBEDDINGS,
    'ai_summaries': ENABLE_INDIVIDUAL_SUMMARIES,
    'pdf_download': True,
    'batch_download': True,
    'export': True,
    'similar_papers': True,
    'category_filter': True,
    'date_filter': True,
}

# ====== PERFORMANCE SETTINGS ======

# Device for computations (auto-detected)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== VALIDATION & COST ESTIMATION ======

def validate_ai_config():
    """Validate AI configuration settings"""
    issues = []
    
    if USE_OPENAI and not OPENAI_API_KEY:
        issues.append("⚠️  OpenAI is enabled but OPENAI_API_KEY is not set. Please add it to your .env file.")
    
    if USE_HUGGINGFACE and not HF_API_TOKEN:
        issues.append("⚠️  Hugging Face is enabled but HF_API_TOKEN is not set. This may limit model access.")
    
    if INDIVIDUAL_SUMMARY_MAX_TOKENS > 500:
        issues.append("⚠️  INDIVIDUAL_SUMMARY_MAX_TOKENS is very high. This may increase costs significantly.")
    
    if MAX_SUMMARIES_PER_SEARCH > 20:
        issues.append("⚠️  MAX_SUMMARIES_PER_SEARCH is high. Consider reducing to control API costs.")
    
    if USE_RERANKING:
        try:
            from sentence_transformers import CrossEncoder
            issues.append("✅ Cross-encoder re-ranking available")
        except ImportError:
            issues.append("⚠️  Cross-encoder re-ranking enabled but sentence-transformers not installed properly")
    
    return issues


def estimate_cost_per_search(num_results: int = TOP_K_RESULTS):
    """
    Estimate OpenAI API cost per search
    Note: Re-ranking is FREE (runs locally with sentence-transformers)
    """
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


def print_config_summary():
    """Print configuration summary"""
    print("="*80)
    print("ArXiv Search System - Configuration Summary")
    print("="*80)
    print(f"\n📊 Dataset:")
    print(f"   Sample size: {SAMPLE_SIZE or 'ALL papers'}")
    print(f"   Categories: {len(INCLUDED_CATEGORIES) if INCLUDED_CATEGORIES else 'ALL'}")
    
    print(f"\n🔍 Embedding Model:")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Dimension: {EMBEDDING_DIMENSION}")
    print(f"   Device: {DEVICE}")
    
    print(f"\n⚡ Enhanced Features:")
    print(f"   {'✅' if USE_HYBRID_EMBEDDINGS else '❌'} Hybrid Embeddings (title+abstract)")
    print(f"   {'✅' if USE_RERANKING else '❌'} Cross-Encoder Re-ranking")
    print(f"   {'✅' if USE_QUERY_EXPANSION else '❌'} Query Expansion")
    print(f"   {'✅' if ENABLE_INDIVIDUAL_SUMMARIES else '❌'} AI Summaries")
    
    print(f"\n🎯 Search Settings:")
    print(f"   Top K results: {TOP_K_RESULTS}")
    print(f"   Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Re-rank candidates: {RERANK_TOP_K if USE_RERANKING else 'N/A'}")
    
    if USE_OPENAI and ENABLE_INDIVIDUAL_SUMMARIES:
        cost = estimate_cost_per_search()
        print(f"\n💰 Estimated Cost:")
        print(f"   Per search: ${cost:.4f}")
        print(f"   Per 100 searches: ${cost * 100:.2f}")
    
    print(f"\n⚠️  Configuration Issues:")
    issues = validate_ai_config()
    for issue in issues:
        print(f"   {issue}")
    
    print("\n" + "="*80 + "\n")


# ====== MODEL COMPARISON INFO ======

MODEL_INFO = {
    "allenai/specter": {
        "dims": 768,
        "speed": "medium",
        "quality": "★★★★★",
        "best_for": "Scientific papers",
        "description": "Trained specifically on scientific literature"
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dims": 768,
        "speed": "medium",
        "quality": "★★★★☆",
        "best_for": "General purpose",
        "description": "Excellent general-purpose model"
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dims": 384,
        "speed": "fast",
        "quality": "★★★☆☆",
        "best_for": "Speed/testing",
        "description": "Fast and lightweight"
    },
    "sentence-transformers/all-MiniLM-L12-v2": {
        "dims": 384,
        "speed": "medium-fast",
        "quality": "★★★★☆",
        "best_for": "Balance",
        "description": "Better quality than L6, still fast"
    }
}


def get_model_info(model_name: str = None):
    """Get information about embedding model"""
    model_name = model_name or EMBEDDING_MODEL
    return MODEL_INFO.get(model_name, {
        "dims": "unknown",
        "speed": "unknown",
        "quality": "unknown",
        "best_for": "unknown",
        "description": "Custom model"
    })


# ====== QUICK START GUIDE ======

QUICK_START = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ENHANCED SEARCH - QUICK START                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. Install Dependencies:
   pip install sentence-transformers transformers torch faiss-cpu

2. Set OpenAI API Key (optional, for AI summaries):
   Create a .env file with: OPENAI_API_KEY=your_key_here

3. Process Data:
   python main.py process

4. Build Enhanced Index:
   python main.py build-index

5. Run Search:
   python main.py search "your query"
   
   Or launch web interface:
   streamlit run app.py

6. Features Enabled:
   ✅ Hybrid embeddings (better semantic matching)
   ✅ Cross-encoder re-ranking (30-50% better relevance)
   ✅ Query expansion (better recall)
   ✅ AI summaries (if OpenAI key provided)

For more info: Check README.md or run config.print_config_summary()
"""


if __name__ == "__main__":
    # Print configuration when run directly
    print_config_summary()
    print(QUICK_START)