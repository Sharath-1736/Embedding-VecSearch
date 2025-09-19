"""
Search engine implementation combining vector search with generative AI
Provides semantic search with AI-powered result summarization, individual paper summaries, and PDF downloads
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import re
import os
import requests
import hashlib
import json
import time
from datetime import datetime

import openai
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

from embedding_generator import EmbeddingGenerator
from vector_store import FAISSVectorStore
from utils import extract_arxiv_id, validate_arxiv_id, download_pdf, format_file_size
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class SearchResult:
    """
    Represents a single search result with PDF download capabilities and AI summary
    """
    
    def __init__(self, document_id: str, title: str, abstract: str, authors: List[str], 
                 categories: List[str], similarity_score: float, update_date: datetime = None,
                 ai_summary: str = None):
        self.document_id = document_id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.categories = categories
        self.similarity_score = similarity_score
        self.update_date = update_date
        self.ai_summary = ai_summary  # AI-generated summary
        
        # Extract ArXiv ID and generate URLs
        self.arxiv_id = extract_arxiv_id(document_id)
        self.abstract_url = f"https://arxiv.org/abs/{self.arxiv_id}" if self.arxiv_id else None
        self.pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf" if self.arxiv_id else None
        self.pdf_available = self._check_pdf_availability() if self.pdf_url else False
    
    def _check_pdf_availability(self) -> bool:
        """Check if PDF is available for download"""
        if not self.pdf_url:
            return False
        
        try:
            response = requests.head(self.pdf_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'document_id': self.document_id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'categories': self.categories,
            'similarity_score': self.similarity_score,
            'update_date': self.update_date.isoformat() if self.update_date else None,
            'arxiv_id': self.arxiv_id,
            'abstract_url': self.abstract_url,
            'pdf_url': self.pdf_url,
            'pdf_available': self.pdf_available,
            'ai_summary': self.ai_summary  # Include AI summary
        }
    
    def get_summary(self, max_length: int = 200) -> str:
        """Get AI summary if available, otherwise truncated abstract"""
        if self.ai_summary:
            return self.ai_summary
        
        if len(self.abstract) <= max_length:
            return self.abstract
        
        # Try to cut at sentence boundary
        truncated = self.abstract[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.7:  # If we have a reasonable sentence break
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def download_pdf(self, download_dir: str = "downloads") -> Dict[str, any]:
        """
        Download PDF for this paper
        
        Args:
            download_dir: Directory to save the PDF
            
        Returns:
            Dictionary with download status and file path
        """
        if not self.arxiv_id or not self.pdf_url:
            return {
                'success': False,
                'error': 'No ArXiv ID or PDF URL available',
                'file_path': None
            }
        
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
        
        file_path = os.path.join(download_dir, f"{self.arxiv_id}.pdf")
        
        try:
            success = download_pdf(self.pdf_url, file_path)
            if success:
                return {
                    'success': True,
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'arxiv_id': self.arxiv_id,
                    'title': self.title
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to download PDF',
                    'file_path': None
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Download error: {str(e)}',
                'file_path': None
            }

class SummaryCache:
    """
    Cache for AI-generated summaries to avoid redundant API calls
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or config.SUMMARY_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "summaries.json")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load summary cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save summary cache: {e}")
    
    def _get_cache_key(self, title: str, abstract: str) -> str:
        """Generate cache key for a paper"""
        content = f"{title}|{abstract[:500]}"  # Use first 500 chars of abstract
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, title: str, abstract: str) -> Optional[str]:
        """Get cached summary"""
        cache_key = self._get_cache_key(title, abstract)
        return self.cache.get(cache_key)
    
    def set(self, title: str, abstract: str, summary: str):
        """Cache a summary"""
        cache_key = self._get_cache_key(title, abstract)
        self.cache[cache_key] = {
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'title': title[:100]  # Store partial title for debugging
        }
        self._save_cache()

class GenerativeAIAssistant:
    """
    Handles AI-powered text generation for search result enhancement and individual summaries
    """
    
    def __init__(self, use_openai: bool = None, use_huggingface: bool = None):
        self.use_openai = use_openai if use_openai is not None else config.USE_OPENAI
        self.use_huggingface = use_huggingface if use_huggingface is not None else config.USE_HUGGINGFACE
        
        self.openai_client = None
        self.hf_generator = None
        self.summary_cache = SummaryCache() if config.ENABLE_SUMMARY_CACHING else None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models based on configuration"""
        if self.use_openai and config.OPENAI_API_KEY:
            try:
                openai.api_key = config.OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized for individual summaries")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.use_openai = False
        
        if self.use_huggingface:
            try:
                logger.info("Initializing Hugging Face model...")
                self.hf_generator = pipeline(
                    "text-generation",
                    model=config.HF_GENERATIVE_MODEL,
                    tokenizer=config.HF_GENERATIVE_MODEL,
                    device=0 if config.HF_GENERATIVE_MODEL != "gpt2" else -1  # Use GPU if available
                )
                logger.info("Hugging Face model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face model: {e}")
                self.use_huggingface = False
    
    def generate_individual_summary(self, title: str, abstract: str) -> Optional[str]:
        """
        Generate AI summary for an individual paper
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            AI-generated summary or None if failed
        """
        if not config.ENABLE_INDIVIDUAL_SUMMARIES:
            return None
        
        # Check cache first
        if self.summary_cache:
            cached_summary = self.summary_cache.get(title, abstract)
            if cached_summary:
                logger.debug(f"Using cached summary for: {title[:50]}...")
                return cached_summary
        
        # Generate new summary
        prompt = config.INDIVIDUAL_SUMMARY_PROMPT.format(
            title=title,
            abstract=abstract[:1000]  # Limit abstract length for API efficiency
        )
        
        summary = None
        if self.use_openai and self.openai_client:
            summary = self._generate_individual_with_openai(prompt)
        elif self.use_huggingface and self.hf_generator:
            summary = self._generate_individual_with_huggingface(prompt)
        
        if not summary and config.FALLBACK_TO_SIMPLE_SUMMARY:
            summary = self._generate_simple_individual_summary(title, abstract)
        
        # Cache the result
        if summary and self.summary_cache:
            self.summary_cache.set(title, abstract, summary)
        
        return summary
    
    def _generate_individual_with_openai(self, prompt: str) -> Optional[str]:
        """Generate individual summary using OpenAI"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research assistant that creates concise, informative summaries of academic papers. Focus on key contributions, methods, and implications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.INDIVIDUAL_SUMMARY_MAX_TOKENS,
                temperature=config.INDIVIDUAL_SUMMARY_TEMPERATURE,
                timeout=config.AI_SUMMARY_TIMEOUT
            )
            
            summary = response.choices[0].message.content.strip()
            logger.debug("Generated individual summary with OpenAI")
            return summary
            
        except Exception as e:
            logger.error(f"OpenAI individual summary failed: {e}")
            return None
    
    def _generate_individual_with_huggingface(self, prompt: str) -> Optional[str]:
        """Generate individual summary using Hugging Face"""
        try:
            # Limit prompt length
            max_prompt_length = 400
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            response = self.hf_generator(
                prompt,
                max_length=len(prompt.split()) + config.INDIVIDUAL_SUMMARY_MAX_TOKENS // 4,
                num_return_sequences=1,
                temperature=config.INDIVIDUAL_SUMMARY_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.hf_generator.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            # Extract only the new generated part
            new_text = generated_text[len(prompt):].strip()
            
            logger.debug("Generated individual summary with Hugging Face")
            return new_text if new_text else None
            
        except Exception as e:
            logger.error(f"Hugging Face individual summary failed: {e}")
            return None
    
    def _generate_simple_individual_summary(self, title: str, abstract: str) -> str:
        """Generate simple rule-based summary when AI fails"""
        # Extract first few sentences as summary
        sentences = abstract.split('.')
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences[:3]:  # Max 3 sentences
            sentence = sentence.strip()
            if sentence and char_count + len(sentence) < 200:
                summary_sentences.append(sentence)
                char_count += len(sentence)
            else:
                break
        
        if summary_sentences:
            return '. '.join(summary_sentences) + '.'
        else:
            return abstract[:150] + "..." if len(abstract) > 150 else abstract
    
    def summarize_results(self, results: List[SearchResult], query: str) -> str:
        """
        Generate a summary of search results using AI
        """
        if not results:
            return "No relevant papers found for your query."
        
        # Prepare context with individual summaries if available
        context = f"Query: {query}\n\n"
        context += f"Found {len(results)} relevant research papers:\n\n"
        
        papers_info = []
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 for summary
            paper_summary = result.ai_summary or result.get_summary(100)
            papers_info.append({
                'title': result.title,
                'summary': paper_summary,
                'categories': result.categories[:2],
                'score': result.similarity_score
            })
            
            context += f"{i}. {result.title}\n"
            context += f"   Categories: {', '.join(result.categories[:3])}\n"
            context += f"   Score: {result.similarity_score:.3f}\n"
            context += f"   Summary: {paper_summary}\n\n"
        
        if config.ENABLE_BATCH_SUMMARIES:
            prompt = config.BATCH_SUMMARY_PROMPT.format(
                count=len(results),
                query=query,
                papers_info=str(papers_info)[:1500]  # Limit context length
            )
            
            if self.use_openai and self.openai_client:
                return self._generate_batch_with_openai(prompt)
            elif self.use_huggingface and self.hf_generator:
                return self._generate_batch_with_huggingface(prompt)
        
        return self._generate_simple_batch_summary(results, query)
    
    def _generate_batch_with_openai(self, prompt: str) -> str:
        """Generate batch summary using OpenAI"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research assistant specializing in academic paper analysis and trend identification."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.BATCH_SUMMARY_MAX_TOKENS,
                temperature=config.BATCH_SUMMARY_TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI batch summary failed: {e}")
            return "AI summary generation failed. Please review the individual results below."
    
    def _generate_batch_with_huggingface(self, prompt: str) -> str:
        """Generate batch summary using Hugging Face"""
        try:
            # Limit prompt length for efficiency
            max_prompt_length = 800
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            response = self.hf_generator(
                prompt,
                max_length=len(prompt.split()) + config.BATCH_SUMMARY_MAX_TOKENS // 4,
                num_return_sequences=1,
                temperature=config.BATCH_SUMMARY_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.hf_generator.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            # Extract only the new generated part
            new_text = generated_text[len(prompt):].strip()
            
            return new_text if new_text else "AI summary generation produced no output."
            
        except Exception as e:
            logger.error(f"Hugging Face batch summary failed: {e}")
            return "AI summary generation failed. Please review the individual results below."
    
    def _generate_simple_batch_summary(self, results: List[SearchResult], query: str) -> str:
        """Generate a simple rule-based summary when AI models are unavailable"""
        if not results:
            return "No relevant papers found."
        
        # Analyze categories
        all_categories = []
        for result in results[:10]:
            all_categories.extend(result.categories)
        
        from collections import Counter
        top_categories = Counter(all_categories).most_common(5)
        
        # Analyze temporal distribution
        recent_papers = sum(1 for r in results if r.update_date and 
                           r.update_date.year >= datetime.now().year - 2)
        
        # Count available PDFs and summaries
        available_pdfs = sum(1 for r in results if r.pdf_available)
        with_summaries = sum(1 for r in results if r.ai_summary)
        
        summary = f"Found {len(results)} papers related to '{query}'. "
        summary += f"{available_pdfs} papers have downloadable PDFs available. "
        summary += f"{with_summaries} papers include AI-generated summaries. "
        
        if top_categories:
            cat_text = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
            summary += f"Main research areas include: {cat_text}. "
        
        if recent_papers:
            summary += f"{recent_papers} papers are from the last 2 years, indicating active research. "
        
        summary += f"Top result: '{results[0].title}' with similarity score {results[0].similarity_score:.3f}."
        
        return summary

class ArXivSearchEngine:
    """
    Main search engine class combining vector search with AI assistance, individual summaries, and PDF downloads
    """
    
    def __init__(self, data_df: pd.DataFrame = None):
        self.data_df = data_df
        self.embedding_generator = None
        self.vector_store = None
        self.ai_assistant = GenerativeAIAssistant()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding generator and vector store"""
        logger.info("Initializing search engine components...")
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize vector store
        embedding_dim = self.embedding_generator.embedding_dim
        self.vector_store = FAISSVectorStore(embedding_dim)
        
        # Try to load existing index
        try:
            self.vector_store.load_index()
            logger.info("Loaded existing FAISS index")
        except FileNotFoundError:
            if self.data_df is not None:
                logger.info("No existing index found. Building new index...")
                self._build_index()
            else:
                logger.warning("No existing index and no data provided. Search functionality limited.")
    
    def _build_index(self):
        """Build vector index from data"""
        if self.data_df is None:
            raise ValueError("No data available for building index")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings_from_dataframe(
            self.data_df, 'combined_text'
        )
        
        # Build vector store
        document_ids = self.data_df['id'].tolist()
        self.vector_store.build_index(embeddings, document_ids)
        self.vector_store.save_index()
        
        logger.info("Index built and saved successfully")
    
    def search(self, query: str, top_k: int = None, min_similarity: float = None, 
               category_filter: List[str] = None, date_filter: Dict = None,
               generate_summaries: bool = None) -> List[SearchResult]:
        """
        Perform semantic search with optional filters and AI summaries
        
        Args:
            generate_summaries: Whether to generate AI summaries for results
        """
        top_k = top_k or config.TOP_K_RESULTS
        min_similarity = min_similarity or config.SIMILARITY_THRESHOLD
        generate_summaries = generate_summaries if generate_summaries is not None else config.ENABLE_INDIVIDUAL_SUMMARIES
        
        if not query.strip():
            return []
        
        logger.info(f"Searching for: '{query}' (summaries: {generate_summaries})")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        # Perform vector search
        document_ids, similarities = self.vector_store.search(query_embedding, top_k * 2)  # Get more for filtering
        
        # Convert to SearchResult objects
        results = []
        summary_count = 0
        
        for doc_id, similarity in zip(document_ids, similarities):
            if similarity < min_similarity:
                continue
            
            # Find document in dataframe
            doc_row = self.data_df[self.data_df['id'] == doc_id]
            
            if doc_row.empty:
                continue
            
            doc_data = doc_row.iloc[0]
            
            # Apply filters
            if category_filter:
                doc_categories = doc_data['categories']
                if not any(cat in doc_categories for cat in category_filter):
                    continue
            
            if date_filter:
                doc_date = doc_data['update_date']
                if doc_date and not self._passes_date_filter(doc_date, date_filter):
                    continue
            
            # Generate AI summary if enabled and within limits
            ai_summary = None
            if (generate_summaries and 
                summary_count < config.MAX_SUMMARIES_PER_SEARCH and
                self.ai_assistant):
                
                logger.info(f"Generating summary for paper: {doc_data['title'][:50]}...")
                ai_summary = self.ai_assistant.generate_individual_summary(
                    doc_data['title'], 
                    doc_data['abstract']
                )
                if ai_summary:
                    summary_count += 1
            
            # Create search result
            result = SearchResult(
                document_id=doc_id,
                title=doc_data['title'],
                abstract=doc_data['abstract'],
                authors=doc_data['authors'],
                categories=doc_data['categories'],
                similarity_score=similarity,
                update_date=doc_data['update_date'],
                ai_summary=ai_summary
            )
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Found {len(results)} results after filtering ({summary_count} with AI summaries)")
        return results
    
    def search_with_ai_summary(self, query: str, **kwargs) -> Dict:
        """
        Perform search and generate AI summary with individual paper summaries
        """
        results = self.search(query, **kwargs)
        
        ai_summary = ""
        if results:
            ai_summary = self.ai_assistant.summarize_results(results, query)
        
        # Count downloadable PDFs and summaries
        downloadable_count = sum(1 for r in results if r.pdf_available)
        summary_count = sum(1 for r in results if r.ai_summary)
        
        # Estimate cost if using OpenAI
        estimated_cost = 0.0
        if config.USE_OPENAI and config.ENABLE_INDIVIDUAL_SUMMARIES:
            estimated_cost = config.estimate_cost_per_search(len(results))
        
        return {
            'query': query,
            'results': [result.to_dict() for result in results],
            'ai_summary': ai_summary,
            'total_results': len(results),
            'downloadable_pdfs': downloadable_count,
            'ai_summaries_count': summary_count,
            'estimated_cost': estimated_cost,
            'search_timestamp': datetime.now().isoformat()
        }
    
    # ... (rest of the methods remain the same as before)
    def download_paper_pdf(self, arxiv_id: str, download_dir: str = "downloads") -> Dict[str, any]:
        """Download PDF for a specific paper by ArXiv ID"""
        if not validate_arxiv_id(arxiv_id):
            return {
                'success': False,
                'error': 'Invalid ArXiv ID format',
                'file_path': None
            }
        
        os.makedirs(download_dir, exist_ok=True)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        file_path = os.path.join(download_dir, f"{arxiv_id}.pdf")
        
        try:
            success = download_pdf(pdf_url, file_path)
            if success:
                return {
                    'success': True,
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'arxiv_id': arxiv_id
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to download PDF',
                    'file_path': None
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Download error: {str(e)}',
                'file_path': None
            }
    
    def _passes_date_filter(self, doc_date: datetime, date_filter: Dict) -> bool:
        """Check if document passes date filter"""
        if 'start_date' in date_filter and doc_date < date_filter['start_date']:
            return False
        
        if 'end_date' in date_filter and doc_date > date_filter['end_date']:
            return False
        
        if 'years_back' in date_filter:
            cutoff_date = datetime.now() - pd.DateOffset(years=date_filter['years_back'])
            if doc_date < cutoff_date:
                return False
        
        return True
    
    def get_similar_papers(self, paper_id: str, top_k: int = 5, generate_summaries: bool = True) -> List[SearchResult]:
        """Find papers similar to a given paper with optional AI summaries"""
        paper_row = self.data_df[self.data_df['id'] == paper_id]
        
        if paper_row.empty:
            logger.warning(f"Paper with ID {paper_id} not found")
            return []
        
        paper_data = paper_row.iloc[0]
        query_text = paper_data['combined_text']
        
        # Search for similar papers
        results = self.search(query_text, top_k=top_k + 1, generate_summaries=generate_summaries)
        
        # Remove the original paper from results
        filtered_results = [r for r in results if r.document_id != paper_id]
        
        return filtered_results[:top_k]

def main():
    """Main function for testing enhanced search engine with individual summaries"""
    from data_processor import ArXivDataProcessor
    
    # Load processed data
    processor = ArXivDataProcessor()
    df = processor.process_dataset()
    
    # Take a sample for testing
    sample_df = df.head(1000)  # Smaller sample for testing summaries
    
    # Initialize search engine
    search_engine = ArXivSearchEngine(sample_df)
    
    # Test query with individual summaries
    query = "machine learning transformers"
    
    print(f"\nTesting enhanced search with AI summaries")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Search with AI summaries
    search_results = search_engine.search_with_ai_summary(query, top_k=3)
    
    print(f"\nOverall AI Summary:")
    print(search_results['ai_summary'])
    
    print(f"\nIndividual Results (with AI Summaries):")
    print(f"Total: {search_results['total_results']}")
    print(f"With AI Summaries: {search_results['ai_summaries_count']}")
    print(f"Estimated Cost: ${search_results['estimated_cost']:.4f}")
    
    for i, result_dict in enumerate(search_results['results'], 1):
        print(f"\n{i}. {result_dict['title']}")
        print(f"   Similarity: {result_dict['similarity_score']:.4f}")
        print(f"   ArXiv ID: {result_dict['arxiv_id']}")
        
        if result_dict['ai_summary']:
            print(f"   AI Summary: {result_dict['ai_summary']}")
        else:
            print(f"   Abstract: {result_dict['abstract'][:150]}...")

if __name__ == "__main__":
    main()