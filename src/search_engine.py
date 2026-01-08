"""
Enhanced Search Engine with Advanced Semantic Matching
Features:
- Cross-encoder re-ranking for improved accuracy
- Query expansion and enhancement
- Hybrid title+abstract embeddings
- Proper similarity normalization
- Multi-stage retrieval pipeline
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
    Represents a single search result with normalized similarity scores
    """
    
    def __init__(self, document_id: str, title: str, abstract: str, authors: List[str], 
                 categories: List[str], similarity_score: float, update_date: datetime = None,
                 ai_summary: str = None, rank: int = None):
        self.document_id = document_id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.categories = categories
        self.similarity_score = self._normalize_score(similarity_score)
        self.update_date = update_date
        self.ai_summary = ai_summary
        self.rank = rank
        
        # Extract ArXiv ID and generate URLs
        self.arxiv_id = extract_arxiv_id(document_id)
        self.abstract_url = f"https://arxiv.org/abs/{self.arxiv_id}" if self.arxiv_id else None
        self.pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf" if self.arxiv_id else None
        self.pdf_available = self._check_pdf_availability() if self.pdf_url else False
    
    def _normalize_score(self, score: float) -> float:
        """Normalize similarity score to 0-1 range"""
        normalized = max(0.0, min(score, 1.0))
        return float(normalized)
    
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
            'ai_summary': self.ai_summary,
            'rank': self.rank
        }
    
    def get_summary(self, max_length: int = 200) -> str:
        """Get AI summary if available, otherwise truncated abstract"""
        if self.ai_summary:
            return self.ai_summary
        
        if len(self.abstract) <= max_length:
            return self.abstract
        
        truncated = self.abstract[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.7:
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def download_pdf(self, download_dir: str = "downloads") -> Dict[str, any]:
        """Download PDF for this paper"""
        if not self.arxiv_id or not self.pdf_url:
            return {
                'success': False,
                'error': 'No ArXiv ID or PDF URL available',
                'file_path': None
            }
        
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
    """Cache for AI-generated summaries"""
    
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
        content = f"{title}|{abstract[:500]}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, title: str, abstract: str) -> Optional[str]:
        """Get cached summary"""
        cache_key = self._get_cache_key(title, abstract)
        cached = self.cache.get(cache_key)
        return cached['summary'] if cached else None
    
    def set(self, title: str, abstract: str, summary: str):
        """Cache a summary"""
        cache_key = self._get_cache_key(title, abstract)
        self.cache[cache_key] = {
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'title': title[:100]
        }
        self._save_cache()


class GenerativeAIAssistant:
    """Handles AI-powered text generation"""
    
    def __init__(self, use_openai: bool = None, use_huggingface: bool = None):
        self.use_openai = use_openai if use_openai is not None else config.USE_OPENAI
        self.use_huggingface = use_huggingface if use_huggingface is not None else config.USE_HUGGINGFACE
        
        self.openai_client = None
        self.hf_generator = None
        self.summary_cache = SummaryCache() if config.ENABLE_SUMMARY_CACHING else None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models"""
        if self.use_openai and config.OPENAI_API_KEY:
            try:
                openai.api_key = config.OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized")
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
                    device=-1
                )
                logger.info("Hugging Face model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face: {e}")
                self.use_huggingface = False
    
    def generate_individual_summary(self, title: str, abstract: str) -> Optional[str]:
        """Generate AI summary for an individual paper"""
        if not config.ENABLE_INDIVIDUAL_SUMMARIES:
            return None
        
        # Check cache
        if self.summary_cache:
            cached = self.summary_cache.get(title, abstract)
            if cached:
                logger.debug(f"Using cached summary")
                return cached
        
        # Generate new summary
        prompt = config.INDIVIDUAL_SUMMARY_PROMPT.format(
            title=title,
            abstract=abstract[:1000]
        )
        
        summary = None
        if self.use_openai and self.openai_client:
            summary = self._generate_individual_with_openai(prompt)
        elif self.use_huggingface and self.hf_generator:
            summary = self._generate_individual_with_huggingface(prompt)
        
        if not summary and config.FALLBACK_TO_SIMPLE_SUMMARY:
            summary = self._generate_simple_individual_summary(title, abstract)
        
        # Cache result
        if summary and self.summary_cache:
            self.summary_cache.set(title, abstract, summary)
        
        return summary
    
    def _generate_individual_with_openai(self, prompt: str) -> Optional[str]:
        """Generate summary using OpenAI"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research assistant that creates concise summaries of academic papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.INDIVIDUAL_SUMMARY_MAX_TOKENS,
                temperature=config.INDIVIDUAL_SUMMARY_TEMPERATURE,
                timeout=config.AI_SUMMARY_TIMEOUT
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI summary failed: {e}")
            return None
    
    def _generate_individual_with_huggingface(self, prompt: str) -> Optional[str]:
        """Generate summary using Hugging Face"""
        try:
            if len(prompt) > 400:
                prompt = prompt[:400] + "..."
            
            response = self.hf_generator(
                prompt,
                max_length=len(prompt.split()) + config.INDIVIDUAL_SUMMARY_MAX_TOKENS // 4,
                num_return_sequences=1,
                temperature=config.INDIVIDUAL_SUMMARY_TEMPERATURE,
                do_sample=True,
                pad_token_id=self.hf_generator.tokenizer.eos_token_id
            )
            
            generated = response[0]['generated_text']
            new_text = generated[len(prompt):].strip()
            
            return new_text if new_text else None
            
        except Exception as e:
            logger.error(f"HuggingFace summary failed: {e}")
            return None
    
    def _generate_simple_individual_summary(self, title: str, abstract: str) -> str:
        """Generate simple rule-based summary"""
        sentences = abstract.split('.')
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences[:3]:
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
        """Generate summary of search results"""
        if not results:
            return "No relevant papers found."
        
        context = f"Query: {query}\n\nFound {len(results)} papers:\n\n"
        
        papers_info = []
        for i, result in enumerate(results[:5], 1):
            paper_summary = result.ai_summary or result.get_summary(100)
            papers_info.append({
                'title': result.title,
                'summary': paper_summary,
                'categories': result.categories[:2],
                'score': result.similarity_score
            })
            
            context += f"{i}. {result.title}\n"
            context += f"   Score: {result.similarity_score:.3f}\n"
            context += f"   Summary: {paper_summary}\n\n"
        
        if config.ENABLE_BATCH_SUMMARIES:
            prompt = config.BATCH_SUMMARY_PROMPT.format(
                count=len(results),
                query=query,
                papers_info=str(papers_info)[:1500]
            )
            
            if self.use_openai and self.openai_client:
                return self._generate_batch_with_openai(prompt)
        
        return self._generate_simple_batch_summary(results, query)
    
    def _generate_batch_with_openai(self, prompt: str) -> str:
        """Generate batch summary using OpenAI"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.BATCH_SUMMARY_MAX_TOKENS,
                temperature=config.BATCH_SUMMARY_TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Batch summary failed: {e}")
            return "AI summary unavailable."
    
    def _generate_simple_batch_summary(self, results: List[SearchResult], query: str) -> str:
        """Generate simple rule-based summary"""
        if not results:
            return "No papers found."
        
        all_categories = []
        for result in results[:10]:
            all_categories.extend(result.categories)
        
        from collections import Counter
        top_categories = Counter(all_categories).most_common(5)
        
        recent = sum(1 for r in results if r.update_date and 
                    r.update_date.year >= datetime.now().year - 2)
        
        available_pdfs = sum(1 for r in results if r.pdf_available)
        with_summaries = sum(1 for r in results if r.ai_summary)
        
        summary = f"Found {len(results)} papers related to '{query}'. "
        summary += f"{available_pdfs} PDFs available. "
        
        if with_summaries:
            summary += f"{with_summaries} with AI summaries. "
        
        if top_categories:
            cats = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
            summary += f"Main areas: {cats}. "
        
        if recent:
            summary += f"{recent} from last 2 years. "
        
        summary += f"Top result: '{results[0].title}' (score: {results[0].similarity_score:.3f})."
        
        return summary


class ArXivSearchEngine:
    """
    Enhanced search engine with multi-stage retrieval pipeline
    Features:
    - Query expansion for better recall
    - Cross-encoder re-ranking for precision
    - Hybrid embeddings for quality
    - Proper similarity normalization
    """
    
    def __init__(self, data_df: pd.DataFrame = None, use_reranker: bool = True, use_hybrid: bool = True):
        self.data_df = data_df
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid
        self.embedding_generator = None
        self.vector_store = None
        self.ai_assistant = GenerativeAIAssistant()
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding generator and vector store"""
        logger.info("Initializing enhanced search engine...")
        logger.info(f"Re-ranker: {self.use_reranker}, Hybrid embeddings: {self.use_hybrid}")
        
        # Initialize embedding generator with enhancements
        self.embedding_generator = EmbeddingGenerator(
            use_reranker=self.use_reranker,
            use_hybrid=self.use_hybrid
        )
        
        embedding_dim = self.embedding_generator.embedding_dim
        self.vector_store = FAISSVectorStore(embedding_dim)
        
        try:
            self.vector_store.load_index()
            logger.info("Loaded existing index")
        except FileNotFoundError:
            if self.data_df is not None:
                logger.info("Building new index with enhanced embeddings...")
                self._build_index()
            else:
                logger.warning("No index or data available")
    
    def _build_index(self):
        """Build vector index from data using enhanced embeddings"""
        if self.data_df is None:
            raise ValueError("No data for building index")
        
        # Generate embeddings with hybrid strategy
        embeddings = self.embedding_generator.generate_embeddings_from_dataframe(
            self.data_df, 
            text_column='combined_text',
            use_hybrid=self.use_hybrid
        )
        
        document_ids = self.data_df['id'].tolist()
        self.vector_store.build_index(embeddings, document_ids)
        self.vector_store.save_index()
        
        logger.info("Enhanced index built successfully")
    
    def _prepare_document_texts(self) -> List[str]:
        """Prepare document texts for re-ranking"""
        document_texts = []
        for _, row in self.data_df.iterrows():
            # Combine title and truncated abstract for re-ranking
            text = f"{row['title']} {row['abstract'][:500]}"
            document_texts.append(text)
        return document_texts
    
    def search(self, query: str, top_k: int = None, min_similarity: float = None, 
               category_filter: List[str] = None, date_filter: Dict = None,
               year_filter: List[int] = None,
               generate_summaries: bool = None, use_reranking: bool = None,
               use_query_expansion: bool = True) -> List[SearchResult]:
        """
        Enhanced semantic search with multi-stage retrieval
        
        Pipeline:
        1. Query expansion (optional)
        2. Fast vector search (retrieve candidates)
        3. Cross-encoder re-ranking (if enabled)
        4. Category/date filtering
        5. AI summary generation (if enabled)
        """
        top_k = top_k or config.TOP_K_RESULTS
        min_similarity = min_similarity or config.SIMILARITY_THRESHOLD
        generate_summaries = generate_summaries if generate_summaries is not None else config.ENABLE_INDIVIDUAL_SUMMARIES
        use_reranking = use_reranking if use_reranking is not None else self.use_reranker
        
        if not query.strip():
            return []
        
        logger.info(f"🔍 Searching: '{query}'")
        logger.info(f"   Settings: top_k={top_k}, rerank={use_reranking}, expand={use_query_expansion}")
        
        # Determine how many candidates to retrieve
        # More candidates if we're filtering or re-ranking
        retrieve_multiplier = 1
        if category_filter or date_filter:
            retrieve_multiplier = max(retrieve_multiplier, 3)
        if use_reranking and self.embedding_generator.reranker:
            retrieve_multiplier = max(retrieve_multiplier, 5)
        
        search_k = min(top_k * retrieve_multiplier, len(self.data_df))
        
        # Get vectors from vector store
        if hasattr(self.vector_store, 'vectors'):
            all_vectors = self.vector_store.vectors
        else:
            # Fallback: load from embeddings
            try:
                all_vectors, _ = self.embedding_generator.load_embeddings()
            except:
                logger.error("Could not load vectors for re-ranking")
                use_reranking = False
                all_vectors = None
        
        # Perform search with or without re-ranking
        if use_reranking and self.embedding_generator.reranker and all_vectors is not None:
            logger.info(f"   Stage 1: Retrieving {search_k} candidates")
            logger.info(f"   Stage 2: Re-ranking with cross-encoder")
            
            # Prepare document texts for re-ranking
            document_texts = self._prepare_document_texts()
            
            # Two-stage retrieval with re-ranking
            top_indices, similarities = self.embedding_generator.find_most_similar_with_reranking(
                query,
                all_vectors,
                document_texts,
                top_k=search_k,
                rerank_top=min(search_k, 100),  # Re-rank top 100 candidates
                use_query_expansion=use_query_expansion
            )
            
            # Convert indices to document IDs
            document_ids = [self.data_df.iloc[idx]['id'] for idx in top_indices]
            
        else:
            # Standard vector search (single stage)
            logger.info(f"   Using vector search (no re-ranking)")
            
            # Generate query embedding with expansion
            if use_query_expansion:
                query_embedding = self.embedding_generator.generate_query_embedding_enhanced(query)
            else:
                query_embedding = self.embedding_generator.generate_single_embedding(query, normalize=True)
            
            # Perform vector search
            document_ids, similarities = self.vector_store.search(query_embedding, search_k)
        
        # Ensure similarities are numpy array
        similarities = np.array(similarities)
        
        # Build SearchResult objects
        results = []
        summary_count = 0
        
        for rank, (doc_id, similarity) in enumerate(zip(document_ids, similarities), 1):
            # Apply minimum similarity threshold
            if similarity < min_similarity:
                continue
            
            # Find document in dataframe
            doc_row = self.data_df[self.data_df['id'] == doc_id]
            
            if doc_row.empty:
                continue
            
            doc_data = doc_row.iloc[0]
            
            # Apply category filter
            if category_filter:
                doc_categories = doc_data['categories']
                if not any(cat in doc_categories for cat in category_filter):
                    continue
            
            # Apply date filter
            if date_filter:
                doc_date = doc_data['update_date']
                if doc_date and not self._passes_date_filter(doc_date, date_filter):
                    continue
            
            if year_filter:
                doc_year = doc_data.get('year')
                if doc_year is None or int(doc_year) not in year_filter:
                    continue
                
            # Generate AI summary if enabled
            ai_summary = None
            if (generate_summaries and 
                summary_count < config.MAX_SUMMARIES_PER_SEARCH and
                self.ai_assistant):
                
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
                similarity_score=float(similarity),
                update_date=doc_data['update_date'],
                ai_summary=ai_summary,
                rank=rank
            )
            
            results.append(result)
            
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        logger.info(f"✅ Found {len(results)} results ({summary_count} with AI summaries)")
        
        return results
    
    def search_with_ai_summary(self, query: str, **kwargs) -> Dict:
        """Perform search and generate overall AI summary"""
        results = self.search(query, **kwargs)
        
        ai_summary = ""
        if results:
            ai_summary = self.ai_assistant.summarize_results(results, query)
        
        downloadable_count = sum(1 for r in results if r.pdf_available)
        summary_count = sum(1 for r in results if r.ai_summary)
        
        estimated_cost = 0.0
        if config.USE_OPENAI and config.ENABLE_INDIVIDUAL_SUMMARIES:
            estimated_cost = config.estimate_cost_per_search(len(results))
        
        # Calculate average similarity for quality metric
        avg_similarity = np.mean([r.similarity_score for r in results]) if results else 0.0
        
        return {
            'query': query,
            'results': [result.to_dict() for result in results],
            'ai_summary': ai_summary,
            'total_results': len(results),
            'downloadable_pdfs': downloadable_count,
            'ai_summaries_count': summary_count,
            'estimated_cost': estimated_cost,
            'average_similarity': float(avg_similarity),
            'used_reranking': kwargs.get('use_reranking', self.use_reranker),
            'search_timestamp': datetime.now().isoformat()
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
    
    def get_similar_papers(self, paper_id: str, top_k: int = 5, 
                          generate_summaries: bool = True,
                          use_reranking: bool = None) -> List[SearchResult]:
        """Find papers similar to a given paper"""
        use_reranking = use_reranking if use_reranking is not None else self.use_reranker
        
        paper_row = self.data_df[self.data_df['id'] == paper_id]
        
        if paper_row.empty:
            logger.warning(f"Paper {paper_id} not found")
            return []
        
        paper_data = paper_row.iloc[0]
        
        # Use title + abstract as query for finding similar papers
        query_text = f"{paper_data['title']} {paper_data['abstract'][:500]}"
        
        # Search for similar papers (get more to filter out the original)
        results = self.search(
            query_text, 
            top_k=top_k + 1, 
            generate_summaries=generate_summaries,
            use_reranking=use_reranking,
            use_query_expansion=False  # Don't expand when finding similar papers
        )
        
        # Remove original paper from results
        filtered_results = [r for r in results if r.document_id != paper_id]
        
        return filtered_results[:top_k]
    
    def download_paper_pdf(self, arxiv_id: str, download_dir: str = "downloads") -> Dict:
        """Download PDF for a specific paper"""
        if not validate_arxiv_id(arxiv_id):
            return {'success': False, 'error': 'Invalid ArXiv ID', 'file_path': None}
        
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
            return {'success': False, 'error': 'Download failed', 'file_path': None}
        except Exception as e:
            return {'success': False, 'error': str(e), 'file_path': None}
    
    def export_search_results(self, results: List[SearchResult], format: str = 'json') -> str:
        """Export search results in various formats"""
        if format == 'json':
            return json.dumps([r.to_dict() for r in results], indent=2)
        elif format == 'csv':
            df = pd.DataFrame([r.to_dict() for r in results])
            return df.to_csv(index=False)
        elif format == 'bibtex':
            bibtex_entries = []
            for r in results:
                entry = f"@article{{{r.arxiv_id},\n"
                entry += f"  title = {{{r.title}}},\n"
                entry += f"  author = {{{' and '.join(r.authors[:3])}}},\n"
                if r.update_date:
                    entry += f"  year = {{{r.update_date.year}}},\n"
                entry += f"  arxiv = {{{r.arxiv_id}}}\n}}\n"
                bibtex_entries.append(entry)
            return '\n'.join(bibtex_entries)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_search_stats(self) -> Dict:
        """Get statistics about the search engine"""
        return {
            'total_documents': len(self.data_df) if self.data_df is not None else 0,
            'embedding_model': self.embedding_generator.model_name,
            'embedding_dimension': self.embedding_generator.embedding_dim,
            'reranker_enabled': self.use_reranker and self.embedding_generator.reranker is not None,
            'hybrid_embeddings': self.use_hybrid,
            'device': self.embedding_generator.device,
            'model_info': self.embedding_generator.get_model_info()
        }


def compare_search_methods(search_engine: ArXivSearchEngine, query: str, top_k: int = 5):
    """
    Compare different search methods for quality assessment
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    
    # Method 1: Basic vector search (no enhancements)
    print("🔹 Method 1: Basic Vector Search (no enhancements)")
    print("-" * 80)
    results_basic = search_engine.search(
        query, 
        top_k=top_k, 
        use_reranking=False, 
        use_query_expansion=False
    )
    for i, r in enumerate(results_basic, 1):
        print(f"{i}. [{r.similarity_score:.4f}] {r.title[:70]}...")
    
    # Method 2: Vector search with query expansion
    print("\n🔹 Method 2: Vector Search + Query Expansion")
    print("-" * 80)
    results_expanded = search_engine.search(
        query, 
        top_k=top_k, 
        use_reranking=False, 
        use_query_expansion=True
    )
    for i, r in enumerate(results_expanded, 1):
        print(f"{i}. [{r.similarity_score:.4f}] {r.title[:70]}...")
    
    # Method 3: Full enhancement (expansion + re-ranking)
    if search_engine.embedding_generator.reranker:
        print("\n⭐ Method 3: Full Enhancement (Expansion + Re-ranking)")
        print("-" * 80)
        results_full = search_engine.search(
            query, 
            top_k=top_k, 
            use_reranking=True, 
            use_query_expansion=True
        )
        for i, r in enumerate(results_full, 1):
            print(f"{i}. [{r.similarity_score:.4f}] {r.title[:70]}...")
    else:
        print("\n⚠️  Re-ranker not available")
    
    print(f"\n{'='*80}\n")


def main():
    """
    Main function for testing enhanced search engine
    """
    from data_processor import ArXivDataProcessor
    
    print("="*80)
    print("Enhanced ArXiv Search Engine - Testing")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    processor = ArXivDataProcessor()
    df = processor.process_dataset()
    
    # Use subset for testing
    sample_size = 1000
    sample_df = df.head(sample_size)
    print(f"✅ Loaded {len(sample_df)} papers for testing\n")
    
    # Initialize enhanced search engine
    print("🚀 Initializing Enhanced Search Engine...")
    print("   - Cross-encoder re-ranking: Enabled")
    print("   - Hybrid embeddings: Enabled")
    print("   - Query expansion: Enabled")
    
    search_engine = ArXivSearchEngine(
        sample_df, 
        use_reranker=True, 
        use_hybrid=True
    )
    
    # Display search engine stats
    stats = search_engine.get_search_stats()
    print(f"\n📈 Search Engine Stats:")
    print(f"   - Total documents: {stats['total_documents']:,}")
    print(f"   - Embedding model: {stats['embedding_model']}")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    print(f"   - Device: {stats['device']}")
    print(f"   - Re-ranker enabled: {stats['reranker_enabled']}")
    print(f"   - Hybrid embeddings: {stats['hybrid_embeddings']}")
    
    # Test queries
    test_queries = [
        "transformer architectures for computer vision",
        "reinforcement learning in robotics",
        "graph neural networks for drug discovery"
    ]
    
    print(f"\n{'='*80}")
    print("Running Comparative Tests")
    print(f"{'='*80}")
    
    # Compare methods for each query
    for query in test_queries:
        compare_search_methods(search_engine, query, top_k=5)
    
    # Test advanced search with filters
    print(f"\n{'='*80}")
    print("Testing Advanced Search with Filters")
    print(f"{'='*80}\n")
    
    query = "neural networks"
    print(f"Query: {query}")
    print(f"Filters: Categories=['cs.LG', 'cs.AI'], Min similarity=0.5")
    print("-" * 80)
    
    results_filtered = search_engine.search(
        query,
        top_k=5,
        category_filter=['cs.LG', 'cs.AI'],
        min_similarity=0.5,
        use_reranking=True
    )
    
    for i, r in enumerate(results_filtered, 1):
        print(f"{i}. [{r.similarity_score:.4f}] {r.title[:70]}...")
        print(f"   Categories: {', '.join(r.categories[:3])}")
    
    # Test similar papers functionality
    print(f"\n{'='*80}")
    print("Testing Similar Papers Feature")
    print(f"{'='*80}\n")
    
    if results_filtered:
        base_paper = results_filtered[0]
        print(f"Finding papers similar to: {base_paper.title[:80]}...")
        print("-" * 80)
        
        similar_papers = search_engine.get_similar_papers(
            base_paper.document_id,
            top_k=5,
            use_reranking=True
        )
        
        for i, r in enumerate(similar_papers, 1):
            print(f"{i}. [{r.similarity_score:.4f}] {r.title[:70]}...")
    
    # Test search with AI summary
    print(f"\n{'='*80}")
    print("Testing Search with AI Summary")
    print(f"{'='*80}\n")
    
    query = "deep learning"
    search_results = search_engine.search_with_ai_summary(
        query,
        top_k=3,
        use_reranking=True
    )
    
    print(f"Query: {query}")
    print(f"Total results: {search_results['total_results']}")
    print(f"Average similarity: {search_results['average_similarity']:.4f}")
    print(f"Used re-ranking: {search_results['used_reranking']}")
    print(f"\nOverall AI Summary:")
    print("-" * 80)
    print(search_results['ai_summary'])
    
    print(f"\n{'='*80}")
    print("Testing Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()