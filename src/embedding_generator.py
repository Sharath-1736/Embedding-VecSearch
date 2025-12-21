"""
Enhanced Embedding Generation Module with Advanced Semantic Search
Features:
- Hybrid title+abstract embeddings
- Cross-encoder re-ranking for improved accuracy
- Query expansion and enhancement
- Multiple embedding strategies
- Proper normalization
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Union, Optional, Tuple
import logging
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Enhanced embedding generator with multiple strategies for better semantic matching
    """
    
    def __init__(self, model_name: str = None, use_reranker: bool = True, use_hybrid: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.batch_size = config.BATCH_SIZE
        self.max_length = config.MAX_TEXT_LENGTH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        
        # Paths for saving embeddings
        self.embeddings_path = config.INDEX_DIR / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        self.metadata_path = config.INDEX_DIR / f"metadata_{self.model_name.replace('/', '_')}.pkl"
        
        logger.info(f"Initializing EmbeddingGenerator with model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Hybrid embeddings: {self.use_hybrid}, Re-ranker: {self.use_reranker}")
        
        # Load models
        self.model = None
        self.tokenizer = None
        self.reranker = None
        self._load_model()
        self._load_reranker()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get actual embedding dimension
            sample_text = "This is a sample text for testing."
            sample_embedding = self.model.encode([sample_text], normalize_embeddings=True)
            self.embedding_dim = sample_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _load_reranker(self):
        """Load cross-encoder model for re-ranking"""
        if not self.use_reranker:
            return
        
        try:
            logger.info("Loading cross-encoder re-ranker...")
            # Use a cross-encoder specifically trained for semantic search
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
            logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load re-ranker: {e}. Continuing without re-ranking.")
            self.reranker = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation
        """
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Truncate text to max length if needed
        try:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                text = self.tokenizer.convert_tokens_to_string(tokens)
        except Exception as e:
            # Fallback if tokenization fails
            if len(text) > self.max_length * 4:  # Rough char estimate
                text = text[:self.max_length * 4]
        
        return text.strip()
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with related terms for better semantic matching
        """
        expansions = {
            'transformer': 'transformer attention mechanism self-attention',
            'transformers': 'transformer attention mechanism self-attention',
            'neural network': 'neural network deep learning artificial intelligence',
            'neural networks': 'neural network deep learning artificial intelligence',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'cnn': 'convolutional neural network image recognition',
            'rnn': 'recurrent neural network sequential data',
            'lstm': 'long short-term memory network recurrent',
            'gru': 'gated recurrent unit network',
            'gnn': 'graph neural network node classification',
            'bert': 'bert transformer language model nlp',
            'gpt': 'gpt generative pre-trained transformer language model',
            'nlp': 'natural language processing text analysis',
            'cv': 'computer vision image recognition visual',
            'rl': 'reinforcement learning agent policy',
            'gan': 'generative adversarial network image generation',
            'vae': 'variational autoencoder generative model',
            'attention': 'attention mechanism self-attention cross-attention',
            'classification': 'classification prediction supervised learning',
            'regression': 'regression prediction supervised learning',
            'clustering': 'clustering unsupervised learning grouping',
            'detection': 'object detection localization bounding box',
            'segmentation': 'image segmentation pixel-level classification',
        }
        
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Find matching expansions
        for term, expansion in expansions.items():
            if f' {term} ' in f' {query_lower} ' or query_lower.startswith(f'{term} ') or query_lower.endswith(f' {term}'):
                expanded_terms.append(expansion)
                logger.debug(f"Expanded '{term}' to '{expansion}'")
        
        return ' '.join(expanded_terms)
    
    def generate_single_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text (e.g., query)
        FIXED: Now properly normalizes embeddings
        """
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text:
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(
            [preprocessed_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize  # FIXED: Now normalizes queries too
        )
        
        return embedding[0]
    
    def generate_query_embedding_enhanced(self, query: str) -> np.ndarray:
        """
        Generate enhanced query embedding with expansion
        Combines original query with expanded version for better recall
        """
        # Original query embedding
        original_emb = self.generate_single_embedding(query, normalize=True)
        
        # Create expanded query with synonyms/related terms
        expanded_query = self._expand_query(query)
        
        # Only create expanded embedding if query was actually expanded
        if expanded_query != query:
            expanded_emb = self.generate_single_embedding(expanded_query, normalize=True)
            # Combine: 70% original, 30% expanded (original query is more important)
            combined = 0.7 * original_emb + 0.3 * expanded_emb
            # Re-normalize
            combined = combined / np.linalg.norm(combined)
            logger.debug(f"Query expanded: '{query}' -> '{expanded_query[:100]}...'")
            return combined
        else:
            return original_emb
    
    def generate_hybrid_embedding(self, title: str, abstract: str, 
                                  title_weight: float = 0.4) -> np.ndarray:
        """
        Generate weighted combination of title and abstract embeddings
        Title often contains key concepts, abstract provides context
        """
        # Generate separate embeddings
        title_emb = self.model.encode(
            [self.preprocess_text(title)],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        abstract_emb = self.model.encode(
            [self.preprocess_text(abstract)],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        # Weighted combination
        hybrid = title_weight * title_emb + (1 - title_weight) * abstract_emb
        
        # Re-normalize
        hybrid = hybrid / np.linalg.norm(hybrid)
        
        return hybrid
    
    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        """
        # Preprocess all texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Filter out empty texts
        valid_texts = [text for text in preprocessed_texts if text]
        
        if not valid_texts:
            return np.zeros((len(texts), self.embedding_dim))
        
        # Generate embeddings
        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Always normalize for cosine similarity
        )
        
        # Handle case where some texts were empty
        if len(valid_texts) < len(texts):
            full_embeddings = np.zeros((len(texts), self.embedding_dim))
            valid_idx = 0
            for i, text in enumerate(preprocessed_texts):
                if text:
                    full_embeddings[i] = embeddings[valid_idx]
                    valid_idx += 1
            return full_embeddings
        
        return embeddings
    
    def generate_embeddings_from_dataframe(self, df: pd.DataFrame, 
                                          text_column: str = 'combined_text',
                                          use_hybrid: bool = None,
                                          save_embeddings: bool = True) -> np.ndarray:
        """
        Generate embeddings for all texts in a DataFrame
        Supports hybrid title+abstract embeddings for better quality
        """
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid
        
        logger.info(f"Generating embeddings for {len(df)} documents...")
        logger.info(f"Using hybrid embeddings: {use_hybrid}")
        
        # Check if embeddings already exist
        if self.embeddings_path.exists() and self.metadata_path.exists():
            logger.info(f"Loading existing embeddings from {self.embeddings_path}")
            with open(self.embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verify embeddings match current data and settings
            if (len(embeddings) == len(df) and 
                metadata.get('model_name') == self.model_name and
                metadata.get('use_hybrid') == use_hybrid):
                logger.info("Existing embeddings are compatible")
                return embeddings
            else:
                logger.info("Existing embeddings are incompatible, regenerating...")
        
        # Generate embeddings based on strategy
        if use_hybrid and 'title' in df.columns and 'abstract' in df.columns:
            logger.info("Generating hybrid title+abstract embeddings (weighted 40%/60%)")
            all_embeddings = []
            
            for i in tqdm(range(0, len(df), self.batch_size), desc="Generating hybrid embeddings"):
                batch_df = df.iloc[i:i + self.batch_size]
                batch_embeddings = []
                
                for _, row in batch_df.iterrows():
                    emb = self.generate_hybrid_embedding(
                        row['title'], 
                        row['abstract'],
                        title_weight=0.4  # 40% title, 60% abstract
                    )
                    batch_embeddings.append(emb)
                
                all_embeddings.append(np.array(batch_embeddings))
            
            embeddings = np.vstack(all_embeddings)
        else:
            # Standard combined text embeddings
            logger.info(f"Generating standard embeddings from '{text_column}'")
            texts = df[text_column].tolist()
            
            # Generate embeddings in batches
            all_embeddings = []
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.generate_batch_embeddings(batch_texts)
                all_embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        # Save embeddings and metadata
        if save_embeddings:
            metadata = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'num_documents': len(embeddings),
                'generation_date': pd.Timestamp.now().isoformat(),
                'text_column': text_column,
                'use_hybrid': use_hybrid,
                'device': self.device
            }
            
            logger.info(f"Saving embeddings to {self.embeddings_path}")
            config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
            
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        return embeddings
    
    def load_embeddings(self) -> Tuple[np.ndarray, dict]:
        """
        Load pre-generated embeddings and metadata
        """
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Pre-generated embeddings not found")
        
        with open(self.embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Loaded embeddings: {embeddings.shape}")
        logger.info(f"Metadata: {metadata}")
        
        return embeddings, metadata
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings
        """
        # Ensure embeddings are normalized
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity (dot product for normalized vectors)
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def find_most_similar(self, query_text: str, document_embeddings: np.ndarray, 
                         top_k: int = 10, use_query_expansion: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar documents to a query
        """
        # Generate query embedding (with optional expansion)
        if use_query_expansion:
            query_embedding = self.generate_query_embedding_enhanced(query_text)
        else:
            query_embedding = self.generate_single_embedding(query_text, normalize=True)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, document_embeddings)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def find_most_similar_with_reranking(self, query_text: str, 
                                        document_embeddings: np.ndarray,
                                        document_texts: List[str],
                                        top_k: int = 10,
                                        rerank_top: int = 50,
                                        use_query_expansion: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage retrieval: fast vector search + accurate re-ranking
        
        Stage 1: Use bi-encoder (fast) to retrieve top candidates
        Stage 2: Use cross-encoder (accurate) to re-rank candidates
        
        This gives much better results than vector search alone!
        """
        # Stage 1: Fast vector search (retrieve more candidates)
        if use_query_expansion:
            query_embedding = self.generate_query_embedding_enhanced(query_text)
        else:
            query_embedding = self.generate_single_embedding(query_text, normalize=True)
        
        similarities = self.compute_similarity(query_embedding, document_embeddings)
        
        # Get top candidates for re-ranking (more than final top_k)
        rerank_top = min(rerank_top, len(similarities))
        top_indices_initial = np.argsort(similarities)[::-1][:rerank_top]
        initial_scores = similarities[top_indices_initial]
        
        # Stage 2: Re-rank with cross-encoder (if available)
        if self.reranker is not None:
            logger.debug(f"Re-ranking top {rerank_top} candidates with cross-encoder")
            
            # Prepare pairs for re-ranking
            pairs = [[query_text, document_texts[idx]] for idx in top_indices_initial]
            
            # Get re-ranking scores
            try:
                rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
                
                # Sort by re-ranking scores
                reranked_order = np.argsort(rerank_scores)[::-1]
                reranked_indices = top_indices_initial[reranked_order]
                reranked_scores = rerank_scores[reranked_order]
                
                # Normalize scores to 0-1 range for consistency
                reranked_scores = (reranked_scores - reranked_scores.min()) / (reranked_scores.max() - reranked_scores.min() + 1e-8)
                
                # Return top-k after re-ranking
                logger.debug(f"Re-ranking improved order. Top score: {reranked_scores[0]:.4f}")
                return reranked_indices[:top_k], reranked_scores[:top_k]
                
            except Exception as e:
                logger.warning(f"Re-ranking failed: {e}. Falling back to vector search.")
        
        # Fallback: just use vector search results
        top_indices = top_indices_initial[:top_k]
        top_scores = initial_scores[:top_k]
        return top_indices, top_scores
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model configuration
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'use_hybrid': self.use_hybrid,
            'use_reranker': self.use_reranker,
            'reranker_available': self.reranker is not None
        }


class MultiModelEmbeddingGenerator:
    """
    Handles multiple embedding models for comparison and ensembling
    """
    
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.generators = {}
        
        for model_name in model_names:
            logger.info(f"Initializing generator for {model_name}")
            self.generators[model_name] = EmbeddingGenerator(model_name)
    
    def generate_all_embeddings(self, df: pd.DataFrame, text_column: str = 'combined_text'):
        """
        Generate embeddings using all models
        """
        embeddings_dict = {}
        
        for model_name, generator in self.generators.items():
            logger.info(f"Generating embeddings with {model_name}")
            embeddings = generator.generate_embeddings_from_dataframe(df, text_column)
            embeddings_dict[model_name] = embeddings
        
        return embeddings_dict
    
    def compare_models(self, query_text: str, document_embeddings_dict: dict, 
                      document_titles: List[str], top_k: int = 5):
        """
        Compare search results across different models
        """
        results = {}
        
        for model_name, embeddings in document_embeddings_dict.items():
            generator = self.generators[model_name]
            top_indices, top_scores = generator.find_most_similar(query_text, embeddings, top_k)
            
            results[model_name] = {
                'indices': top_indices,
                'scores': top_scores,
                'titles': [document_titles[i] for i in top_indices]
            }
        
        return results


def main():
    """
    Main function for testing enhanced embedding generation
    """
    from data_processor import ArXivDataProcessor
    
    # Load processed data
    processor = ArXivDataProcessor()
    df = processor.process_dataset()
    
    # Take a small sample for testing
    sample_df = df.head(500)
    
    # Initialize enhanced embedding generator
    logger.info("Testing Enhanced Embedding Generator")
    generator = EmbeddingGenerator(use_reranker=True, use_hybrid=True)
    
    print(f"\nModel Info: {generator.get_model_info()}")
    
    # Generate embeddings
    embeddings = generator.generate_embeddings_from_dataframe(sample_df)
    
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    
    # Prepare document texts for re-ranking
    document_texts = [f"{row['title']} {row['abstract'][:500]}" for _, row in sample_df.iterrows()]
    
    # Test queries
    test_queries = [
        "transformer architectures for computer vision",
        "reinforcement learning in robotics",
        "graph neural networks for molecular prediction"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Without re-ranking
        print("\n🔍 WITHOUT Re-ranking (vector search only):")
        top_indices, top_scores = generator.find_most_similar(
            query, embeddings, top_k=5, use_query_expansion=True
        )
        
        for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            paper = sample_df.iloc[idx]
            print(f"{i}. [{score:.4f}] {paper['title'][:80]}...")
        
        # With re-ranking
        if generator.reranker:
            print("\n⭐ WITH Re-ranking (cross-encoder):")
            top_indices, top_scores = generator.find_most_similar_with_reranking(
                query, embeddings, document_texts, top_k=5, rerank_top=20, use_query_expansion=True
            )
            
            for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                paper = sample_df.iloc[idx]
                print(f"{i}. [{score:.4f}] {paper['title'][:80]}...")


if __name__ == "__main__":
    main()