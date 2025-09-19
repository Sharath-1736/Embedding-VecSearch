"""
Embedding generation module using Sentence Transformers
Handles text vectorization for semantic search
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Union, Optional
import logging
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text documents using pre-trained models
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.batch_size = config.BATCH_SIZE
        self.max_length = config.MAX_TEXT_LENGTH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Paths for saving embeddings
        self.embeddings_path = config.INDEX_DIR / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        self.metadata_path = config.INDEX_DIR / f"metadata_{self.model_name.replace('/', '_')}.pkl"
        
        logger.info(f"Initializing EmbeddingGenerator with model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Load model
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get actual embedding dimension
            sample_text = "This is a sample text for testing."
            sample_embedding = self.model.encode([sample_text])
            self.embedding_dim = sample_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation
        """
        if not isinstance(text, str):
            return ""
        
        # Truncate text to max length if needed
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            text = self.tokenizer.convert_tokens_to_string(tokens)
        
        return text.strip()
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        """
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text:
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(
            [preprocessed_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embedding[0]
    
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
            normalize_embeddings=True  # Normalize for cosine similarity
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
    
    def generate_embeddings_from_dataframe(self, df: pd.DataFrame, text_column: str = 'combined_text', 
                                         save_embeddings: bool = True) -> np.ndarray:
        """
        Generate embeddings for all texts in a DataFrame
        """
        logger.info(f"Generating embeddings for {len(df)} documents...")
        
        # Check if embeddings already exist
        if self.embeddings_path.exists() and self.metadata_path.exists():
            logger.info(f"Loading existing embeddings from {self.embeddings_path}")
            with open(self.embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verify embeddings match current data
            if len(embeddings) == len(df) and metadata.get('model_name') == self.model_name:
                logger.info("Existing embeddings are compatible")
                return embeddings
            else:
                logger.info("Existing embeddings are incompatible, regenerating...")
        
        # Extract texts
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
                'text_column': text_column
            }
            
            logger.info(f"Saving embeddings to {self.embeddings_path}")
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        return embeddings
    
    def load_embeddings(self) -> tuple[np.ndarray, dict]:
        """
        Load pre-generated embeddings and metadata
        """
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Pre-generated embeddings not found")
        
        with open(self.embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return embeddings, metadata
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings
        """
        # Ensure embeddings are normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def find_most_similar(self, query_text: str, document_embeddings: np.ndarray, 
                         top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Find most similar documents to a query
        """
        # Generate query embedding
        query_embedding = self.generate_single_embedding(query_text)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, document_embeddings)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }

class MultiModelEmbeddingGenerator:
    """
    Handles multiple embedding models for comparison
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
    Main function for testing embedding generation
    """
    from data_processor import ArXivDataProcessor
    
    # Load processed data
    processor = ArXivDataProcessor()
    df = processor.process_dataset()
    
    # Take a small sample for testing
    sample_df = df.head(1000)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator()
    
    # Generate embeddings
    embeddings = generator.generate_embeddings_from_dataframe(sample_df)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Model info: {generator.get_model_info()}")
    
    # Test similarity search
    query = "machine learning neural networks"
    top_indices, top_scores = generator.find_most_similar(query, embeddings, top_k=5)
    
    print(f"\nTop 5 results for query: '{query}'")
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        paper = sample_df.iloc[idx]
        print(f"{i+1}. Score: {score:.4f}")
        print(f"   Title: {paper['title'][:100]}...")
        print(f"   Categories: {', '.join(paper['categories'][:3])}")
        print()

if __name__ == "__main__":
    main()