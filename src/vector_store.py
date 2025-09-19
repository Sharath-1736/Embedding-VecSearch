"""
Vector store implementation using FAISS
Handles efficient storage and retrieval of high-dimensional embeddings
"""
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import time

import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search
    """
    
    def __init__(self, embedding_dim: int, index_type: str = None):
        self.embedding_dim = embedding_dim
        self.index_type = index_type or config.FAISS_INDEX_TYPE
        self.n_clusters = config.N_CLUSTERS
        self.n_probe = config.N_PROBE
        
        # Paths for saving index
        self.index_path = config.INDEX_DIR / f"faiss_index_{self.index_type}.index"
        self.document_map_path = config.INDEX_DIR / f"document_map_{self.index_type}.pkl"
        
        # Initialize index
        self.index = None
        self.document_map = {}
        self.is_trained = False
        
        logger.info(f"Initializing FAISS vector store with {self.index_type} index")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create appropriate FAISS index based on configuration
        """
        n_vectors = embeddings.shape[0]
        
        if self.index_type == "Flat":
            # Exact search using L2 distance
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            logger.info("Created Flat index (exact search)")
            
        elif self.index_type == "IVF":
            # Inverted file index for faster search
            n_clusters = min(self.n_clusters, n_vectors // 10)  # Adjust clusters based on data size
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            index.nprobe = self.n_probe
            logger.info(f"Created IVF index with {n_clusters} clusters")
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 is M parameter
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            logger.info("Created HNSW index")
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def build_index(self, embeddings: np.ndarray, document_ids: List[str] = None):
        """
        Build FAISS index from embeddings
        """
        logger.info(f"Building FAISS index for {len(embeddings)} vectors...")
        
        # Ensure embeddings are float32 (required by FAISS)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = self._create_index(embeddings)
        
        # Train index if needed
        if hasattr(self.index, 'train'):
            logger.info("Training index...")
            start_time = time.time()
            self.index.train(embeddings)
            train_time = time.time() - start_time
            logger.info(f"Index trained in {train_time:.2f} seconds")
            self.is_trained = True
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        start_time = time.time()
        self.index.add(embeddings)
        add_time = time.time() - start_time
        logger.info(f"Vectors added in {add_time:.2f} seconds")
        
        # Create document mapping
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(embeddings))]
        
        self.document_map = {i: doc_id for i, doc_id in enumerate(document_ids)}
        
        logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")
    
    def save_index(self):
        """
        Save FAISS index and document mapping to disk
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        logger.info(f"Saving index to {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))
        
        logger.info(f"Saving document mapping to {self.document_map_path}")
        with open(self.document_map_path, 'wb') as f:
            pickle.dump(self.document_map, f)
    
    def load_index(self):
        """
        Load FAISS index and document mapping from disk
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        if not self.document_map_path.exists():
            raise FileNotFoundError(f"Document mapping file not found: {self.document_map_path}")
        
        logger.info(f"Loading index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"Loading document mapping from {self.document_map_path}")
        with open(self.document_map_path, 'rb') as f:
            self.document_map = pickle.load(f)
        
        # Set search parameters for IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.n_probe
        
        self.is_trained = True
        logger.info(f"Index loaded successfully. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search for k most similar vectors
        """
        if self.index is None:
            raise ValueError("No index available. Build or load index first.")
        
        # Ensure query is float32 and 2D
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Perform search
        start_time = time.time()
        scores, indices = self.index.search(query_embedding, k)
        search_time = time.time() - start_time
        
        # Convert results
        document_ids = []
        similarities = []
        
        for i, score in zip(indices[0], scores[0]):
            if i != -1:  # -1 indicates no match found
                document_ids.append(self.document_map.get(i, f"unknown_{i}"))
                similarities.append(float(score))
        
        logger.debug(f"Search completed in {search_time:.4f} seconds")
        
        return document_ids, similarities
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Batch search for multiple queries
        """
        if self.index is None:
            raise ValueError("No index available. Build or load index first.")
        
        # Ensure queries are float32
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Normalize queries
        faiss.normalize_L2(query_embeddings)
        
        # Perform batch search
        start_time = time.time()
        scores, indices = self.index.search(query_embeddings, k)
        search_time = time.time() - start_time
        
        # Convert results
        all_document_ids = []
        all_similarities = []
        
        for query_indices, query_scores in zip(indices, scores):
            document_ids = []
            similarities = []
            
            for i, score in zip(query_indices, query_scores):
                if i != -1:
                    document_ids.append(self.document_map.get(i, f"unknown_{i}"))
                    similarities.append(float(score))
            
            all_document_ids.append(document_ids)
            all_similarities.append(similarities)
        
        logger.debug(f"Batch search for {len(query_embeddings)} queries completed in {search_time:.4f} seconds")
        
        return all_document_ids, all_similarities
    
    def get_stats(self) -> Dict:
        """
        Get index statistics
        """
        if self.index is None:
            return {"status": "No index loaded"}
        
        stats = {
            "index_type": self.index_type,
            "embedding_dimension": self.embedding_dim,
            "total_vectors": self.index.ntotal,
            "is_trained": self.is_trained,
        }
        
        if hasattr(self.index, 'nprobe'):
            stats["nprobe"] = self.index.nprobe
        
        if hasattr(self.index, 'nlist'):
            stats["nlist"] = self.index.nlist
        
        return stats
    
    def add_vectors(self, embeddings: np.ndarray, document_ids: List[str]):
        """
        Add new vectors to existing index
        """
        if self.index is None:
            raise ValueError("No index available. Build index first.")
        
        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Get current size
        current_size = self.index.ntotal
        
        # Add vectors
        self.index.add(embeddings)
        
        # Update document mapping
        for i, doc_id in enumerate(document_ids):
            self.document_map[current_size + i] = doc_id
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def remove_vectors(self, document_ids: List[str]):
        """
        Remove vectors from index (note: FAISS doesn't support efficient removal)
        """
        # Find indices to remove
        indices_to_remove = []
        for idx, doc_id in self.document_map.items():
            if doc_id in document_ids:
                indices_to_remove.append(idx)
        
        if not indices_to_remove:
            logger.warning("No matching document IDs found for removal")
            return
        
        # FAISS doesn't support efficient removal, so we need to rebuild
        logger.warning("FAISS doesn't support efficient vector removal. Consider rebuilding the index.")
        raise NotImplementedError("Vector removal requires index rebuild")

class MultiIndexVectorStore:
    """
    Manages multiple FAISS indices for different embedding models
    """
    
    def __init__(self, embedding_dims: Dict[str, int]):
        self.embedding_dims = embedding_dims
        self.stores = {}
        
        for model_name, dim in embedding_dims.items():
            logger.info(f"Initializing vector store for {model_name}")
            self.stores[model_name] = FAISSVectorStore(dim)
    
    def build_indices(self, embeddings_dict: Dict[str, np.ndarray], document_ids: List[str]):
        """
        Build indices for all embedding models
        """
        for model_name, embeddings in embeddings_dict.items():
            logger.info(f"Building index for {model_name}")
            self.stores[model_name].build_index(embeddings, document_ids)
            self.stores[model_name].save_index()
    
    def search_all_models(self, query_embeddings: Dict[str, np.ndarray], k: int = 10):
        """
        Search across all models
        """
        results = {}
        
        for model_name, query_emb in query_embeddings.items():
            if model_name in self.stores:
                doc_ids, scores = self.stores[model_name].search(query_emb, k)
                results[model_name] = {
                    "document_ids": doc_ids,
                    "scores": scores
                }
        
        return results

def main():
    """
    Main function for testing vector store
    """
    from data_processor import ArXivDataProcessor
    from embedding_generator import EmbeddingGenerator
    
    # Load processed data
    processor = ArXivDataProcessor()
    df = processor.process_dataset()
    
    # Take a small sample for testing
    sample_df = df.head(1000)
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings_from_dataframe(sample_df)
    
    # Create document IDs
    document_ids = sample_df['id'].tolist()
    
    # Initialize vector store
    vector_store = FAISSVectorStore(embeddings.shape[1])
    
    # Build and save index
    vector_store.build_index(embeddings, document_ids)
    vector_store.save_index()
    
    print(f"Index stats: {vector_store.get_stats()}")
    
    # Test search
    query_text = "machine learning neural networks"
    query_embedding = generator.generate_single_embedding(query_text)
    
    doc_ids, scores = vector_store.search(query_embedding, k=5)
    
    print(f"\nTop 5 results for query: '{query_text}'")
    for doc_id, score in zip(doc_ids, scores):
        paper = sample_df[sample_df['id'] == doc_id].iloc[0]
        print(f"Score: {score:.4f} | {paper['title'][:100]}...")
    
    # Test loading index
    vector_store2 = FAISSVectorStore(embeddings.shape[1])
    vector_store2.load_index()
    print(f"\nLoaded index stats: {vector_store2.get_stats()}")

if __name__ == "__main__":
    main()