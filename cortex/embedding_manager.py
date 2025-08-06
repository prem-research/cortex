"""
Unified embedding manager supporting both OpenAI and local SentenceTransformers models.

Features:
- Automatic model type detection (OpenAI vs local)
- Thread-safe singleton pattern for efficient resource reuse
- Consistent embeddings across all components  
- Automatic batching for multiple text processing
- Memory-efficient processing with caching
- Error handling and fallbacks
- Seamless switching between embedding backends
"""

import threading
from typing import List, Dict, Optional
import logging
import os
from cortex.constants import (
    DEFAULT_EMBEDDING_MODEL, is_openai_model, is_local_model, 
    get_embedding_dimension
)

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Thread-safe singleton manager supporting both OpenAI and local embedding models.
    
    Automatically detects model type and uses appropriate backend:
    - OpenAI models: Uses OpenAI API
    - Local models: Uses SentenceTransformers
    """
    
    _instances: Dict[str, 'EmbeddingManager'] = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Singleton pattern: return existing instance or create new one."""
        with cls._lock:
            if model_name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[model_name] = instance
                instance._initialized = False
            return cls._instances[model_name]

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize embedding manager for any supported model."""
        if self._initialized:
            return
            
        self.model_name = model_name
        self._initialized = True
        self.is_openai = is_openai_model(model_name)
        self.is_local = is_local_model(model_name)
        
        if self.is_openai:
            self._init_openai()
        elif self.is_local:
            self._init_local()
        else:
            # Try OpenAI as fallback for unknown models
            logger.warning(f"Unknown model {model_name}, trying OpenAI backend")
            self.is_openai = True
            self._init_openai()

    def _init_openai(self):
        """Initialize OpenAI backend"""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Created EmbeddingManager for OpenAI model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI backend: {e}")
            raise

    def _init_local(self):
        """Initialize local SentenceTransformers backend"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Created EmbeddingManager for local model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize local model {self.model_name}: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """ the dimension of embeddings for this model."""
        if self.is_local and hasattr(self, 'model'):
            # For local models, get dimension from loaded model
            return self.model.get_sentence_embedding_dimension()
        else:
            # Use predefined dimensions from constants
            return get_embedding_dimension(self.model_name)

    def get_embedding(self, content: str) -> List[float]:
        """ embeddings for text content using appropriate backend."""
        if not content or not content.strip():
            # Return zero embedding for empty content
            return [0.0] * self.get_embedding_dimension()
        
        try:
            if self.is_openai:
                return self._get_openai_embedding(content.strip())
            else:
                return self._get_local_embedding(content.strip())
        except Exception as e:
            logger.error(f"Error generating embedding with {self.model_name}: {e}")
            # Return zero embedding as fallback
            return [0.0] * self.get_embedding_dimension()

    def _get_openai_embedding(self, content: str) -> List[float]:
        """ OpenAI embedding"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=content
        )
        return response.data[0].embedding

    def _get_local_embedding(self, content: str) -> List[float]:
        """ local SentenceTransformers embedding"""
        embedding = self.model.encode(content)
        return embedding.tolist()

    def get_embeddings(self, contents: List[str]) -> List[List[float]]:
        """ embeddings for multiple text contents efficiently."""
        if not contents:
            return []
        
        # Prepare contents and track empty ones
        processed_contents = []
        empty_indices = []
        
        for i, content in enumerate(contents):
            if not content or not content.strip():
                empty_indices.append(i)
                processed_contents.append("")  # Placeholder
            else:
                processed_contents.append(content.strip())
        
        try:
            if self.is_openai:
                return self._get_openai_embeddings(processed_contents, empty_indices)
            else:
                return self._get_local_embeddings(processed_contents, empty_indices)
        except Exception as e:
            logger.error(f"Error generating batch embeddings with {self.model_name}: {e}")
            # Return zero embeddings as fallback
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in contents]

    def _get_openai_embeddings(self, processed_contents: List[str], empty_indices: List[int]) -> List[List[float]]:
        """ batch OpenAI embeddings"""
        # Filter out empty contents for API call
        non_empty_contents = [c for c in processed_contents if c]
        
        if not non_empty_contents:
            # All contents are empty
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in processed_contents]
        
        # Use OpenAI batch embedding API
        response = self.client.embeddings.create(
            model=self.model_name,
            input=non_empty_contents
        )
        
        # Reconstruct full list with zero embeddings for empty contents
        embeddings = []
        non_empty_idx = 0
        dim = self.get_embedding_dimension()
        
        for i, content in enumerate(processed_contents):
            if i in empty_indices:
                embeddings.append([0.0] * dim)
            else:
                embeddings.append(response.data[non_empty_idx].embedding)
                non_empty_idx += 1
                
        return embeddings

    def _get_local_embeddings(self, processed_contents: List[str], empty_indices: List[int]) -> List[List[float]]:
        """ batch local embeddings"""
        # Filter out empty contents
        non_empty_contents = [c for c in processed_contents if c]
        
        if not non_empty_contents:
            # All contents are empty
            dim = self.get_embedding_dimension()
            return [[0.0] * dim for _ in processed_contents]
        
        # Use SentenceTransformers batch encoding
        batch_embeddings = self.model.encode(non_empty_contents)
        
        # Reconstruct full list with zero embeddings for empty contents
        embeddings = []
        non_empty_idx = 0
        dim = self.get_embedding_dimension()
        
        for i, content in enumerate(processed_contents):
            if i in empty_indices:
                embeddings.append([0.0] * dim)
            else:
                embeddings.append(batch_embeddings[non_empty_idx].tolist())
                non_empty_idx += 1
                
        return embeddings

    @classmethod
    def get_manager(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> 'EmbeddingManager':
        """ or create an embedding manager for the specified model."""
        return cls(model_name)

    @classmethod
    def clear_cache(cls):
        """ all cached embedding managers."""
        with cls._lock:
            cls._instances.clear()
        logger.info("Cleared embedding manager cache")
