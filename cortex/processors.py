from typing import List, Dict, Optional, Any
import numpy as np
import logging

from cortex.constants import (
    MAX_KEYWORDS, DEFAULT_SCORE_FALLBACK, CONTEXT_WEIGHT,
    ORIGINAL_WEIGHT, COMMON_WORDS
)

logger = logging.getLogger(__name__)

class LightPreprocessor:
    """Lightweight preprocessing for STM"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
    
    def process(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """extract basic metadata for STM"""
        result = metadata.copy() if metadata else {}
        
        embedding = self.memory_system.embedding_manager.get_embedding(content)
        result["embedding"] = embedding
        
        if "keywords" not in result:
            words = content.lower().split()
            keywords = [w for w in words if w not in COMMON_WORDS and len(w) > 3]
            result["keywords"] = keywords[:MAX_KEYWORDS] if keywords else []
        
        if "context" not in result:
            result["context"] = "General"
        
        return result


class DeepPreprocessor:
    """Deep preprocessing for LTM"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.llm_controller = memory_system.llm_controller
    
    def process(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Thoroughly extract metadata for LTM storage using LLM for deeper understanding"""
        result = metadata.copy() if metadata else {}
        
        if "keywords" in result and len(result["keywords"]) > 0 and result.get("context") not in ["General", None]:
            return result
        
        try:
            llm_metadata = self.memory_system.analyze_content(content)
            if llm_metadata:
                result["keywords"] = llm_metadata.get("keywords", result.get("keywords", []))
                if result.get("context") in [None, "General"]:
                    result["context"] = llm_metadata.get("context", result.get("context", "General"))
                result["tags"] = llm_metadata.get("tags", result.get("tags", []))
                # Allow LLM to set category when missing or generic
                if "category" in llm_metadata:
                    current_cat = result.get("category")
                    if not current_cat or current_cat in ["", "Uncategorized", "General"]:
                        result["category"] = llm_metadata["category"]
        except Exception as e:
            logger.warning(f"Error in deep preprocessing: {e}")
        
        return result


class RetrievalProcessor:
    """Post-processes retrieved memories from LTM"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
    
    def process(self, results: List[Dict], context: Optional[str] = None) -> List[Dict]:
        if not results:
            return []
        
        if not context:
            return results
        
        context_embedding = self.memory_system.embedding_manager.get_embedding(context)
        
        for result in results:
            content = result.get("content", "")
            content_embedding = self.memory_system.embedding_manager.get_embedding(content)
            
            context_similarity = np.dot(context_embedding, content_embedding) / (
                np.linalg.norm(context_embedding) * np.linalg.norm(content_embedding)
            )
            
            original_score = result.get("score", DEFAULT_SCORE_FALLBACK)
            result["context_score"] = float(context_similarity)
            result["score"] = ORIGINAL_WEIGHT * original_score + CONTEXT_WEIGHT * context_similarity
        
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return results 