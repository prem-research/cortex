
# Processing constants
MAX_KEYWORDS = 5
DEFAULT_SCORE_FALLBACK = 0.5
CONTEXT_WEIGHT = 0.7
ORIGINAL_WEIGHT = 0.3

# Memory system constants  
LARGE_CONTENT_THRESHOLD = 10000
MIN_CONNECTION_STRENGTH = 0.65
DEFAULT_LTM_WORKERS = 1
DEFAULT_STM_CAPACITY = 20

# Model defaults
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'  # OpenAI by default
# Alternative local models: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_BACKEND = "openai"

OPENAI_EMBEDDING_MODELS = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536
}

LOCAL_EMBEDDING_MODELS = {
    'all-MiniLM-L6-v2': 384,
    'all-mpnet-base-v2': 768,
    'sentence-transformers/all-MiniLM-L6-v2': 384,
    'sentence-transformers/all-mpnet-base-v2': 768
}

def is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI embedding model"""
    return model_name in OPENAI_EMBEDDING_MODELS

def is_local_model(model_name: str) -> bool:
    """Check if the model is a local SentenceTransformers model"""
    return model_name in LOCAL_EMBEDDING_MODELS

def get_embedding_dimension(model_name: str) -> int:
    """ embedding dimension for any supported model"""
    if is_openai_model(model_name):
        return OPENAI_EMBEDDING_MODELS[model_name]
    elif is_local_model(model_name):
        return LOCAL_EMBEDDING_MODELS[model_name]
    else:
        # Default for unknown models
        return 1536

# Search defaults
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_RELATED_MEMORIES_COUNT = 5

# ChromaDB configuration
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8003
DEFAULT_CHROMA_URI = f"http://{DEFAULT_CHROMA_HOST}:{DEFAULT_CHROMA_PORT}"

# Collection management
COLLECTION_CREATION_THRESHOLD = 10
TOP_COLLECTIONS_FOR_SEARCH = 4
COLLECTION_SIMILARITY_WEIGHT = 0.3
ITEM_SIMILARITY_WEIGHT = 0.7

# Collection update timing (in seconds)
MIN_UPDATE_INTERVAL = 3600  # 1 hour - minimum time between metadata updates
FORCE_UPDATE_INTERVAL = 86400  # 24 hours - force update regardless of growth

# Collection growth thresholds
SIGNIFICANT_GROWTH_RATIO = 1.5  # 50% growth triggers significant growth
SIGNIFICANT_GROWTH_ABSOLUTE = 25  # 25+ new memories also triggers significant growth
METADATA_UPDATE_THRESHOLD = 1.2  # 20% growth required for metadata update

# Sample sizes for collection metadata generation
COLLECTION_METADATA_SAMPLE_SIZE = 20  # Max memories to sample for metadata generation
COLLECTION_CONTENT_SAMPLE_SIZE = 5  # Max content samples to include in LLM prompt
MIN_MEMORIES_FOR_COLLECTION_CREATION = 5  # Min memories needed from cache for collection creation

# Common words for keyword filtering
COMMON_WORDS = {
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'and', 
    'or', 'but', 'is', 'are', 'was', 'were'
} 