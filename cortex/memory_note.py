from typing import List, Dict, Optional
import uuid
from datetime import datetime

class MemoryNote:
    
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        

        self.content = content
        self.id = id or str(uuid.uuid4())
        

        self.keywords = keywords or []
        self.links = links or {}
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        

        current_time = datetime.now().astimezone().isoformat()
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        

        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

        # User and session data
        self.user_id = user_id
        self.session_id = session_id 