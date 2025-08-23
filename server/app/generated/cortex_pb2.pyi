from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StoreMemoryRequest(_message.Message):
    __slots__ = ("content", "context", "tags", "timestamp", "user_id", "session_id", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    content: str
    context: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    timestamp: str
    user_id: str
    session_id: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, content: _Optional[str] = ..., context: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., timestamp: _Optional[str] = ..., user_id: _Optional[str] = ..., session_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StoreMemoryResponse(_message.Message):
    __slots__ = ("id", "success", "message")
    ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    id: str
    success: bool
    message: str
    def __init__(self, id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ...) -> None: ...

class SearchMemoryRequest(_message.Message):
    __slots__ = ("query", "limit", "memory_source", "context", "tags", "exclude_content", "include_links", "apply_postprocessing", "user_id", "session_id", "temporal_weight", "date_range")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_CONTENT_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_LINKS_FIELD_NUMBER: _ClassVar[int]
    APPLY_POSTPROCESSING_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    DATE_RANGE_FIELD_NUMBER: _ClassVar[int]
    query: str
    limit: int
    memory_source: str
    context: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    exclude_content: bool
    include_links: bool
    apply_postprocessing: bool
    user_id: str
    session_id: str
    temporal_weight: float
    date_range: str
    def __init__(self, query: _Optional[str] = ..., limit: _Optional[int] = ..., memory_source: _Optional[str] = ..., context: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., exclude_content: bool = ..., include_links: bool = ..., apply_postprocessing: bool = ..., user_id: _Optional[str] = ..., session_id: _Optional[str] = ..., temporal_weight: _Optional[float] = ..., date_range: _Optional[str] = ...) -> None: ...

class Memory(_message.Message):
    __slots__ = ("id", "content", "context", "tags", "keywords", "timestamp", "score", "is_linked", "memory_tier", "collection_name", "category", "composite_score", "relationship_type", "relationship_strength", "relationship_reason")
    ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    IS_LINKED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_TIER_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    COMPOSITE_SCORE_FIELD_NUMBER: _ClassVar[int]
    RELATIONSHIP_TYPE_FIELD_NUMBER: _ClassVar[int]
    RELATIONSHIP_STRENGTH_FIELD_NUMBER: _ClassVar[int]
    RELATIONSHIP_REASON_FIELD_NUMBER: _ClassVar[int]
    id: str
    content: str
    context: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    keywords: _containers.RepeatedScalarFieldContainer[str]
    timestamp: str
    score: float
    is_linked: bool
    memory_tier: str
    collection_name: str
    category: str
    composite_score: float
    relationship_type: str
    relationship_strength: float
    relationship_reason: str
    def __init__(self, id: _Optional[str] = ..., content: _Optional[str] = ..., context: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., keywords: _Optional[_Iterable[str]] = ..., timestamp: _Optional[str] = ..., score: _Optional[float] = ..., is_linked: bool = ..., memory_tier: _Optional[str] = ..., collection_name: _Optional[str] = ..., category: _Optional[str] = ..., composite_score: _Optional[float] = ..., relationship_type: _Optional[str] = ..., relationship_strength: _Optional[float] = ..., relationship_reason: _Optional[str] = ...) -> None: ...

class SearchMemoryResponse(_message.Message):
    __slots__ = ("memories", "count", "query_metadata")
    class QueryMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MEMORIES_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    QUERY_METADATA_FIELD_NUMBER: _ClassVar[int]
    memories: _containers.RepeatedCompositeFieldContainer[Memory]
    count: int
    query_metadata: _containers.ScalarMap[str, str]
    def __init__(self, memories: _Optional[_Iterable[_Union[Memory, _Mapping]]] = ..., count: _Optional[int] = ..., query_metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class DeleteMemoryRequest(_message.Message):
    __slots__ = ("memory_id", "user_id", "session_id")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    user_id: str
    session_id: str
    def __init__(self, memory_id: _Optional[str] = ..., user_id: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class DeleteMemoryResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class UpdateMemoryRequest(_message.Message):
    __slots__ = ("memory_id", "content", "context", "tags", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    content: str
    context: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, memory_id: _Optional[str] = ..., content: _Optional[str] = ..., context: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class UpdateMemoryResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class ClearMemoryRequest(_message.Message):
    __slots__ = ("memory_source", "user_id", "session_id")
    MEMORY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    memory_source: str
    user_id: str
    session_id: str
    def __init__(self, memory_source: _Optional[str] = ..., user_id: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class ClearMemoryResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class GetMemoryRequest(_message.Message):
    __slots__ = ("memory_id", "user_id", "session_id")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    user_id: str
    session_id: str
    def __init__(self, memory_id: _Optional[str] = ..., user_id: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class GetMemoryResponse(_message.Message):
    __slots__ = ("memory", "found")
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    FOUND_FIELD_NUMBER: _ClassVar[int]
    memory: Memory
    found: bool
    def __init__(self, memory: _Optional[_Union[Memory, _Mapping]] = ..., found: bool = ...) -> None: ...
