"""
Connection managers for shared infrastructure.

Both the drug and guidelines pipelines depend on these abstractions, not on
raw client constructors. This satisfies the Dependency Inversion Principle:
high-level pipeline code depends on these stable abstractions; low-level
client details (MongoClient, chromadb.PersistentClient) are encapsulated here.
"""

from data_ingestion.connections.mongodb_connection_manager import (
    MongoDBConnectionManager,
)
from data_ingestion.connections.chroma_connection_manager import (
    ChromaConnectionManager,
)

__all__ = [
    "MongoDBConnectionManager",
    "ChromaConnectionManager",
]
