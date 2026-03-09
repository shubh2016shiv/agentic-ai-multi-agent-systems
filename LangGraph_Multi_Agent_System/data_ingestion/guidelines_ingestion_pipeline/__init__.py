"""
Guidelines Ingestion Pipeline.

Production-grade PDF ingestion system for medical clinical guidelines, implementing
parent-child chunking, figure description via vision LLMs, 3-layer deduplication,
and idempotent ingestion with granular failure recovery.

Architecture:
    - domain/ — Pure business logic (zero I/O)
    - infrastructure/ — Concrete I/O implementations (parsers, embedders, stores)
    - application/ — Composition root and entry points
    - config/ — Pydantic settings
    - exceptions/ — Custom exception hierarchy
    - utils/ — Pure utility functions
"""
