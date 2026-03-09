import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from pymongo import MongoClient
client = MongoClient("mongodb://admin:adminpassword@localhost:27017")
db = client["medical_knowledge_base"]
jobs = list(db["ingestion_jobs"].find({}, {"pdf_name": 1, "status": 1, "total_chunks": 1, "_id": 0}))
print(f"\nMongoDB ingestion_jobs ({len(jobs)} total):")
for j in sorted(jobs, key=lambda x: x["pdf_name"]):
    print(f"  {j['pdf_name']:<35} {j['status']:<12} chunks={j.get('total_chunks', 0)}")

import chromadb
chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("medical_guidelines_v1")
print(f"\nChromaDB 'medical_guidelines_v1': {col.count()} vectors stored\n")
