# debug_collection.py
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

store = "./vector_store"  # <-- your path
coll_name = "pdf_openai"  # <-- or whatever you used

client = chromadb.PersistentClient(path=store, settings=Settings(allow_reset=True))
ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-large"))
coll = client.get_or_create_collection(name=coll_name, embedding_function=ef)

print("COUNT:", coll.count())

if coll.count() > 0:
    got = coll.get(include=["documents","metadatas"], limit=3)
    print("SAMPLE IDS:", got["ids"])
    for doc, md in zip(got["documents"], got["metadatas"]):
        print("----")
        print("DOC:", (doc[:200] + "…") if doc and len(doc)>200 else doc)
        print("META:", md)
