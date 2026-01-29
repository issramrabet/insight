import json
from sentence_transformers import SentenceTransformer as st
import numpy as np
from qdrant_client import QdrantClient as qc
from qdrant_client.models import VectorParams, Distance, PointStruct
f=open("../data/products.json","r")
products=json.load(f)
f.close()
model=st("all-MiniLM-L6-v2")
client=qc(
    url="https://1d32ce5a-6976-4b02-9c2c-92a667ac0abd.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.X-hpyRvPwRFVwpD_ziBw4ZgCwNjgWg10NzlukJNW8Rg"
)
if client.collection_exists(collection_name="products"):
    client.delete_collection(collection_name="products")
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
pts = []
for p in products:
    vector=model.encode(p["description"],normalize_embeddings=True).tolist()
    pts.append(
        PointStruct(
            id=p["id"],
            vector=vector,
            payload=p

        )
    )
client.upsert(
    collection_name="products",
    points=pts
)
command=input("Enter search command: ")
qv=model.encode(command,normalize_embeddings=True).tolist()
results=client.query_points(
    collection_name="products",
    query=qv,
    limit=6
).points
for r in results:
    print(r.payload["name"],r.score)
    
    



