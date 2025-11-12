import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Nouvel import

# Charger les chunks
df = pd.read_csv("../extracted_data/chunk.csv")

# Préparer les métadonnées
metadatas = [
    {"source": row["source"], "file_type": row["file_type"]}
    for _, row in df.iterrows()
]

# ✅ Utiliser le bon import
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Créer la base vectorielle
vector_store = Chroma(
    collection_name="municipal_documents",
    embedding_function=embeddings,
    persist_directory="../chroma_langchain_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Ajouter les documents si la base est vide
if len(vector_store.get()["ids"]) == 0:
    vector_store.add_texts(
        texts=df["chunk"].tolist(),
        metadatas=metadatas
    )
    print(f"✅ {len(df)} chunks vectorisés et sauvegardés.")

# Après avoir chargé vector_store
print(f"📊 Nombre de documents dans la base : {len(vector_store.get()['ids'])}")