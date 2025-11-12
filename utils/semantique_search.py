from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
VECTORSTORE_PATH = Path(__file__).parent.parent / "vectorstore"

def retrieve_relevant_chunks(question: str, k: int = 5):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    # Après avoir chargé vector_store
    results = vector_store.similarity_search_with_score(
        query=question,
        k=k
    )

    relevant_chunks = []
    for doc, score in results:
        relevant_chunks.append({
            "chunk": doc.page_content,
            "source": doc.metadata.get("source", "inconnu"),
            "file_type": doc.metadata.get("file_type", "inconnu"),
            "similarity_score": float(score)  # distance (plus petit = plus similaire)
        })
    
    return relevant_chunks



"""if __name__ == "__main__":
    question = "Quel est le budget alloué à l'éducation en 2024 ?"
    chunks = retrieve_relevant_chunks(question, k=3)
    
    print(f"🔍 Résultats pour la question : '{question}'\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} (score: {chunk['similarity_score']:.4f}) ---")
        print(f"Source : {chunk['source']}")
        print(f"Texte : {chunk['chunk'][:200]}...\n")"""