# rag_mistral.py
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.semantique_search import retrieve_relevant_chunks
from langchain_core.callbacks import StreamingStdOutCallbackHandler

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError(" Clé API Mistral non trouvée. Vérifie le fichier .env")

llm = ChatMistralAI(
    model="open-mistral-7b",
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0,
    max_tokens=500,  
    callbacks=[StreamingStdOutCallbackHandler()]  
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant municipal fiable. Réponds en français, "
               "en utilisant UNIQUEMENT le contexte ci-dessous. Si l'info n'est pas présente, dis 'Je ne sais pas.."),
    ("human", "Contexte :\n{context}\n\nQuestion :\n{question}")
])

def ask_question_with_mistral(question: str, k: int = 10):
    chunks = retrieve_relevant_chunks(question, k=k)
    
    if not chunks:
        print(" Aucun contexte trouvé.")
        

    # Formater le contexte
    context = "\n\n".join([
        f"[Source: {chunk['source']}]\n{chunk['chunk']}"
        for chunk in chunks
    ])
    print(f" Contexte récupéré ({len(chunks)} chunks)")

    # Créer la chaîne
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return response,chunks

    

if __name__ == "__main__":
    ask_question_with_mistral("Quels nouveaux aménagements cyclables sont prévus en 2026, et sur quelles voiries ?")