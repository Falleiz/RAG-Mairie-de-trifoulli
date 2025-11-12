# app.py
import streamlit as st 
from MistralChat import ask_question_with_mistral  # Assuming the file is renamed or as provided
import os

import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # Pour le développement local
except ImportError:
    pass  # Streamlit Cloud utilise ses propres secrets

# Récupérer la clé
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY")

st.set_page_config(page_title="Assistant Municipal", page_icon="🏛️")
st.title("🏛️ Assistant Municipal de Triffouillis-sur-Loire")
st.subheader("Posez vos questions sur les projets, budgets, événements municipaux")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input(placeholder="Votre question : Ex: Quel est le budget éducation 2024 ?")

if question:
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate response
    with st.spinner("Recherche en cours ..."):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response, chunks = ask_question_with_mistral(question, k=4)
            
            # For streaming: Since the original uses invoke, we'll simulate accumulation here.
            # If you modify the chain to use .stream, you can iterate over tokens.
            # For now, display the full response once ready.
            if response and chunks:
                sources_str = "\n\n**Sources :** " + ", ".join([c["source"] for c in chunks])
                full_response = response + sources_str
                message_placeholder.markdown(full_response)
                
                # Append assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Limit to last 10 conversations (20 messages: 10 user + 10 assistant)
                if len(st.session_state.messages) > 20:
                    st.session_state.messages = st.session_state.messages[-20:]
            else:
                message_placeholder.markdown("Désolé, je n'ai pas pu trouver de réponse.")