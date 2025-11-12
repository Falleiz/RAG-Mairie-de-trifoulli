import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

def convert_table_to_structured_chunks(md_table: str, source: str):
    """
    Convertit un tableau Markdown en chunks du type:
    "Colonne1: Valeur1\nColonne2: Valeur2"
    """
    lines = [line.strip() for line in md_table.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return [], []
    
    # Extraire les en-têtes (1ère ligne)
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    # Sauter la ligne de séparation (2ème ligne)
    data_rows = lines[2:]
    
    chunks, metas = [], []
    for row in data_rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if len(cells) == len(headers):
            # Créer une ligne structurée
            structured_lines = [f"{header}: {cell}" for header, cell in zip(headers, cells)]
            chunk = "\n".join(structured_lines)
            chunks.append(chunk)
            metas.append({"source": source, "file_type": ".csv"})
    return chunks, metas

def chunk_dataframe_with_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Chunk les fichiers non-CSV avec RecursiveCharacterTextSplitter
    - Convertit les fichiers .csv en chunks structurés (header + ligne)
    - Retourne un seul DataFrame combiné
    """
    all_chunks = []
    all_sources = []
    all_types = []
    
    # 🔹 1. Traitement des fichiers CSV
    df_csv = df[df["text_type"] == ".csv"]
    for _, row in df_csv.iterrows():
        text = row["text_content"]
        source = row["text_source"]
        if isinstance(text, str) and len(text.strip()) > 0:
            chunks, metas = convert_table_to_structured_chunks(text, source)
            for chunk, meta in zip(chunks, metas):
                if len(chunk) >= 20:  # Filtrer les chunks trop courts
                    all_chunks.append(chunk)
                    all_sources.append(meta["source"])
                    all_types.append(meta["file_type"])
    
    # 🔹 2. Traitement des autres fichiers (PDF, DOCX, etc.)
    df_non_csv = df[df["text_type"] != ".csv"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n# ", "\n## ", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len
    )
    
    for _, row in df_non_csv.iterrows():
        text = row["text_content"]
        source = row["text_source"]
        file_type = row["text_type"]
        
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
            
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) >= 20:
                all_chunks.append(chunk)
                all_sources.append(source)
                all_types.append(file_type)
    
    # 🔹 3. Combiner dans un seul DataFrame
    return pd.DataFrame({
        "chunk": all_chunks,
        "source": all_sources,
        "file_type": all_types
    })


if __name__ == "__main__":
    # Charger les données extraites
    df = pd.read_csv("../extracted_data/final_extracted_text.csv")
    
    # Générer les chunks (CSV + autres)
    chunked_df = chunk_dataframe_with_csv(df)
    
    # Sauvegarder
    chunked_df.to_csv("../extracted_data/chunk.csv", index=False)
    print(f"✅ {len(chunked_df)} chunks sauvegardés dans ../extracted_data/chunk.csv")