import os 
from pathlib import Path
from docling.document_converter import DocumentConverter
import logging
import torch
import whisper
import pandas as pd 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def extracting_file_path(inputs_dir) :
    document_path=[]
    for (root ,dirs,files) in os.walk(inputs_dir,topdown=True):
        for file in files : 
            chemin=os.path.join(root,file)
            document_path.append(chemin)
    
    return document_path



def text_extraction(document_paths):
    converter = DocumentConverter()
    ignore=0
    document_paths_lenght=len(document_paths)
    documents_text={
        "text_content": [],
        "text_source" : [],
        "text_type" : []
    }
    supported_extensions = {
        # Documents
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', 
        '.xlsx', '.xls', '.csv', '.txt', '.rtf', '.md',
        # Images (Docling fait l'OCR)
        '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif',
        
        
    }
    for path in document_paths : 
        _,extension =os.path.splitext(path)
        if extension not in supported_extensions :
            print(f"Fichier avec extension en .{extension} n'est pas supporter")
            continue
        try : 
            print(f"extraction du dossier {path}")
            result=converter.convert(Path(path)).document.export_to_markdown()
            documents_text['text_content'].append(result)
            documents_text['text_type'].append(extension)
            documents_text['text_source'].append(path)

        except  Exception as e : 
            print(f"erreur lor de l'ectraction du document {path} :{e}")
            ignore+=1
    print(f"total doc to convert : {document_paths_lenght}\n Doc ignored : {ignore}")
        
    return  documents_text

        


def extract_text_from_audio(extracted_path): 
    documents_text={
        "text_content": [],
        "text_source" : [],
        "text_type" : []
    }

    model_name='base'
    ignore=0
     
    supported_extensions = { 
        '.wav', '.mp3', '.m4a', '.flac', '.aac',
       
    }
     
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🎙️ Chargement du modèle Whisper '{model_name}' sur {device}...")
        
        # Charger le modèle (une seule fois en pratique)
    model = whisper.load_model(model_name, device=device)

    for path in extracted_path : 
        _,extension =os.path.splitext(path)
        if extension not in supported_extensions :
            print(f"Fichier avec extension en .{extension} n'est pas supporter")
            ignore+=1
            continue

        try : 
            print(f"extraction du dossier {path}")
            result = model.transcribe(path,fp16=(device == "cuda")   )
        
            text = result["text"].strip()
            documents_text['text_content'].append(text)
            documents_text['text_type'].append(extension)
            documents_text['text_source'].append(path)

        except  Exception as e : 
            print(f"erreur lor de l'ectraction du document {path} :{e}")
            ignore+=1
    print(f" Doc ignored : {ignore}")
        
    return  documents_text


if __name__ =="__main__" :
    inputs_dir="../inputs"
    document_path=extracting_file_path(inputs_dir)
    document_text_from_text_dict=text_extraction(document_path)
    print(document_text_from_text_dict)
    df1=pd.DataFrame(document_text_from_text_dict)
    print(df1.head())
    document_text_from_audio_dict=extract_text_from_audio(document_path)
    print(document_text_from_audio_dict)
    df2=pd.DataFrame(document_text_from_audio_dict)
    print(df2.head())
    df=pd.concat([df1,df2],ignore_index=True)
    output_dir = "../extracted_data"
    os.makedirs(output_dir, exist_ok=True)  
    df.to_csv(os.path.join(output_dir, "final_extracted_text.csv"), index=False)

