from fastapi import FastAPI,File,UploadFile
import os
from PyPDF2 import PdfFileReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer

app=FastAPI()

@app.post('/uploadfile/')
async def upload_file(files:list[UploadFile]):
    if len(files)>5:
        return {'error':'You can only upload 5 files at a time'}
    file_paths=[]
    for file in files:
        file_path=f'./{file.filename}'
        with open(file_path,'wb') as f:
            f.write(file.file.read())
            file_paths.append(file_path)
            return {'filepaths':file_paths}
        

def extract_text_from_pdf(pdf_path):
    reader=PdfFileReader(pdf_path)
    text=''
    for page in reader.pages:
        text+=page.extract_text()
        return text
    

def chuck_text(text,chunk_size=1000,overlap=100):
    chunks=[]
    for i in range(0,len(text),chunk_size-overlap):
        chunks.append(text[i:i+chunk_size])
        return chunks
    

model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def generate_embeddings(chunks):
    embeddings=model.encode(chunks,convert_to_tensor=True)
    return embeddings.numpy()


def create_faiss_index(embeddings):
    index=faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_similer_chunks(query,chunks,index,top_k=3):
    query_embedding=model.encode([query],convert_to_tensor=True).to_numpy()
    distances,indices=index.search(query_embedding,top_k)
    return [chunks[i] for i in indices[0]]


tokenizer=LlamaTokenizer.from_pretrained('llama-2-7b-hf')
model=LlamaForCausalLM.from_pretrained('llama-2-7b-hf',device_map='auto')
def generate_response(prompt):
    inputs=tokenizer(prompt,return_tensors='pt')
    outputs=model.generate(inputs.input_ids,max_length=500,temperature=0.7)
    return tokenizer.decode(outputs[0],skip_special_tokens=True)


def answer_question(query,index,chunks):
    relevant_chunks=retrieve_similer_chunks(query,chunks,index)
    context=' '.join(relevant_chunks)
    prompt=f'Question: {query} Context: {context}'
    response=generate_response(prompt)