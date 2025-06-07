from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import os, shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from chromadb import Client

import fitz  # PyMuPDF

UPLOAD_DIR = "./uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
collection = None
embedding_function = OllamaEmbeddings(model="deepseek-r1")

def read_pdf(filepath):
    text = ""
    doc = fitz.open(filepath)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = read_pdf(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    global collection
    client = Client(Settings())
    try:
        client.delete_collection(name="ragdb")
    except Exception as e:
        print(f"Ignore error when deleting collection: {e}")
    collection = client.create_collection(name="ragdb")
    embeddings = [embedding_function.embed_query(doc.page_content) for doc in docs]
    for idx, doc in enumerate(docs):
        collection.add(
            documents=[doc.page_content],
            metadatas=[{'id': idx}],
            embeddings=[embeddings[idx]],
            ids=[str(idx)]
        )
    return {"status": "ok", "detail": f"File {file.filename} processed!"}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return JSONResponse({"error": "Missing question."}, status_code=400)
    if collection is None:
        return JSONResponse({"error": "No document loaded. Please upload a file first."}, status_code=400)
    retriever = Chroma(collection_name="ragdb", client=collection._client, embedding_function=embedding_function).as_retriever()
    results = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Question: {question}\n\nContext: {context}"
    answer = f"(Demo) Based on the document, relevant content: {context[:300]}..."  # Replace with real LLM call
    return {"answer": answer}
