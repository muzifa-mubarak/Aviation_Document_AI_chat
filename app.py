from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
import os
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

embedd = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)


@app.post("/ask")
async def ask_pdf(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    # Save uploaded pdf temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embedd)

    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY using the provided context.
If the answer is not present, say information not available.

Question:
{question}

Context:
{context}
"""

    response = model.invoke(prompt)

    return {"answer": response.content}






