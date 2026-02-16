import gradio as gr
import tempfile
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedd = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = None


def process_pdf(pdf_file):
    global vectorstore

    if pdf_file is None:
        return "Please upload a PDF."

    loader = PyPDFLoader(pdf_file.name)  # <-- use path directly
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embedd)

    return "PDF processed successfully."


def chat_fn(message, history):
    global vectorstore

    if vectorstore is None:
        return "Please upload a PDF first."

    docs = vectorstore.similarity_search(message, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are an aviation assistant. Answer the question using ONLY the provided context.
If the answer is not present in the context, say “This information is not available in the provided document(s).”
if the question is not related to aviation, say "I can only answer aviation-related questions."
if greetings are detected, respond with a greeting."
if they are going to end the conversation, respond with a farewell."
    

Question:
{message}

Context:
{context}
"""

    response = model.invoke(prompt)
    return response.content


with gr.Blocks() as demo:
    gr.Markdown("# Aviation RAG Chatbot")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Process PDF")
        status = gr.Textbox()

    upload_btn.click(process_pdf, inputs=pdf_input, outputs=status)

    gr.ChatInterface(
        fn=chat_fn,
        title="Chat with your PDF",
    )

demo.launch()