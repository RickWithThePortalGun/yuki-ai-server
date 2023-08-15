__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

app = Flask(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Your API key
os.environ["OPENAI_API_KEY"] = api_key

# Set up your PDF processing components here
pdf_path = "goz-kommentar-bzaek.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory
)

@app.route("/get_answer", methods=["POST", "GET"])
def get_answer():
    if request.method == "GET":
        return jsonify({"message": "Hello, I am Yuki's AI"})

    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    result = pdf_qa({"question": question})
    answer = result["answer"]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
