# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Your API key
os.environ["OPENAI_API_KEY"] = api_key

# Set up your PDF processing components here
# pdf_path = "goz-kommentar-bzaek_merged.pdf"
loader=DirectoryLoader('./text/',glob = "**/*.txt")
docs=loader.load()
# loader = PyPDFLoader(pdf_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
# pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=".")
vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# llm = ChatOpenAI(model_name='gpt-4')
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(model_name="gpt-3.5-turbo-16k",temperature=0.8), vectordb.as_retriever(), memory=memory)

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
    app.run(host="0.0.0.0", port=10000)
