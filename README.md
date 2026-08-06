# 🚀 FullStack RAG Explorer

> A production-ready Retrieval-Augmented Generation (RAG) application that enables users to upload documents, retrieve relevant context using vector search, and interact with them through a real-time AI chat interface powered entirely by local models.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# 📌 Overview

FullStack RAG Explorer is an end-to-end Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **React**, **LangChain**, **ChromaDB**, and **Ollama**.

Users can upload PDF or Markdown documents, which are automatically parsed, chunked, embedded using a local HuggingFace model, and stored in ChromaDB. During conversations, the application retrieves the most relevant chunks and sends them as context to a locally running LLM through Ollama, producing grounded responses with source citations.

The entire application works **without OpenAI APIs or paid cloud services**, making it ideal for learning, experimentation, and private document question answering.

---

# ✨ Features

- 📄 PDF & Markdown document ingestion
- ✂️ Intelligent document chunking
- 🧠 Local HuggingFace embeddings
- 📚 Persistent ChromaDB vector storage
- 🔍 Semantic similarity search
- 🤖 Local LLM inference using Ollama
- ⚡ Real-time streaming AI responses
- 📌 Source citations with page/line numbers
- 🌐 React frontend
- 🚀 FastAPI backend
- 🔄 WebSocket-based streaming
- 💻 Completely offline after model download

---

# 🏗️ System Architecture

```
                   +----------------------+
                   |      React UI        |
                   +----------+-----------+
                              |
                       WebSocket / REST
                              |
                   +----------v-----------+
                   |      FastAPI API     |
                   +----------+-----------+
                              |
          +-------------------+-------------------+
          |                                       |
          |                               Document Upload
          |                                       |
          |                                 Loader Pipeline
          |                                       |
          |                                 Text Splitter
          |                                       |
          |                            HuggingFace Embeddings
          |                                       |
          |                                 ChromaDB Vector Store
          |                                       |
          +-------------------+-------------------+
                              |
                       Similarity Search
                              |
                      Prompt Construction
                              |
                          Ollama (LLM)
                              |
                    Streaming AI Response
                              |
                          React Frontend
```

---

# 🛠 Tech Stack

### Frontend

- React
- Vite
- Redux Toolkit
- WebSocket API

### Backend

- FastAPI
- LangChain
- ChromaDB
- Ollama
- HuggingFace Embeddings
- Pydantic

### AI / RAG

- sentence-transformers
- Chroma Vector Database
- Local LLM (Ollama)

---

# 📂 Project Structure

```
FullStack-RAG-Explorer
│
├── backend
│   ├── app
│   │   ├── core
│   │   ├── db
│   │   ├── rag
│   │   ├── routes
│   │   ├── scripts
│   │   └── main.py
│   ├── requirements.txt
│   └── README.md
│
├── frontend
│   ├── public
│   ├── src
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/fullstack-rag-explorer.git

cd fullstack-rag-explorer
```

---

## 2. Backend Setup

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Start FastAPI

```bash
uvicorn app.main:app --reload
```

Backend runs at

```
http://localhost:8000
```

---

## 3. Install Ollama

Download:

https://ollama.com/download

Pull model

```bash
ollama pull llama3
```

Start server

```bash
ollama serve
```

---

## 4. Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend runs at

```
http://localhost:5173
```

---

# 📄 Upload Documents

Supported formats

- PDF
- Markdown

Upload through

```
POST /api/v1/ingest
```

Example

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
-F "file=@document.pdf"
```

---

# 💬 Chat API

WebSocket Endpoint

```
ws://localhost:8000/ws/chat
```

Client Request

```json
{
  "type":"query",
  "payload":{
      "text":"Explain FastAPI architecture"
  }
}
```

Server streams

```json
{
   "type":"token",
   "payload":{
      "text":"FastAPI..."
   }
}
```

Completion

```json
{
    "type":"done"
}
```

---

# 📚 RAG Workflow

1. Upload document
2. Parse PDF / Markdown
3. Split into chunks
4. Generate embeddings
5. Store vectors in ChromaDB
6. User asks question
7. Similarity search retrieves context
8. Prompt created
9. Ollama generates answer
10. Tokens streamed back to frontend

---



# 🚀 Future Improvements

- Authentication
- Conversation History
- Multi-user support
- Hybrid Search
- Re-ranking
- Multiple LLM support
- Docker Deployment
- Cloud Storage
- Citation Preview
- Admin Dashboard

---

# 📖 Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- FastAPI Development
- React Frontend
- Vector Databases
- LangChain Pipelines
- Semantic Search
- WebSockets
- Streaming Responses
- Local AI Deployment
- Prompt Engineering

---

# 🤝 Contributing

Contributions are welcome!

```bash
Fork the repository

Create your feature branch

Commit your changes

Push to the branch

Open a Pull Request
```

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Rohan Das P**

- GitHub: https://github.com/yourusername
- LinkedIn: https://linkedin.com/in/yourprofile

---

⭐ If you found this project useful, consider giving it a star!
