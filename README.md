# Offline_RAG_Study_Assistant
A fully offline RAG pipeline for document-grounded quiz generation and question answering using local embeddings and LLM inference.

## Folder Structure

```bash
offline-rag-quiz-generator/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── routes_upload.py
│   │   │   ├── routes_quiz.py
│   │   │   └── routes_health.py
│   │   ├── core/
│   │   │   ├── document_loader.py
│   │   │   ├── text_cleaner.py
│   │   │   ├── chunker.py
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   ├── retriever.py
│   │   │   └── storage.py
│   │   ├── llm/
│   │   │   ├── ollama_client.py
│   │   │   ├── prompts.py
│   │   │   ├── quiz_generator.py
│   │   │   ├── output_parser.py
│   │   │   └── validators.py
│   │   ├── models/
│   │   │   ├── request_models.py
│   │   │   └── response_models.py
│   │   └── utils/
│   │       ├── logger.py
│   │       └── helpers.py
│   ├── data/
│   │   ├── uploads/
│   │   ├── extracted/
│   │   ├── processed/
│   │   └── chroma_db/
│   ├── tests/
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   └── test_quiz_generation.py
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.js
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   ├── QuizControls.jsx
│   │   │   ├── QuizCard.jsx
│   │   │   ├── MCQCard.jsx
│   │   │   ├── ShortAnswerCard.jsx
│   │   │   └── LoadingState.jsx
│   │   ├── pages/
│   │   │   └── Home.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── README.md
├── sample_docs/
│   ├── os_notes.pdf
│   ├── dbms_notes.pdf
│   └── cn_notes.pdf
├── docs/
│   ├── api_contract.md
│   ├── demo_script.md
│   └── architecture.md
└── .gitignore
```
