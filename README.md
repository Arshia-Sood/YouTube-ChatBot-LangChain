# YouTube RAG

A **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about the content of a YouTube video.

The application retrieves the video's transcript, splits it into smaller chunks, converts the chunks into vector embeddings, stores them in a FAISS vector database, and retrieves the most relevant chunks to provide context to a Gemini LLM for generating answers.

## How It Works

```text
YouTube Video
      ↓
YouTube Transcript
      ↓
Text Splitting
      ↓
Google Gemini Embeddings
      ↓
FAISS Vector Store
      ↓
Similarity Search
      ↓
Relevant Transcript Chunks
      ↓
Prompt + Context
      ↓
Gemini LLM
      ↓
Generated Answer
```

## RAG Pipeline

### 1. Transcript Extraction

The application uses `youtube-transcript-api` to retrieve the transcript of a YouTube video.

### 2. Text Chunking

The transcript is divided into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

```python
chunk_size = 1000
chunk_overlap = 200
```

The overlap helps preserve context between consecutive chunks.

### 3. Embeddings

Each text chunk is converted into a vector representation using Google's Gemini embedding model.

### 4. Vector Store

The generated embeddings are stored using **FAISS**, which allows efficient similarity-based retrieval.

### 5. Retrieval

When a user asks a question, the retriever performs a similarity search and retrieves the most relevant transcript chunks.

The current implementation retrieves the top 4 chunks.

### 6. Augmentation

The retrieved chunks are formatted into a context and passed to the LLM along with the user's question.

### 7. Generation

Google Gemini generates the final answer using only the retrieved transcript context.

If the retrieved context does not contain enough information, the prompt instructs the model to respond that it does not know.

## Technologies Used

* Python
* LangChain
* LangChain Core
* LangChain Community
* LangChain Text Splitters
* Google Gemini
* Google Generative AI Embeddings
* FAISS
* YouTube Transcript API
* python-dotenv

## Project Structure

```text
YT-RAG/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── README.md
├── .gitignore
│
└── src/
    └── yt_rag/
        └── __init__.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/youtube-rag.git
cd youtube-rag
```

### 2. Install dependencies

This project uses `uv`.

```bash
uv sync
```

### 3. Add your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not commit the `.env` file to GitHub.

### 4. Run the application

```bash
uv run main.py
```

## Example

The application can be used to ask questions about the transcript of a YouTube video, for example:

```text
Question: Why was CID called?

Answer: ...
```

Other questions can be asked such as:

```text
Who is the little girl?
What happened to the victim?
Who called CID?
What happened at the end?
```

## Key Concepts Demonstrated

This project demonstrates the core components of a RAG pipeline:

* Document loading
* Text splitting
* Embeddings
* Vector databases
* Similarity search
* Retrieval
* Prompt construction
* Context augmentation
* LLM-based generation
* LangChain Runnable pipelines

## Limitations

The current implementation retrieves only the top 4 most relevant transcript chunks for a question. Therefore, it is primarily designed for **question answering over the video transcript**, rather than generating a complete summary of the entire video.

The quality of the answers also depends on the availability and quality of the video's transcript.

## Future Improvements

* Support more transcript languages
* Add a user interface using Streamlit
* Allow users to enter any YouTube URL
* Improve retrieval using MMR or hybrid search
* Add metadata filtering
* Implement conversational memory
* Add better handling for videos without transcripts
* Implement a dedicated long-document summarization pipeline


## 👩‍💻 Author
Arshia Sood Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: https://github.com/Arshia-Sood

⭐ If you like this project, give it a star!