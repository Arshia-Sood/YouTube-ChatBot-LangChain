import os
import warnings

warnings.filterwarnings('ignore')

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title= 'YouTube RAG Assistant',
    page_icon = "🎥",
    layout = "centered"
)

st.title(
    "Ask questions about a YouTube video using"
    "Retrieval-Augmented Generation (RAG)."
)

youtube_url = st.text_input(
    "Enter YouTube Video URL",

    placeholder = "https://www.youtube.com/watch?v=..."
)

def extract_video_id(url):
    if 'v=' in url:
        return url.split("v=")[1].split("&")[0]

    elif "youtu.be/" in url: return url.split("youtu.be/")[1].split("?")[0]

    return None

def create_retriever(video_id):
    yt_api = YouTubeTranscriptApi()

    transcript_list = yt_api.fetch(video_id, languages= ["en-US"])

    transcript = " ".join(
        snippet.text for snippet in transcript_list
    )

    splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
    )

    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(model= 'gemini-embedding-2')

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 4})

    return retriever


if st.button("Process Video"):
    if not youtube_url:
        st.warning("Please enter a YouTube URL.")

    else:
        video_id = extract_video_id(youtube_url)

        if video_id is None: 
            st.error("Invalid YouTube URL.")

        else:
            with st.spinner("Processing video..."):
                try:
                    retriever = create_retriever(video_id)
                    st.session_state.retriever = retriever
                    st.session_state.video_processed = True
                    st.success("Video processed successfully!")

                except TranscriptsDisabled:
                    st.error("No captions are available for this video.")

                except Exception as e:
                    st.error(f"Error processing video: {e}")


if st.session_state.get("video_processed", False):
    st.divider()

    question = st.text_input(
        "Ask a question about the video",
        placeholder= "Who was the little girl?"
    )

    if st.button("Ask"):
        if not question:
            st.warning("Please enter a question.")

        else:
            retriever = st.session_state.retriever

            def format_docs(retrieved_docs):
                return '\n\n'.join(doc.page_content for doc in retrieved_docs)

            prompt = PromptTemplate(
                template = """
                You are a helpful assistant.
                Answer only from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,

                input_variables= ['context', 'question']
            )

            model = ChatGoogleGenerativeAI(model = 'gemini-3.6-flash')

            parallel_chain = RunnableParallel({
            'context' : retriever | RunnableLambda(format_docs),
            'question' : RunnablePassthrough()
            })

        parser = StrOutputParser()

        main_chain = parallel_chain | prompt | model | parser

        with st.spinner("Generating answer..."):
            try:
                answer = main_chain.invoke(question)
                st.subheader("Answer")
                st.write(answer)

            except Exception as e:
                st.error( f"Error generating answer: {e}" )

