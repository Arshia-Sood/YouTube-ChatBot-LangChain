import os
import warnings

warnings.filterwarnings('ignore')

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

video_id = 'MfLQZ0n2l2w'

try:
    yt_api = YouTubeTranscriptApi()

    transcript_list = yt_api.fetch(video_id, languages= ["en-US"])

    transcript = " ".join(
        snippet.text for snippet in transcript_list
    )


except TranscriptsDisabled:
    print("No captions available for this video")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.create_documents([transcript])

len(chunks)

embeddings = GoogleGenerativeAIEmbeddings(model= 'gemini-embedding-2')

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 4})

retriever.invoke('Who is little girl')

model = ChatGoogleGenerativeAI(model = 'gemini-3.6-flash')

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

def format_docs(retrieved_docs):
    context_text = '\n\n'.join(doc.page_content for doc in retrieved_docs)

    return context_text

parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

print(main_chain.invoke('summarize the video'))

