import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# Streamlit UI
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:  # Only append non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_cohere.pkl"
main_placeholder = st.empty()

# Define custom prompt template for better results
custom_prompt_template = """
You are a financial research assistant that helps users analyze news articles.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite your sources at the end of your answer.

Context: {context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

# Initialize Cohere LLM
llm = Cohere(
    temperature=0.0,  # Reduced temperature for more factual responses
    max_tokens=500,   # Increased token limit for more comprehensive answers
    cohere_api_key=cohere_api_key
)

# Process URLs and create vector store
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL")
    else:
        try:
            with st.spinner("Processing URLs..."):
                # Create progress indicators
                progress_text = st.empty()
                
                # Load data from URLs
                progress_text.text("Loading data from URLs...")
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                
                # Split text into chunks
                progress_text.text("Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", ","],
                    chunk_size=1000,
                    chunk_overlap=200  # Added overlap to improve context preservation
                )
                docs = text_splitter.split_documents(data)
                
                # Create embeddings
                progress_text.text("Creating embeddings...")
                embeddings = CohereEmbeddings(
                    cohere_api_key=cohere_api_key,
                    model="embed-english-v3.0",
                    user_agent="news-research-tool/1.0"
                )
                
                # Create vector store
                vectorstore = FAISS.from_documents(docs, embeddings)
                pkl = vectorstore.serialize_to_bytes()
                
                # Save vector store
                progress_text.text("Saving vector store...")
                with open(file_path, "wb") as f:
                    pickle.dump(pkl, f)
                
                progress_text.text("Processing complete! ✅")
                st.sidebar.success(f"Processed {len(docs)} text chunks from {len(urls)} URLs")
        except Exception as e:
            st.sidebar.error(f"Error processing URLs: {str(e)}")

# Query section
st.subheader("Ask questions about your articles")
query = st.text_input("Your question:")

if query:
    if os.path.exists(file_path):
        try:
            with st.spinner("Searching for answer..."):
                # Load vector store
                with open(file_path, "rb") as f:
                    pkl = pickle.load(f)
                
                # Initialize embeddings
                embeddings = CohereEmbeddings(
                    cohere_api_key=cohere_api_key,
                    model="embed-english-v3.0",
                    user_agent="news-research-tool/1.0"
                )
                
                # Deserialize vector store
                vectorstore = FAISS.deserialize_from_bytes(
                    embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True
                )
                
                # Create retrieval chain with custom prompt
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Retrieve more documents for better context
                )
                
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                # Execute query
                result = chain({"query": query})
                
                # Display results
                st.header("Answer")
                st.write(result["result"])
                
                # Display sources
                st.subheader("Sources")
                sources = result["source_documents"]
                used_sources = set()
                
                for i, doc in enumerate(sources):
                    source_url = doc.metadata.get('source', 'Unknown source')
                    if source_url not in used_sources:
                        used_sources.add(source_url)
                        st.write(f"{len(used_sources)}. {source_url}")
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
    else:
        st.warning("Please process some URLs first before asking questions.")
else:
    st.info("Enter a question above to search your processed news articles.")
