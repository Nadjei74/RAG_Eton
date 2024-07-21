#Buswaybot.py
#import ollama
import streamlit as st
import argparse
from query import query_rag
#import subprocess
#loading of documents
import argparse
import os
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding import get_embedding_function
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import flashrank_rerank

from langchain_community.vectorstores import Chroma


####################################################################
#            Create app interface with streamlit
####################################################################


st.set_page_config(
    page_title="Welcome to the 🤖 Eaton Busway chatbot",
    layout="wide",

)



#########
CHROMA_PATH = "chroma"
DATA_PATH = "Test_Files"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)




def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
##########################################################################


def main():
    ###################
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    print(documents)
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    #############################################



    st.title("🤖 Busway chatbot")
    st.write("""
    This app allows you to ask questions and get answers based on Eaton Test Data.
    """)

    # Initialize session state if 'messages' doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    message_container = st.container(height=500, border=True)

    # Display existing messages in the session state
    for message in st.session_state["messages"]:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Input prompt from the user
    prompt = st.text_input("Enter your query:")

    if st.button("Submit"):
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    try:
                        response = query_rag(prompt)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")


"""
compressor = FlashrankRerank()
"""

if __name__ == "__main__":
    main()



