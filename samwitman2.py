import streamlit as st
import os
import time
import pickle
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from embedding import get_embedding_function
from langchain_community.llms.ollama import Ollama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

# Constants
PERSIST_DIRECTORY = 'db'
DATA_PATH = 'Test_Files'
PROCESSED_FILES_PATH = 'processed_files.pkl'

# Initialize embedding function and model
embedding = get_embedding_function()
model = Ollama(model="phi3")

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'rb') as f:
            return pickle.load(f)
    return set()

def save_processed_files(processed_files):
    with open(PROCESSED_FILES_PATH, 'wb') as f:
        pickle.dump(processed_files, f)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def update_vector_db():
    processed_files = load_processed_files()
    current_files = set(f for f in os.listdir(DATA_PATH) if f.endswith('.pdf'))

    new_files = current_files - processed_files
    if new_files:
        st.write(f"Processing {len(new_files)} new files...")

        texts = load_documents()
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding,
                                         persist_directory=PERSIST_DIRECTORY)
        vectordb.persist()

        # Update processed files record
        processed_files.update(new_files)
        save_processed_files(processed_files)
    else:
        st.write("No new files to process.")

def init_retriever():
    update_vector_db()
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    Query_Prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are AI model assistant. Your task is to generate five different versions of the user question
        to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question
        your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide the alternative questions separated by newlines.
        Original question: {question}
        """,
    )
    retriever = MultiQueryRetriever.from_llm(
        vectordb.as_retriever(),
        model,
        prompt=Query_Prompt
    )
    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
# Set page configuration
st.set_page_config(
    page_title="Welcome to the 🤖 Eaton Busway chatbot",
    layout="wide",
)
# Streamlit UI
def main():
    st.title("🤖 Busway chatbot")
    st.write("""
    This app allows you to ask questions and get answers based on Eaton Test Data.
    """)

    # Initialize session state if 'messages' doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load or initialize the QA chain
    qa_chain = init_retriever()

    message_container = st.container()

    # Display existing messages in the session state
    for message in st.session_state["messages"]:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container:
            st.markdown(f"{avatar} {message['content']}")

    # Input prompt from the user
    prompt = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with message_container:
                st.markdown(f"😎 {prompt}")

            with message_container:
                with st.spinner(":green[processing...]"):
                    try:
                        start = time.time()
                        response = qa_chain(prompt)
                        end = time.time()
                        answer = response['result']
                        sources = "\n".join([source.metadata['source'] for source in response["source_documents"]])
                        response_message = f"**Answer:**\n{answer}\n\n**Sources:**\n{sources}\n\n*Query executed in {end - start:.2f} seconds.*"
                        st.session_state["messages"].append({"role": "assistant", "content": response_message})
                        st.markdown(f"🤖 {response_message}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

    # Upload section
    st.header("Upload New Documents")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

    if st.button("Add Documents"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(PERSIST_DIRECTORY, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
            st.success("Documents uploaded successfully!")

            # Reinitialize the vector database with new documents
            qa_chain = init_retriever()
        else:
            st.error("Please upload files before clicking 'Add Documents'.")

if __name__ == "__main__":
    main()


