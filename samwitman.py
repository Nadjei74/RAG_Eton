import time

from langchain_community.vectorstores import Chroma
#from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
#from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from embedding import get_embedding_function
from langchain_community.llms.ollama import Ollama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import flashrank_rerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate

#DATA_PATH = "Test_Files"
#document_loader = PyPDFDirectoryLoader(DATA_PATH)
#documents = document_loader.load()


#splitting the text into
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#texts = text_splitter.split_documents(documents)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = get_embedding_function()

"""vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persiste the db to disk
vectordb.persist()
vectordb = None"""

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

Query_Prompt = PromptTemplate(
    input_variables=["question"],
    template = """
    You are AI model assistant. Your task is to generate five different versions of the user question
    to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question
    your goal is to help the user overcome of some of the limitations of distance based similarity  distancde-based 
    similarity search. Provide the alternative questions seprated by newlines.
    Original question: {question}
    """,
)
model = Ollama(model="phi3")
"""compressor = flashrank_rerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor = compressor, base_retriever = results
        )"""
retriever = MultiQueryRetriever.from_llm(
    vectordb.as_retriever(),
    model,
    prompt=Query_Prompt
)

#RAG prompt
template = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

#retriever = vectordb.as_retriever()

docs = retriever.get_relevant_documents("What tests were run?")
print(docs)


# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm= model,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# full example
start = time.time()
query = "What tests were run on the Type Pow-R-Way III 4000A?:"
llm_response = qa_chain(query)
process_llm_response(llm_response)
end = time.time()
print(end - start)
