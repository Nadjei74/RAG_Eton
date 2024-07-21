#query.py
import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from embedding import get_embedding_function

from langchain.chains import  RetrievalQA
import time
import asyncio

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
For the answer you give, make sure to add the source which you can 
get from the metadata of the context, where the first part is the source file's name,
and the second part is the page number.
If you don't know the answer, just say you don't know.  

"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    """res = db.as_retriever(search_kwargs = {"k":5})
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
            base_compressor = compressor, base_retriever = res
        )
    #compressed_docs = compression_retriever.invoke(query_text)"""
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    """


    """
    # llm to use
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # Measure model invocation time
    model = Ollama(model="phi3")

    start_time = time.time()
    """qa_chain = RetrievalQA.from_chain_type(

        llm=model,
        retriever = compression_retriever,
        #return_source_documents=True
    )
    response_text = qa_chain.invoke(query_text)  # Assuming async_invoke is supported"""

    end_time = time.time()
    print(f"Model invocation time: {end_time - start_time} seconds")

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    #sources = [doc.metadata.get("id", None) for doc, _score in compressed_docs]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return formatted_response


"""def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    #query_text = "Who was the manufacurer's representative? "
    #query_rag(query_text)


if __name__ == "__main__":
    main()
"""