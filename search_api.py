import sys
import os

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from loguru import logger
from dotenv import load_dotenv

load_dotenv()
# logger
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))


def create_opensearch_vector_search_client(index_name, user_name, opensearch_password, embeddings_client, opensearch_endpoint, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embeddings_client,
        opensearch_url=opensearch_endpoint,
        http_auth=(user_name, opensearch_password),
        is_aoss=_is_aoss,
    )
    return docsearch


if __name__ == "__main__":

    index_name = "my-test-index"
    host = 'localhost'
    port = 9200
    user_name = "admin"
    opensearch_password = ""
    opensearch_endpoint = f"http://{host}:{port}"

    llm = ChatOpenAI(temperature = 0.0, model="gpt-4o-mini")

    embedding_client = OpenAIEmbeddings(model="text-embedding-ada-002")

    opensearch_vector_search_client = create_opensearch_vector_search_client(index_name, user_name, opensearch_password, embedding_client, opensearch_endpoint)

    prompt = ChatPromptTemplate.from_template("""If the context is not relevant, please answer the question by using your own knowledge about the topic. If you don't know the answer, just say that you don't know, don't try to make up an answer. don't include harmful content

    {context}

    Question: {input}
    Answer:""")
    
    docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=opensearch_vector_search_client.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5},
        ),
        combine_docs_chain = docs_chain
    )

    question = "What is risk culture and adoption?"
    logger.info(f"Invoking the chain with KNN similarity using OpenSearch")
    response = retrieval_chain.invoke({"input": question})
    
    logger.info("These are the similar documents from OpenSearch based on the provided query:")
    source_documents = response.get('context')
    for d in source_documents:
        print("")
        logger.info(f"Text: {d.page_content}")
        print("")
        logger.info(f"MetaData: {d.metadata}")

    print("")
    logger.info(f"The answer from llm is: {response.get('answer')}")
    