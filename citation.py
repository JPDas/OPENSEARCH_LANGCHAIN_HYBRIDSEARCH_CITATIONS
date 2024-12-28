import sys
import os
import hashlib
import re


from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from typing import List
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


def generate_anchor_name(text):
    """Generates a unique anchor name from text using SHA-256."""
    hashed_text = hashlib.sha256(text.encode()).hexdigest()
    return f"quote-{hashed_text[:8]}"  # Use first 8 characters for brevity

def generate_citation(source_id, quote, base_url):
    """Generates a formatted citation with an anchor link."""
    anchor_name = generate_anchor_name(quote)
    link = f"{base_url}#{anchor_name}"
    return f"<citation><source_id>[{source_id}]</source_id><link>({link})</link></citation>"


# def update_citation_with_streaming(llm_stream, link_base):
#     """Processes an LLM streaming response and adds citations."""
#     full_response = ""
#     citations = []
#     in_citation_block = False  # Track if we are inside a citation block

#     for chunk in llm_stream:
#         text_chunk = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
#         full_response += text_chunk

#         # Regular expression to find potential quotes (improve as needed)
#         quote_matches = re.findall(r'"([^"]*)"', full_response) #finds the string between two double quotes

#         for quote in quote_matches:
#             # Check if this quote has already been cited
#             if not any(quote in cit["quote"] for cit in citations):

#                 # Simulate retrieving source ID (replace with your actual retrieval)
#                 source_id = f"Source-{len(citations) + 1}"  # Replace with actual source ID retrieval

#                 citation = {
#                     "source_id": source_id,
#                     "quote": quote,
#                     "formatted_citation": generate_citation(source_id, quote, link_base),
#                 }
#                 citations.append(citation)

#                 # Remove the quote from the response so it's not repeated.
#                 full_response = full_response.replace(f'"{quote}"', "")

#         # Check if we are at the end of the LLM stream
#         if chunk.get("choices", [{}])[0].get("finish_reason"):
#             cited_answer = "<cited_answer>\n"
#             cited_answer += f"    <answer>{full_response.strip()}</answer>\n"
#             cited_answer += "    <citations>\n"
#             for cit in citations:
#                 cited_answer += f"        {cit['formatted_citation']}\n"
#             cited_answer += "    </citations>\n"
#             cited_answer += "</cited_answer>"
#             yield cited_answer
#             return

#         yield text_chunk  # Yield the current chunk of the response


def update_citation_without_streaming(response, link_base):
    """Generates an LLM response with citations."""
       
    llm_response = response.get('answer')

    logger.info(f"llm response: {llm_response}")
    answer_match = re.search(r"<cited_answer>(.*?)</cited_answer>", llm_response, re.DOTALL)
    cited_answer_match = re.search(r"<citations>(.*?)</citations>", llm_response, re.DOTALL)

    cited_answer_content = cited_answer_match.group(0)

    logger.info(f"Cited answer : {cited_answer_content}")

    new_citations_string = "<citations>"

    citation_matches = re.findall(r"<citation>(.*?)</citation>", cited_answer_content, re.DOTALL)

    logger.info(f"citation_matches : {citation_matches}")
    for citation_content in citation_matches:
        source_id_match = re.search(r"<source_id>(.*?)</source_id>", citation_content)
        source_id = source_id_match.group(1) if source_id_match else None

        quote_match = re.search(r"<quote>(.*?)</quote>", citation_content)
        quote = quote_match.group(1) if quote_match else None

        new_citations_string += generate_citation(source_id, quote, link_base)

    new_citations_string += "</citations>"

    final_output = llm_response.replace(cited_answer_content,new_citations_string) #add answer to the final output.


    return final_output

if __name__ == "__main__":

    index_name = "my-test-index"
    host = 'localhost'
    port = 9200
    user_name = "admin"
    opensearch_password = ""
    opensearch_endpoint = f"http://{host}:{port}"

    link_base = "https://example.com/article.html"

    llm = ChatOpenAI(temperature = 0.0, model="gpt-4o-mini")
        
    embedding_client = OpenAIEmbeddings(model="text-embedding-ada-002")

    opensearch_vector_search_client = create_opensearch_vector_search_client(index_name, user_name, opensearch_password, embedding_client, opensearch_endpoint)

    prompt = ChatPromptTemplate.from_template("""
        If the context is not relevant, please answer the question by using your own knowledge about the topic. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Don't include harmful content.

        Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
        justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
        that justify the answer. Use the following format for your final output:

        <cited_answer>
            <answer></answer>
            <citations>
                <h4>Sources Cited:</h4>
                <citation><source_id>[source_id]</source_id><quote><u>quote</u></quote></citation>
                <citation><source_id>[source_id]</source_id><quote><u>quote</u></quote></citation>
                ...
            </citations>
        </cited_answer>

        {context}
        Question: {input}
        Answer:""")
        
    docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=opensearch_vector_search_client.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5},
        ),
        combine_docs_chain = docs_chain,

    )

    question = "What is risk culture and adoption?"
    logger.info(f"Invoking the chain with KNN similarity using OpenSearch")
    response = retrieval_chain.invoke({"input": question})

    answer = response.get('answer')
    logger.info(f"The answer from llm is: {answer}")
    
    # response = update_citation_without_streaming(response, link_base)

    # logger.info(f"Final response: {response}")
