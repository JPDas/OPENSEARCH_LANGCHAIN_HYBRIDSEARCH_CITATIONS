
import sys
import os
import pdfplumber

from loguru import logger
from opensearchpy import OpenSearch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))

def get_opensearch_cluster_client(index_name, host, port, user_id, password):
    opensearch_endpoint = host
    opensearch_client = OpenSearch(
        hosts=[{
            'host': opensearch_endpoint,
            'port': port
            }],
        http_auth=(user_id, password),
        index_name = index_name,
        )
    return opensearch_client
def check_opensearch_index(opensearch_client, index_name):
    return opensearch_client.indices.exists(index=index_name)

def create_index(opensearch_client, index_name):
    settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil"
                }
            }
        }
    response = opensearch_client.indices.create(index=index_name, body=settings)
    return bool(response['acknowledged'])

def create_index_mapping(opensearch_client, index_name):
    response = opensearch_client.indices.put_mapping(
        index=index_name,
        body={
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss"
                    }
                },
                "text": {
                    "type": "text",
                    "store": True
                },
                "metadata": {
                    "properties": {
                        "version" : {
                            "type" : "text",
                            "store": True
                        },
                        "document_name": {
                            "type" : "text",
                            "store": True
                        },
                        "page_number": {
                            "type" : "integer",
                            "store": True
                        },
                        "chunk_number": {
                            "type" : "integer",
                            "store": True
                        }
                    }                   
                }
            }
        }
    )
    return bool(response['acknowledged'])

def delete_opensearch_index(opensearch_client, index_name):
    logger.info(f"Trying to delete index {index_name}")
    try:
        response = opensearch_client.indices.delete(index=index_name)
        logger.info(f"Index {index_name} deleted")
        return response['acknowledged']
    except Exception as e:
        logger.info(f"Index {index_name} not found, nothing to delete")
        return True

def insert_doc(opensearch_client, index_name, docs, meta_datas, embeddings):

    chunk_count, success_count, failure_count = 0, 0, 0
    for doc, meta, vector in zip(docs, meta_datas, embeddings):

        
        # Add a document to the index.

        document = {
            'vector_field': vector,
            'metadata': meta,
            'text': doc
        }
        id = f'chunk_{chunk_count}'

        response = opensearch_client.index(
            index = index_name,
            body = document,
            id = id,
            refresh = True
        )
        
        chunk_count += 1
        
        logger.info(response)

        if response['result'] == 'created':
            success_count +=1
        else:
            failure_count +=1
    return success_count, failure_count

def get_chunks_with_meta(embed, file_path, meta):
    with pdfplumber.open(file_path) as pdf:
        embeddings, metadatas, docs = [], [], []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size=512,
                chunk_overlap=20,
                separators=["\n\n", "\n", " ", ".", "?", "!", ",", ""]
                )
            
            chunks = text_splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                meta_data = {
                    "page_number": page_num,
                    "chunk_number": i+1,
                    **meta
                    }

                embedding_response = embed.embed_documents([chunk])

                # logger.info(embedding_response)

                
                embeddings.append(embedding_response[0])

                metadatas.append(meta_data)
                docs.append(chunk)
            
    

        return embeddings, metadatas, docs

    

if __name__ == "__main__":
    
    index_name = "my-test-index"
    host = 'localhost'
    port = 9200
    user_name = "admin"
    password = ""
    opensearch_client =  get_opensearch_cluster_client(index_name, host, port, user_name, password)
    pdf_path = r"Dataset\\enterprise-risk-management-policy.pdf"
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # Index operation

    # delete_opensearch_index(opensearch_client, index_name)
    exists = check_opensearch_index(opensearch_client, index_name)

    if not exists:
        logger.info("Creating OpenSearch index")
        success = create_index(opensearch_client, index_name)
        if success:
            logger.info("Creating OpenSearch index mapping")
            success = create_index_mapping(opensearch_client, index_name)
            logger.info("OpenSearch Index mapping created")
        
    #Ingest documents
    meta = {
        "version" : "1.1",
        "document_name": "test_file"
    }
    embeddings, metadatas, docs = get_chunks_with_meta(embeddings, pdf_path, meta)

    success_count, failure_count = insert_doc(opensearch_client, index_name, docs, metadatas, embeddings)

    logger.info(f"Inserted documents success count {success_count} and failure count {failure_count}")