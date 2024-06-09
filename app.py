import json
import os
import sys
import boto3

# to create the embeddings by using titan embedding model
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# for data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter# Split the documents
from langchain_community.document_loaders import PyPDFDirectoryLoader# read all the PDF's

#vector embeddings and vector store
from langchain.vectorstores import FAISS

#LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")# it will access all the models
bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", 
                                       client=bedrock)# whcih will present in the bedrock

# data ingestion
def data_ingestion():
    """
    Reading the data then dividing the data into the smaller chunks.
    """
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split(documents)
    return docs

# vector embeddings and vector store.
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    # storing the vectors into the local
    vectorstore_faiss.save_local("faiss_index")


def get_claude_llm():
    # create the anthropic model
    llm = Bedrock(model_id = "ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens':512})   
    return llm

def get_llama2_llm():
    # create the llama2 model
    llm = Bedrock(model_id = "meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_length':512})   
    return llm



