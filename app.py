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






