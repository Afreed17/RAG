#import packages
import streamlit as st
from PyPDF2 import PdfReader
import pandas
import base64
import os 

#imports for langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from datetime import datetime



def get_pdf_text(pdfDoc):
    text = ""
    for pdf in pdfDoc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text,model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000,chunk_overlap = 1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(model_name,text_chunks,api_key = None):
    if model_name == "Google AI":
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks,embedding=embedding)
    vector_store.save_local("faiss_index")
    return vector_store