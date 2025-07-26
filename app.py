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

def get_conversational_chain(model_name,vectorstore = None,api_key = None):
    if model_name == "Google AI":
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',temperature = 0.3,google_api_key = api_key)
        prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
        chain = load_qa_chain(model,chain_type='stuff',prompt = prompt)
        return chain 
    

    def user_input(user_question,model_name,api_key,pdf_docs,conversation_history):
        if api_key is None or pdf_docs is None:
            st.warning("plese do provide apiKey and pdf_docs")
            return
        pdf_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(pdf_text,model)
        user_question_output = ""
        response_output = ""
        if model_name == "Google AI":
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)
            new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain("Google AI",vectorstore=new_db,api_key=api_key)
            response = chain({'input_documents':docs , 'question':user_question},return_only_outputs=True)
            user_question_output = user_question
            response_output = response['output_text']
            pdf_names = [pdf.names for pdf in pdf_docs] if pdf_docs else []
            conversation_history.append((user_question_output,response_output,model_name,datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"".join(pdf_names)))
            st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
            </div>
            
        """,
        unsafe_allow_html=True
    )
    