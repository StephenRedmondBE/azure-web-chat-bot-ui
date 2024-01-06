import os
import tiktoken
import numpy as np
import nest_asyncio
from typing import List
import requests


from langchain.schema import BaseRetriever, Document, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.globals import set_debug

from flask import Flask, render_template,jsonify,request
from flask_cors import CORS

from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

#set_debug(True)

customer_website_domain = "bearingpoint.com"
bing_subscription_key = ""

os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = ""

llm = AzureChatOpenAI(
    openai_api_version="2023-09-01-preview",
    azure_deployment="gpt-35-turbo",
    temperature=0
    )

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def BingSingleDomainSearch(query):
    #Ref doc: https://learn.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference#rankingresponse
    search_term = "site:" + customer_website_domain + " " + query
    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {
        "q": search_term,
        "responseFilter" : "Webpages",
        "count": 5,
        "setLang": "en-GB",
        "textDecorations": False,
        "textFormat": "HTML",
        "safeSearch": "Strict"
        }
    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)

    search_result_urls = []
    if "webPages" in search_results:
        for r in search_results["webPages"]["value"]:
            if not any(value in r["url"] for value in ("pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx")):
                search_result_urls.append(r["url"])
    
    #print(search_result_urls)

    return(search_result_urls)

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class get_web_docs(BaseRetriever, BaseModel):
    def get_relevant_documents(self, query: str) -> List[Document]:
            nest_asyncio.apply()
            loader = WebBaseLoader(BingSingleDomainSearch(query))
            loader.requests_per_second = 3
            docs = loader.aload()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 150,
                chunk_overlap  = 7,
                length_function = tiktoken_len
                )
            texts = text_splitter.split_documents(docs)

            for i in range(len(texts)):
                texts[i].page_content = " ".join(texts[i].page_content.split())
            
            embeddings = AzureOpenAIEmbeddings(
                openai_api_version="2023-09-01-preview",
                azure_deployment="embedding")
            
            doc_vectors = []
            for i in range(len(texts)):
                vector = embeddings.embed_documents([texts[i].page_content])
                doc_vectors.append(vector[0])
                
            query_embedding = embeddings.embed_query(query)

            doc_vector_array = np.array(doc_vectors)
            similarity = lambda x: cosine_similarity(x,query_embedding)
            print(doc_vector_array.shape)
            doc_vector_array = np.argpartition(similarity(doc_vector_array),-10)[-10:]
            similar_docs = [texts[i] for i in doc_vector_array]

            return similar_docs
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("get_web_docs does not support async")

retriever = get_web_docs()

rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    text=data.get('data')
    user_input = text
    try:
        ai_msg = rag_chain.invoke({"question": user_input, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=user_input), ai_msg])
        return jsonify({"response":True,"message":ai_msg.content})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})
    
if __name__ == '__main__':
    chat_history = []
    app.run()