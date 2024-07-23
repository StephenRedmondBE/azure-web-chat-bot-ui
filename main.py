import os
import tiktoken
import numpy as np
import nest_asyncio
from typing import List
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)


#import openai as OpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
'''
llm = AzureChatOpenAI(
    openai_api_version="2023-09-01-preview",
    azure_deployment="gpt-35-turbo",
    temperature=0
    )
'''

from langchain.schema import BaseRetriever, Document, HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.globals import set_debug

from flask import Flask, render_template,jsonify,request
from flask_cors import CORS

#from langchain_community.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough,RunnableLambda


bing_subscription_key = os.environ["BING_API_KEY"]

#set_debug(True)

# Constrain the chat to search only one site
customer_website_domain = "gov.ie"

# Constrain the agent via System Prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use four sentences maximum and keep the answer concise. \
If the original question is in the Irish language, please respond in the Irish language.\

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
    logging.info("contextualized_question")

    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]
    
def format_docs(docs):
    logging.info("format_docs")
    return "\n\n".join(doc.page_content for doc in docs)

def BingSingleDomainSearch(query):
    logging.info("BingSingleDomainSearch")
    logging.info("query: " + str(query))
    print('\n## Sending query to Bing.')

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
        print('\n## Retrieved pages from web search:')
        for r in search_results["webPages"]["value"]:
            if not any(value in r["url"] for value in ("pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx")):
                search_result_urls.append(r["url"])
                print(r['url'])
    
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
    logging.info("get_web_docs")
    print("Performing Search");
    def get_relevant_documents(self, query: str) -> List[Document]:
            nest_asyncio.apply()
            loader = WebBaseLoader(BingSingleDomainSearch(query), continue_on_failure=True)
            loader.requests_per_second = 3
            print("\n## Retrieve text from the pages")
            docs = loader.aload()
            
            my_chunk_size = 200
            my_overlap_size = 20
            print("\n## Splitting retrieved pages into chunks of " + str(my_chunk_size) + " characters with overlap of " + str(my_overlap_size) + " characters:")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = my_chunk_size,
                chunk_overlap  = my_overlap_size,
                length_function = tiktoken_len
                )
            texts = text_splitter.split_documents(docs)

            for i in range(len(texts)):
                texts[i].page_content = " ".join(texts[i].page_content.split())
            #text-embedding-3-small
            #text-embedding-ada-002
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
#            embeddings = AzureOpenAIEmbeddings(
#                openai_api_version="2023-09-01-preview",
#                azure_deployment="embedding")
         
            doc_vectors = []
            
            for i in range(len(texts)):
                print('\n' + str(i) + ". Retrieve embeddings for: " + str([texts[i].metadata['source']]))
                print([texts[i].page_content])
                vector = embeddings.embed_documents([texts[i].page_content])
                doc_vectors.append(vector[0])
                
            print("\n## Now retrieve the embeddings for the User Query.")
            print('[' + query + ']')
            query_embedding = embeddings.embed_query(query)
            #query_embedding = embeddings.embed_documents([query])

            print("\n## Transform the returned data vectors into an array.")
            doc_vector_array = np.array(doc_vectors)
            print("\n## Calculate the similarity between the document vectors and the query.")
            similarity = lambda x: cosine_similarity(x,query_embedding)
            
            #print("Shape of the document vector array: " + str(doc_vector_array.shape))

            print("\n\n## Get the top 10 closest embeddings:")
            doc_vector_array = np.argpartition(similarity(doc_vector_array),-10)[-10:]
            similar_docs = [texts[i] for i in doc_vector_array]
            for i in doc_vector_array:
                print(str(i) + ':\nSource: ' + texts[i].metadata['source'] + '\n' + texts[i].page_content + '\n')

            #logging.info("Context:\n\n" + format_docs(similar_docs) + "\n\n")

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
        logging.info("Retrieve AI Message based on user input")
        logging.info("user_input: " + str(user_input))
        print("\n# Run the process.")
        ai_msg = rag_chain.invoke({"question": user_input, "chat_history": chat_history}).replace("System: ", "AI: ", 1)

        print("\n## Returned message from AI:\n" + str(ai_msg) + '\n')

        chat_history.extend([HumanMessage(content=user_input), AIMessage(content=ai_msg)])

        return jsonify({"response":True,"message":ai_msg})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'

        logging.error(error_message)

        return jsonify({"message":error_message,"response":False})
    
if __name__ == '__main__':
    chat_history = []
    app.run()
    #app.run(host="0.0.0.0", port=8777)