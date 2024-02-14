from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import PineconeHybridSearchRetriever
import pinecone
from pinecone_text.sparse import BM25Encoder
import chainlit as cl
import os

if not load_dotenv(override=True):
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    pass

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Loading the retriever
def setup_retriever():
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))
    bm25_encoder = BM25Encoder().default()
    embeddings = HuggingFaceEmbeddings(model_name=os.environ.get("EMBEDDINGS_MODEL_NAME"))
    index = pinecone.Index(os.environ.get("PINECONE_INDEX"))
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, top_k=4)        
    return retriever

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=setup_retriever(),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
        )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0.1)
    
    return llm

#QA Model Function
def qa_bot():
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

