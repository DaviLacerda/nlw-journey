import json
import os
import bs4 as bs
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

query = """
Vou viajar para Londres em agosto de 2024.
Quero que faça para um roteiro de viagem para mim com eventos que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Londres.
"""

url = "https://www.dicasdeviagem.com/inglaterra/"


def researchAgent(query, llm):
    prompt = hub.pull("hwchase17/react")
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    # tools = load_tools(["wikipedia"], llm=llm)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt)
    return agent_executor.invoke({"input": query})["output"]


def load_data(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs.SoupStrainer(
                class_=(
                    "postcontentwrap",
                    "pagetitleloading background-imaged loading-dark",
                )
            )
        ),
    )

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma().from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def get_relevant_docs(query):
    retriever = load_data(url)
    return retriever.invoke(query)


def supervisorAgent(query, llm, web_context, relevant_documents):
    prompt_template = """
    Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado.
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    Contexto: {web_context}
    Documentos Relevantes: {relevant_documents}
    Usuário: {query}
    Assistente:
    """

    prompt = PromptTemplate(
        input_variables=["web_context", "relevant_documents", "query"],
        template=prompt_template,
    )

    sequence = RunnableSequence(prompt | llm)

    response = sequence.invoke(
        {
            "web_context": web_context,
            "relevant_documents": relevant_documents,
            "query": query,
        }
    )

    return response


def get_response(query, llm):
    web_context = researchAgent(query, llm)
    relevant_docs = get_relevant_docs(query)
    response = supervisorAgent(query, llm, web_context, relevant_docs)

    return response


def lambda_handler(event, context):
    body = json.loads(event.get('body', {}))
    query = body.get('question', 'Parametro question não fornecido')
    response = get_response(query, llm).content
    return {
        "statusCode": 200,
        "headers": {
        "Content-Type": "application/json"
        },
        "body": json.dumps({
        "message": "Tarefa concluída com sucesso",
        "details": response
        }), 
    }
