import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key_groq = os.getenv("GROQ_API_KEY")

modelo = ChatGroq(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key_groq,
)

prompt_consultor = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens."),
        ("human", "{query}")
    ]
)

assistente = prompt_consultor | modelo | StrOutputParser()

resposta = assistente.invoke({
    "query": "Quero férias em praias do Brasil?"
})

print(resposta)