import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio


# ============================================================
# PASSO 1: CARREGAR AS CONFIGURACOES
# ============================================================
#
# O arquivo .env guarda a chave da Groq para acessar o modelo.
load_dotenv()

api_key_groq = os.getenv("GROQ_API_KEY")

# ============================================================
# PASSO 2: DEFINIR O MODELO DE LINGUAGEM
# ============================================================
#
# Este modelo sera reutilizado pelos diferentes nos do grafo.
modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    api_key=api_key_groq,
)

# ============================================================
# PASSO 3: CRIAR PROMPTS ESPECIALIZADOS
# ============================================================
#
# Aqui criamos dois "especialistas":
# - um para praia
# - outro para montanha
#
# Cada prompt orienta o modelo a responder como um consultor tematico.
prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se com Sra Praia. Você é uma espcialista em viagens com destinos para praias brasileiras."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se com Sr. Montanha. Você é uma espcialista em viagens com destinos para montanhas e atividades radicais."),
        ("human", "{query}")
    ]
)


# Cada cadeia usa o mesmo modelo, mas com um prompt diferente.
cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

# ============================================================
# PASSO 4: DEFINIR A ESTRUTURA DA ROTA
# ============================================================
#
# TypedDict ajuda a documentar o formato esperado da saida.
# Neste caso, o roteador deve devolver apenas "praia" ou "montanha".
class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

# O prompt do roteador serve para classificar a pergunta.
# Em vez de responder ao usuario, ele decide para qual especialista
# a pergunta deve ser enviada.
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'."),
        ("human", "{query}")
    ]
)

# with_structured_output(Rota) pede ao modelo que responda obedecendo
# a estrutura definida anteriormente.
roteador = prompt_roteador | modelo.with_structured_output(Rota)

# ============================================================
# PASSO 5: DEFINIR O ESTADO DO GRAFO
# ============================================================
#
# O estado e o conjunto de dados que circula entre os nos do grafo.
# Aqui ele guarda:
# - a pergunta original
# - a decisao do roteador
# - a resposta final
class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

# ============================================================
# PASSO 6: CRIAR OS NOS DO GRAFO
# ============================================================
#
# Cada no e uma funcao que recebe o estado atual e devolve novos dados.
# No LangGraph, podemos pensar em cada no como uma etapa do fluxo.
async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["query"]}, config)}

async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}

# Esta funcao decide qual caminho o grafo deve seguir depois do roteador.
def escolher_no(estado: Estado)->Literal["praia", "montanha"]:
   return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

# ============================================================
# PASSO 7: MONTAR O GRAFO
# ============================================================
#
# O StateGraph permite conectar os nos e definir o fluxo de execucao.
# START e o ponto de entrada do grafo.
# END representa o fim do processamento.
grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

# Depois de definido, o grafo precisa ser compilado para virar uma aplicacao executavel.
app = grafo.compile()

# ============================================================
# PASSO 8: EXECUTAR O GRAFO
# ============================================================
#
# Neste exemplo enviamos uma pergunta sobre escalar montanhas.
# O roteador deve identificar o tema e encaminhar para o especialista correto.
async def main():
    resposta = await app.ainvoke(
        {"query": "Quero viajar para escalar montanhas no sul do Brasil."}
    )
    print(resposta["resposta"])

asyncio.run(main())
