from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# ============================================================
# PASSO 1: CARREGAR AS CONFIGURACOES DO AMBIENTE
# ============================================================
#
# Aqui lemos as variaveis do arquivo .env, onde esta a chave da OpenAI.
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# PASSO 2: DEFINIR O MODELO DE LINGUAGEM
# ============================================================
#
# Este e o modelo que vai gerar a resposta final ao usuario.
# Como estamos em um cenario de consulta a documento, deixamos
# temperature=0 para reduzir variacoes desnecessarias.
modelo = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0, 
    api_key=api_key
)

# ============================================================
# PASSO 3: DEFINIR O MODELO DE EMBEDDINGS
# ============================================================
#
# Embeddings transformam texto em vetores numericos.
# Esses vetores permitem comparar significados e fazer busca semantica.
embeddings = OpenAIEmbeddings()

# ============================================================
# PASSO 4: CARREGAR O DOCUMENTO
# ============================================================
#
# O arquivo de texto funciona como base de conhecimento para o RAG.
documento = TextLoader(
    "documentos/GTB_gold_Nov23.txt"
).load()

# ============================================================
# PASSO 5: DIVIDIR O DOCUMENTO EM CHUNKS
# ============================================================
#
# Em vez de trabalhar com o documento inteiro de uma vez, dividimos
# o conteudo em partes menores.
#
# Isso melhora a busca e permite enviar somente o contexto relevante ao modelo.
pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documento)

# ============================================================
# PASSO 6: CRIAR A BUSCA VETORIAL COM FAISS
# ============================================================
#
# FAISS armazena os embeddings dos pedaços do documento e ajuda a encontrar
# quais trechos sao mais parecidos com a pergunta do usuario.
#
# Ao transformar o indice em retriever, ganhamos um componente pronto
# para recuperar os trechos mais relevantes.
dados_recuperados = FAISS.from_documents(
    pedacos, 
    embeddings
).as_retriever(search_kwargs={"k": 2})

# ============================================================
# PASSO 7: MONTAR O PROMPT
# ============================================================
#
# Este prompt orienta o modelo a responder apenas com base no contexto
# recuperado do documento.
prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido."),
        ("human", "{query}\n\nContexto:\n{contexto}\n\nResposta:")
    ]
)

# ============================================================
# PASSO 8: MONTAR A CADEIA
# ============================================================
#
# A cadeia organiza o prompt, chama o modelo e converte a saida para texto.
cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

# ============================================================
# PASSO 9: EXECUTAR O FLUXO RAG
# ============================================================
#
# Esta funcao:
# 1. busca trechos relevantes no documento
# 2. junta esses trechos em um contexto
# 3. envia pergunta + contexto ao modelo
def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)

    return cadeia.invoke({
        "query": pergunta, 
        "contexto": contexto
    })


# ============================================================
# PASSO 10: TESTAR O SISTEMA
# ============================================================
print(responder("Como devo precedorer caso tenha um ítem roubado."))
