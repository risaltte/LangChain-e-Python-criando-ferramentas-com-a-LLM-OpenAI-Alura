from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# ============================================================
# PASSO 1: CARREGAR AS CONFIGURACOES DO AMBIENTE
# ============================================================
#
# Antes de usar qualquer modelo de IA hospedado na nuvem, precisamos
# informar nossas chaves de acesso (API keys).
#
# Em vez de escrever essas chaves diretamente no codigo, o mais seguro
# e organizado e guarda-las em um arquivo chamado .env.
#
# A funcao load_dotenv() le esse arquivo e torna essas variaveis
# disponiveis dentro do programa.
load_dotenv()

# Esta chave sera usada para acessar a Groq, provedora do modelo de linguagem
# que vai gerar a resposta final para o usuario.
api_key_groq = os.getenv("GROQ_API_KEY")

# Esta chave sera usada na Gemini para criar embeddings.
# Embeddings sao representacoes numericas do texto.
# Pense neles como uma forma de transformar frases em vetores,
# para que possamos comparar significados, e nao apenas palavras iguais.
api_key_gemini = os.getenv("GEMINI_API_KEY")

# Se a chave da Groq nao existir, interrompemos o programa com uma mensagem clara.
# Isso evita erros confusos mais adiante.
if not api_key_groq:
    raise ValueError("GROQ_API_KEY nao encontrada. Confira se ela existe no arquivo .env.")

# Fazemos a mesma validacao para a chave da Gemini.
# Sem embeddings, o sistema nao consegue localizar os trechos mais relevantes.
if not api_key_gemini:
    raise ValueError("GEMINI_API_KEY nao encontrada. Confira se ela existe no arquivo .env.")

# ============================================================
# PASSO 2: DEFINIR O MODELO DE LINGUAGEM QUE VAI RESPONDER
# ============================================================
#
# Um LLM (Large Language Model) e o modelo responsavel por entender
# a pergunta e gerar uma resposta em linguagem natural.
#
# Aqui usamos a Groq como provedora e o modelo Llama 3.3 70B como motor
# principal da resposta.
#
# O parametro temperature controla o quanto a resposta sera criativa.
# - temperature alta: respostas mais variadas
# - temperature baixa: respostas mais consistentes
#
# Como estamos trabalhando com consulta a documentos, preferimos
# temperature=0 para deixar a resposta mais objetiva e previsivel.
modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key_groq,
)

# ============================================================
# PASSO 3: DEFINIR O MODELO DE EMBEDDINGS
# ============================================================
#
# O modelo de embeddings nao gera respostas para o usuario.
# O trabalho dele e outro: transformar texto em vetores numericos.
#
# Por que isso e importante?
# Porque computadores comparam numeros melhor do que comparam "significados".
# Quando transformamos textos em vetores, conseguimos medir quais textos
# sao semanticamente parecidos.
#
# Exemplo:
# "Meu item foi roubado"
# "Tive um objeto furtado"
#
# Mesmo sem usar exatamente as mesmas palavras, os embeddings podem indicar
# que os significados sao proximos.
#
# Neste projeto, usamos a Gemini para gerar embeddings remotamente.
# Isso evita baixar modelos locais do Hugging Face e simplifica a execucao.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key_gemini,
)

# ============================================================
# PASSO 4: CARREGAR O DOCUMENTO QUE SERA CONSULTADO
# ============================================================
#
# Aqui carregamos o arquivo de texto que servira como base de conhecimento.
#
# Em um sistema RAG, esse documento funciona como a fonte de informacao
# que o modelo deve consultar antes de responder.
#
# Isso e importante porque um LLM sozinho pode "inventar" respostas.
# Com RAG, tentamos reduzir isso, obrigando o modelo a responder com base
# em trechos reais do documento.
documento = TextLoader(
    "documentos/GTB_gold_Nov23.txt"
).load()

# ============================================================
# PASSO 5: DIVIDIR O DOCUMENTO EM PEDACOS MENORES
# ============================================================
#
# Um documento grande nao costuma ser usado de uma vez so.
# Em vez disso, ele e quebrado em pequenos trechos chamados chunks.
#
# Isso melhora bastante o RAG porque:
# - a busca fica mais precisa
# - o contexto enviado ao modelo fica menor e mais relevante
# - evitamos mandar texto demais sem necessidade
#
# chunk_size:
# define o tamanho aproximado de cada pedaco
#
# chunk_overlap:
# cria uma pequena sobreposicao entre os pedacos
#
# Essa sobreposicao ajuda quando uma ideia comeca no final de um trecho
# e termina no inicio do proximo.
pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documento)

# ============================================================
# PASSO 6: CRIAR A BASE DE BUSCA VETORIAL
# ============================================================
#
# Agora pegamos os pedacos do documento e transformamos cada um em embedding.
# Esses embeddings sao armazenados em uma estrutura de busca vetorial.
#
# Aqui usamos o FAISS, que e uma biblioteca muito comum para busca por similaridade.
#
# O papel do FAISS e responder perguntas como:
# "Quais trechos do documento mais se parecem com esta pergunta?"
#
# Quando chamamos as_retriever(), transformamos essa base vetorial em um retriever.
#
# Retriever:
# e o componente que recebe a pergunta do usuario e devolve os trechos
# mais relevantes do documento.
#
# k=2 significa:
# "traga os 2 trechos mais proximos semanticamente da pergunta".
dados_recuperados = FAISS.from_documents(
    pedacos, 
    embeddings
).as_retriever(search_kwargs={"k": 2})

# ============================================================
# PASSO 7: MONTAR O PROMPT
# ============================================================
#
# Prompt e a instrucao que enviamos para o modelo.
# Ele define o comportamento esperado e quais informacoes o modelo recebera.
#
# Aqui usamos dois tipos de mensagem:
#
# system:
# define a regra geral do assistente
#
# human:
# envia a pergunta do usuario junto com o contexto recuperado
#
# Esta parte representa o coracao do RAG:
# 1. buscamos trechos relevantes no documento
# 2. passamos esses trechos como contexto
# 3. o modelo responde com base nesse material
prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido."),
        ("human", "{query}\n\nContexto:\n{contexto}\n\nResposta:")
    ]
)

# ============================================================
# PASSO 8: CRIAR A CADEIA DE EXECUCAO NO LANGCHAIN
# ============================================================
#
# O LangChain permite montar pipelines de forma bem legivel.
# Esta linha pode ser lida assim:
#
# prompt_consulta_seguro
# -> prepara a mensagem no formato correto
#
# modelo
# -> envia o prompt para o LLM da Groq
#
# StrOutputParser()
# -> converte a resposta final para texto simples
#
# O operador | funciona como um encadeamento de etapas.
cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

# ============================================================
# PASSO 9: CRIAR A FUNCAO QUE EXECUTA O RAG
# ============================================================
#
# Esta funcao recebe a pergunta do usuario e executa o fluxo completo:
# 1. busca os trechos mais relevantes
# 2. monta o contexto
# 3. envia a pergunta + contexto para o modelo
# 4. devolve a resposta final
def responder(pergunta: str):
    # Busca os trechos do documento que fazem mais sentido para a pergunta.
    trechos = dados_recuperados.invoke(pergunta)

    # Junta todos os trechos encontrados em um unico texto.
    # Esse sera o contexto fornecido ao modelo.
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)

    # Executa a cadeia completa:
    # - preenche o prompt
    # - chama o modelo
    # - recebe a resposta em texto
    return cadeia.invoke({
        "query": pergunta, 
        "contexto": contexto
    })


# ============================================================
# PASSO 10: TESTAR O SISTEMA
# ============================================================
#
# Aqui fazemos uma pergunta de exemplo para validar o fluxo.
# Se tudo estiver configurado corretamente, o programa vai:
# - ler o documento
# - buscar os trechos mais relevantes
# - enviar esses trechos ao modelo
# - imprimir a resposta final
print(responder("Como devo precedorer caso tenha um ítem roubado."))
