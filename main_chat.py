import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Carrega as variáveis de ambiente do arquivo .env para obter as chaves de API
load_dotenv()

# Obtém a chave de API do modelo Groq a partir das variáveis de ambiente
api_key_groq = os.getenv("GROQ_API_KEY")

# Configura o modelo de linguagem Groq com a chave de API e outras opções
modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    api_key=api_key_groq,
)

# Configura o prompt para a sugestão de destinos de viagem, utilizando o formato de mensagens do ChatPromptTemplate
prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Vocé é um assistente de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios."),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

# Configura a cadeia de execução combinando o prompt, o modelo de linguagem e o parser de saída para obter respostas em formato de string
cadeia = prompt_sugestao | modelo | StrOutputParser()

# Configura a memória para armazenar o histórico de mensagens por sessão, utilizando um dicionário para associar cada sessão a um objeto de histórico de mensagens em memória
memoria = {}
sessao = "aula_langchain_alura"

# Função para obter o histórico de mensagens de uma sessão específica, criando um novo histórico se a sessão ainda não existir
def historico_por_sessao(sessao: str) -> InMemoryChatMessageHistory:
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

# Lista de perguntas para testar a cadeia de execução com memória, onde cada pergunta é processada e a resposta é obtida considerando o histórico de mensagens da sessão
lista_de_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

# 
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia, # A cadeia de execução que combina o prompt, o modelo e o parser
    get_session_history=historico_por_sessao, # Função para obter o histórico de mensagens da sessão
    input_messages_key="query", # Chave para a mensagem de entrada que contém a consulta do usuário
    history_messages_key="historico" # Chave para a mensagem de histórico que será preenchida com o histórico de mensagens da sessão
)

# Loop para processar cada pergunta na lista, invocando a cadeia de execução com memória e imprimindo a resposta obtida do modelo de linguagem
for uma_pergunta in lista_de_perguntas:
    resposta = cadeia_com_memoria.invoke(
       {
           "query": uma_pergunta
       },
       config={"session_id": sessao}
    )

    print("Usuário: ", uma_pergunta)
    print("IA: ", resposta, "\n")
