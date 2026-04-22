import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ============================================================
# PASSO 1: CARREGAR AS CONFIGURACOES DO AMBIENTE
# ============================================================
#
# O arquivo .env guarda informacoes sensiveis, como chaves de API.
# A funcao load_dotenv() le esse arquivo e disponibiliza essas variaveis
# para o programa sem precisarmos escrever segredos diretamente no codigo.
load_dotenv()

# Lemos a chave da Groq, que sera usada para acessar o modelo de linguagem.
api_key_groq = os.getenv("GROQ_API_KEY")

# ============================================================
# PASSO 2: DEFINIR O MODELO DE LINGUAGEM
# ============================================================
#
# Um LLM (Large Language Model) e o componente que entende a pergunta
# e gera a resposta em linguagem natural.
#
# Neste exemplo usamos a Groq como provedora e o modelo Llama 3.3 70B.
# A temperatura em 0.5 deixa as respostas equilibradas:
# nem tao rigidas, nem tao criativas.
modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    api_key=api_key_groq,
)

# ============================================================
# PASSO 3: MONTAR O PROMPT
# ============================================================
#
# Prompt e a instrucao enviada ao modelo.
# Aqui usamos um ChatPromptTemplate, que organiza a conversa em mensagens:
#
# system:
# define o papel do assistente
#
# placeholder:
# reserva um espaco para o historico da conversa
#
# human:
# recebe a nova pergunta do usuario
#
# Essa estrutura e muito util em chatbots com memoria, porque o modelo
# pode responder considerando o que ja foi dito antes.
prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Vocé é um assistente de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios."),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

# ============================================================
# PASSO 4: CRIAR A CADEIA BASICA
# ============================================================
#
# No LangChain, podemos montar pipelines ligando varias etapas com o operador |.
# Aqui a cadeia faz tres coisas:
# 1. monta a mensagem final com base no prompt
# 2. envia essa mensagem para o modelo
# 3. transforma a resposta em texto simples
cadeia = prompt_sugestao | modelo | StrOutputParser()

# ============================================================
# PASSO 5: DEFINIR UMA MEMORIA DE CONVERSA
# ============================================================
#
# Um chatbot com memoria precisa guardar o historico das mensagens.
# Aqui usamos um dicionario simples em memoria:
# - a chave identifica a sessao
# - o valor guarda as mensagens trocadas naquela sessao
#
# Isso significa que conversas diferentes podem ter historicos diferentes.
memoria = {}
sessao = "aula_langchain_alura"

# Esta funcao recebe o nome de uma sessao e devolve o historico correspondente.
# Se a sessao ainda nao existir, criamos um novo historico vazio.
def historico_por_sessao(sessao: str) -> InMemoryChatMessageHistory:
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

# ============================================================
# PASSO 6: PREPARAR PERGUNTAS DE TESTE
# ============================================================
#
# Estas perguntas mostram como a memoria funciona na pratica.
# A segunda pergunta depende da primeira:
# "Qual a melhor epoca do ano para ir?"
#
# Sem memoria, o modelo talvez nao saiba a que lugar essa pergunta se refere.
# Com memoria, ele consegue usar o contexto da conversa anterior.
lista_de_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

# ============================================================
# PASSO 7: ADICIONAR MEMORIA A CADEIA
# ============================================================
#
# RunnableWithMessageHistory e um componente do LangChain que "envolve"
# a cadeia basica e injeta automaticamente o historico da conversa.
#
# input_messages_key:
# indica onde esta a nova pergunta do usuario
#
# history_messages_key:
# indica o nome do campo do prompt onde o historico sera colocado
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

# ============================================================
# PASSO 8: EXECUTAR O CHAT COM MEMORIA
# ============================================================
#
# Aqui percorremos a lista de perguntas e enviamos cada uma para a cadeia.
# O config com session_id informa qual historico deve ser usado.
#
# Como usamos sempre a mesma sessao, a conversa vai acumulando contexto.
for uma_pergunta in lista_de_perguntas:
    resposta = cadeia_com_memoria.invoke(
       {
           "query": uma_pergunta
       },
       config={"session_id": sessao}
    )

    print("Usuário: ", uma_pergunta)
    print("IA: ", resposta, "\n")
