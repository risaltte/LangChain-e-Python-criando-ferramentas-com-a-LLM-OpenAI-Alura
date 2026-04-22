import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PASSO 1: CARREGAR AS CONFIGURACOES DO AMBIENTE
# ============================================================
#
# As chaves de API ficam no arquivo .env para nao serem escritas
# diretamente no codigo.
load_dotenv()

# Lemos a chave da Groq para autenticar as requisicoes.
api_key = os.getenv("GROQ_API_KEY")

# Se a chave nao existir, interrompemos a execucao com uma mensagem clara.
if not api_key:
    raise ValueError("GROQ_API_KEY nao encontrada. Confira se ela existe no arquivo .env.")

# ============================================================
# PASSO 2: DEFINIR AS INFORMACOES DO PEDIDO
# ============================================================
#
# Estas variaveis representam os dados que serao usados para montar o prompt.
# Em vez de escrever tudo em uma frase fixa, montamos a mensagem
# dinamicamente com f-strings.
numero_dias = 5
numero_criancas = 2
atividade = "música"

# O prompt e a instrucao enviada ao modelo.
# Aqui pedimos um roteiro de viagem com base nas variaveis acima.
prompt = (
    f"Crie um roteiro de viagem de {numero_dias} dias, "
    f"para uma família com {numero_criancas} crianças, "
    f"que gosta de {atividade}."
)

# ============================================================
# PASSO 3: CRIAR O CLIENTE DA GROQ
# ============================================================
#
# O cliente e o objeto responsavel por conversar com a API da Groq.
cliente = Groq(api_key=api_key)

# ============================================================
# PASSO 4: ENVIAR A SOLICITACAO AO MODELO
# ============================================================
#
# A API de chat recebe uma lista de mensagens.
# Isso imita a estrutura de uma conversa:
#
# system:
# define o comportamento geral do assistente
#
# user:
# traz a pergunta ou pedido do usuario
resposta = cliente.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente de roteiro de viagens.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
)

# A resposta da API vem como um objeto estruturado.
# Aqui extraimos apenas o texto principal da primeira escolha retornada.
resposta_em_texto = resposta.choices[0].message.content

# Mostra a resposta final no terminal.
print(resposta_em_texto)
