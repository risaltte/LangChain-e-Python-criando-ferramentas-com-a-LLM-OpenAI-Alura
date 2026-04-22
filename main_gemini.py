import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PASSO 1: CARREGAR AS CONFIGURACOES DO AMBIENTE
# ============================================================
#
# A chave da Gemini fica no arquivo .env.
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Se a chave nao existir, interrompemos a execucao com uma mensagem clara.
if not api_key:
    raise ValueError("GEMINI_API_KEY nao encontrada. Confira se ela existe no arquivo .env.")

# ============================================================
# PASSO 2: DEFINIR AS INFORMACOES DO PEDIDO
# ============================================================
#
# Estas variaveis alimentam o prompt de forma dinamica.
numero_dias = 5
numero_criancas = 2
atividade = "música"

# O prompt e a instrucao principal enviada ao modelo.
prompt = (
    f"Crie um roteiro de viagem de {numero_dias} dias, "
    f"para uma família com {numero_criancas} crianças, "
    f"que gosta de {atividade}."
)

# ============================================================
# PASSO 3: CRIAR O CLIENTE DA GEMINI
# ============================================================
#
# O cliente e o objeto responsavel por acessar a API da Gemini.
cliente = genai.Client(api_key=api_key)

# ============================================================
# PASSO 4: CHAMAR O MODELO
# ============================================================
#
# generate_content envia o prompt para o modelo.
#
# system_instruction:
# define o comportamento geral do assistente
#
# temperature:
# controla o nivel de variacao na resposta
resposta = cliente.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="Você é um assistente de roteiro de viagens.",
        temperature=0.7,
    ),
)

# A resposta vem estruturada e aqui pegamos apenas o texto final.
resposta_em_texto = resposta.text

# Exibe a resposta no terminal.
print(resposta_em_texto)
