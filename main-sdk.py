from openai import OpenAI
from dotenv import load_dotenv
import os

# ============================================================
# PASSO 1: CARREGAR AS CONFIGURACOES DO AMBIENTE
# ============================================================
#
# A chave da OpenAI fica no arquivo .env.
# Isso e mais seguro do que deixar a chave escrita diretamente no codigo.
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# PASSO 2: DEFINIR OS DADOS DO PEDIDO
# ============================================================
#
# Estas variaveis representam as informacoes que queremos enviar ao modelo.
numero_dias = 5
numero_criancas = 2
atividade = "música"

# Com essas variaveis, montamos o prompt final usando f-string.
prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma família de {numero_criancas} crianças, que gosta de {atividade}"

# ============================================================
# PASSO 3: CRIAR O CLIENTE DA OPENAI
# ============================================================
#
# O cliente e o objeto usado para fazer requisicoes para a API.
cliente = OpenAI(api_key=api_key)

# ============================================================
# PASSO 4: ENVIAR A CONVERSA PARA O MODELO
# ============================================================
#
# Assim como em outras APIs de chat, enviamos uma lista de mensagens.
# A mensagem "system" define o papel do assistente.
# A mensagem "user" contem o pedido em si.
resposta = cliente.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente de roteiro de viagens."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# A resposta vem estruturada em um objeto.
# Aqui pegamos apenas o texto retornado pelo modelo.
resposta_em_texto = resposta.choices[0].message.content

# Exibimos a resposta final.
print(resposta_em_texto)
