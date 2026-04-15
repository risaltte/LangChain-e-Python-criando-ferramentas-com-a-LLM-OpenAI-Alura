import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY nao encontrada. Confira se ela existe no arquivo .env.")

numero_dias = 5
numero_criancas = 2
atividade = "música"

prompt = (
    f"Crie um roteiro de viagem de {numero_dias} dias, "
    f"para uma família com {numero_criancas} crianças, "
    f"que gosta de {atividade}."
)

cliente = Groq(api_key=api_key)

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

resposta_em_texto = resposta.choices[0].message.content
print(resposta_em_texto)
