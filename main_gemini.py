import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY nao encontrada. Confira se ela existe no arquivo .env.")

numero_dias = 5
numero_criancas = 2
atividade = "música"

prompt = (
    f"Crie um roteiro de viagem de {numero_dias} dias, "
    f"para uma família com {numero_criancas} crianças, "
    f"que gosta de {atividade}."
)

cliente = genai.Client(api_key=api_key)

resposta = cliente.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="Você é um assistente de roteiro de viagens.",
        temperature=0.7,
    ),
)

resposta_em_texto = resposta.text
print(resposta_em_texto)
