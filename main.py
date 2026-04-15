from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key_openia = os.getenv("OPENAI_API_KEY")
api_key_groq = os.getenv("GROQ_API_KEY")
api_key_gemini = os.getenv("GEMINI_API_KEY")

modelo_openia = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key_openia
)

modelo_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    api_key=api_key_groq,
)

modelo_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    api_key=api_key_gemini,
)


numero_dias = 5
numero_criancas = 2
atividade = "praia"

prompt = f"Crie um roteiro de viagens para um período de{numero_dias} dias, para uma família com  {numero_criancas} crianças cque buscam atividades relacionadas a {atividade}."

modelo = modelo_gemini

resposta = modelo.invoke(prompt)

print(resposta.content)
