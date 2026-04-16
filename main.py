from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Lê e carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém as chaves de API dos modelos a partir das variáveis de ambiente
api_key_openia = os.getenv("OPENAI_API_KEY")
api_key_groq = os.getenv("GROQ_API_KEY")
api_key_gemini = os.getenv("GEMINI_API_KEY")

# Configura os modelos de linguagem com as chaves de API e outras opções
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

# Criação de um modelo/template de prompt usando a classe PromptTemplate
modelo_de_prompt = PromptTemplate(
    template = """
    Crie um roteiro de viagem de {dias}dias,
    para uma família com {numero_criancas} crianças,
    que gostam de {atividade}
    """
)

# Formatação do prompt usando o modelo/template criado, preenchendo os valores das variáveis
prompt = modelo_de_prompt.format(
    dias=numero_dias, 
    numero_criancas=numero_criancas, 
    atividade=atividade
)

# Escolha do modelo de linguagem a ser utilizado 
modelo = modelo_gemini

# Invocação do modelo de linguagem com o prompt formatado e obtenção da resposta
resposta = modelo.invoke(prompt)

# Impressão do conteúdo da resposta obtida do modelo de linguagem
print(resposta.content)
