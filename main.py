from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

# Faz o debug da cadeia -monitora e exibe o fluxo
# set_debug(True)

# Lê e carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém as chaves de API dos modelos a partir das variáveis de ambiente
api_key_openia = os.getenv("OPENAI_API_KEY")
api_key_groq = os.getenv("GROQ_API_KEY")
api_key_gemini = os.getenv("GEMINI_API_KEY")

# Define a estrutura de dados para a resposta esperada do modelo de linguagem usando Pydantic
class Destino(BaseModel):
    cidade:str = Field(description="A cidade recomendada para visitar")
    motivo:str = Field("Motivo pelo qual é interessante visitar essa cidade")

# Configura o parser de saída para interpretar a resposta do modelo de linguagem como um objeto JSON seguindo a estrutura definida pela classe Destino
parseador = JsonOutputParser(pydantic_object=Destino)

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

# Configura o prompt para solicitar ao modelo de linguagem uma sugestão de cidade com base em um interesse específico, utilizando o formato de saída definido pelo parser
prompt_cidade = PromptTemplate(
    template = """
    Sugira uma cidade dao o meu interresse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()},

)

# Define o modelo LLM a ser utilizado
modelo = modelo_groq

# Configura a cadeia de processamento, encadeando o prompt, o modelo de linguagem e o parser de saída para obter a resposta formatada corretamente
cadeia = prompt_cidade | modelo | parseador

# Executa a cadeia de processamento, passando o interesse como entrada, e obtém a resposta formatada como um objeto Destino
resposta = cadeia.invoke({
    "interesse": "praia"
})

# Exibe a resposta obtida do modelo de linguagem, que deve conter a cidade recomendada e o motivo para visitá-la
print(resposta)
