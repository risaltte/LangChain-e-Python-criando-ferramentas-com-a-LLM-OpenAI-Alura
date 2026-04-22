from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

# ============================================================
# PASSO 1: CONFIGURACOES INICIAIS
# ============================================================
#
# set_debug(True) faz o LangChain mostrar detalhes internos da execucao,
# o que pode ser util para estudar o fluxo ou depurar problemas.
# set_debug(True)

# Carrega as variaveis de ambiente do arquivo .env.
load_dotenv()

# Le as chaves dos provedores que serao usados nos exemplos.
api_key_openia = os.getenv("OPENAI_API_KEY")
api_key_groq = os.getenv("GROQ_API_KEY")
api_key_gemini = os.getenv("GEMINI_API_KEY")

# ============================================================
# PASSO 2: DEFINIR A ESTRUTURA DAS RESPOSTAS
# ============================================================
#
# Pydantic e uma biblioteca muito usada em Python para definir estruturas
# de dados com validacao.
#
# Aqui criamos dois formatos de saida:
# - Destino: espera cidade e motivo
# - Restaurantes: espera cidade e lista de restaurantes
#
# Isso ajuda o modelo a responder de forma mais organizada.
class Destino(BaseModel):
    cidade:str = Field(description="A cidade recomendada para visitar")
    motivo:str = Field("Motivo pelo qual é interessante visitar essa cidade")

class Restaurantes(BaseModel):
    cidade:str = Field(description="A cidade recomendada para visitar")
    restaurantes:str = Field(description="Restaurantes recomendados na cidade")


# ============================================================
# PASSO 3: CRIAR OS PARSERS DE SAIDA
# ============================================================
#
# Um parser de saida diz ao LangChain como interpretar a resposta do modelo.
# Neste caso, queremos respostas em JSON compatíveis com as classes Pydantic.
parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

# ============================================================
# PASSO 4: CONFIGURAR OS MODELOS
# ============================================================
#
# Aqui criamos tres modelos equivalentes, cada um vindo de um provedor:
# - OpenAI
# - Groq
# - Gemini
#
# Isso e util para comparar provedores sem mudar o restante do fluxo.
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

# ============================================================
# PASSO 5: CRIAR OS PROMPTS
# ============================================================
#
# PromptTemplate e uma forma de montar instrucoes dinamicas.
# Os campos entre chaves, como {interesse} e {cidade}, serao preenchidos
# automaticamente durante a execucao.
#
# Nos dois primeiros prompts tambem incluimos as instrucoes do parser,
# para orientar o modelo a responder no formato esperado.
prompt_cidade = PromptTemplate(
    template = """
    Sugira uma cidade dado o meu interresse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()},
)

prompt_restaurantes = PromptTemplate(
    template = """
    Sugira restaurantes populares entre locais em {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()},
)

prompt_cultural = PromptTemplate(
    template = """
    Sugira atividades e locais culturais em {cidade}.
    """
)


# ============================================================
# PASSO 6: ESCOLHER QUAL MODELO SERA USADO
# ============================================================
#
# Apesar de termos configurado tres provedores, podemos escolher qual deles
# sera usado pela cadeia principal trocando apenas esta variavel.
modelo = modelo_groq

# ============================================================
# PASSO 7: MONTAR AS CADEIAS
# ============================================================
#
# Cada cadeia representa uma etapa do raciocinio:
#
# cadeia_1:
# recebe um interesse e sugere uma cidade em formato estruturado
#
# cadeia_2:
# recebe a cidade e sugere restaurantes
#
# cadeia_3:
# recebe a cidade e sugere atividades culturais em texto livre
#
# Depois juntamos tudo em uma cadeia maior, onde a saida de uma etapa
# alimenta a proxima.
cadeia_1 = prompt_cidade | modelo | parseador_destino
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia_1 | cadeia_2 | cadeia_3)


# ============================================================
# PASSO 8: EXECUTAR O FLUXO
# ============================================================
#
# Aqui comecamos o fluxo informando apenas o interesse.
# O restante e encadeado automaticamente pelas cadeias.
resposta = cadeia.invoke({
    "interesse": "praia"
})

# Exibe o resultado final.
print(resposta)
