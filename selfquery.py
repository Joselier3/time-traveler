from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("../.env")

class TimescaleQuery(BaseModel):
    query: str
    start_date: str
    end_date: str

class Queries(BaseModel):
    queries: list[TimescaleQuery]

def selfQuery(query) -> dict:
    print("Generando parametros de busqueda...")
    sqstart_time = datetime.now()
    SYSTEM_PROMPT = """Imagina que eres un asistente muy preciso, que debe definir parametros de busqueda en una base de datos.
La base de datos en la que buscaras se tratanicano de documentos que resumen activos de estados financerios del banco domi\
CITIBANK. Todos los estados financieros se emiten el 31 DE DICIEMBRE DEL AÑO CORRESPONDIENTE.

Primero, debes identificar las distintas busquedas que deben ser realizadas para dar una decision informada.
Cada busqueda es del tipo TimescaleQuery, compuesta por un query, una fecha de inicio y una fecha de fin. El query\
es una cadena de texto que indica que documento deberia buscar el algoritmo de manera mas precisa que la pregunta que hace\
el usuario.

Luego, un algoritmo de busqueda buscara por similitud las entradas mas parecidas, en el rango de fecha indicado, de todas las\
busquedas devueltas por el asistente.

Dependiendo del tipo de preguntas sera necesario realizar una o varias busquedas.

EJEMPLO 1: Cual es el total de inversiones realizadas por el banco en el 2022?
[
    {
        query: "Total de Inversiones en 2022",
        start_date: "2022-1-1",
        end_date: "2022-12-31"
    }
]

EJEMPLO 2: Cual ha sido el total en caja de los ultimos 3 años?
[
    {
        query: "Total en Caja en 2023",
        start_date: "2023-1-1",
        end_date: "2023-12-31"
    },
    {
        query: "Total en Caja en 2022"
        start_date: "2022-1-1"
        end_date: "2022-12-31"
    },
    {
        query: "Total en Caja en 2021"
        start_date: "2021-1-1"
        end_date: "2021-12-31"
    }
]

Sigue las instrucciones paso a paso, tomandote tu tiempo para devolver una repuesta precisa. Asegurate de devolver\
la respuesta en formato JSON, solo devolviendo el objeto JSON.
"""

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        response_format=Queries
    )

    queries = completion.choices[0].message.parsed
    
    dateFormat = r"%Y-%m-%d %H:%M:%S"
    parsedQueries = []
    for query in queries.queries:
        startDate = datetime.strptime(f"{query.start_date} 00:00:00", dateFormat)
        endDate = datetime.strptime(f"{query.end_date} 23:59:59", dateFormat)
        parsedQueries.append({
            "query": query.query,
            "start_date": startDate,
            "end_date": endDate,
        })

    sqend_time = datetime.now()
    sqdelta = sqend_time - sqstart_time
    print(f"({int(sqdelta.total_seconds()*1000)} ms)")

    return parsedQueries

if __name__=="__main__":
    queries = selfQuery("Cual ha sido el total de activos desde el 2019 hasta el 2023")
    print(queries)