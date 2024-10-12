"""
Defines and properly comments tools to be called from langchain chat models
"""

import statistics
from typing import List
from langchain_core.tools import tool

@tool
def suma(valores: List[float]) -> float:
    """Suma una lista de valores.

    Args:
        valores: lista de numeros
    """
    return sum(valores)

@tool
def promedio(valores: List[float]) -> float:
    """Calcula el promedio de una lista de valores.

    Args:
        valores: lista de numeros
    """
    return statistics.mean(valores)

@tool
def mediana(valores: List[float]) -> float:
    """Calcula la mediana de una lista de valores.

    Args:
        valores: lista de numeros
    """
    return statistics.median(valores)

@tool
def moda(valores: List[float]) -> List[float]:
    """Calcula la moda de una lista de valores.

    Args:
        valores: lista de numeros
    """
    return statistics.multimode(valores)

@tool
def rango(valores: List[float]) -> float:
    """Calcula el rango de una lista de valores.

    Args:
        valores: lista de numeros
    """
    max_val = max(valores)
    min_val = min(valores)
    val_range = max_val - min_val

    return val_range

@tool
def cuantiles(valores: List[float]) -> List[float]:
    """Calcula los cuantiles de una lista de valores.

    Args:
        valores: lista de numeros
    """

    return statistics.quantiles(valores, n=4)

@tool
def varianza(valores: List[float]) -> float:
    """Calcula la varianza de una lista de valores.

    Args:
        valores: lista de numeros
    """
    return statistics.variance(valores)

@tool
def desviacion_estandar(valores: List[float]) -> float:
    """Calcula la desviacion estandar de una lista de valores.

    Args:
        valores: lista de numeros
    """
    return statistics.stdev(valores)

TOOLS_LIST = [suma, promedio, mediana]
TOOLS_DICT = {
    "suma": suma,
    "promedio": promedio,
    "mediana": mediana,
    "moda": moda,
    "rango": rango,
    "cuantiles": cuantiles,
    "varianza": varianza,
    "desviacion_estandar": desviacion_estandar
}