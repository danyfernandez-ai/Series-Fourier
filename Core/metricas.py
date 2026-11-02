"""
metricas.py
Métricas de evaluación de la reconstrucción (RF5) y tabla de resultados para la sustentación.

Definiciones:
    MSE (según el enunciado, incluye la raíz) = sqrt( (1/N) * sum (x[n] - x_rec[n])^2 )
    Potencia en el tiempo   P = (1/N) * sum x[n]^2           (promedio de x^2 en un periodo)
    Potencia por Parseval   P = a0^2 + (1/2) * sum_{n=1}^{N} (a_n^2 + b_n^2)

Las métricas de la tabla se calculan sobre EXACTAMENTE un periodo con un muestreo fijo
(puntos medios), de modo que los resultados no dependen del intervalo visualizado y son
reproducibles (RNF3).
"""

import csv

import numpy as np

from Core.fourier import ARMONICOS_PERMITIDOS, coeficientes_analiticos, reconstruir, validar_armonicos
from Core.signals import SENALES, T, obtener_senal

MUESTRAS_POR_PERIODO = 20000


def _validar_vectores(x, x_rec):
    x = np.asarray(x, dtype=float)
    x_rec = np.asarray(x_rec, dtype=float)
    if x.size == 0 or x.shape != x_rec.shape:
        raise ValueError("Las señales deben tener la misma longitud y no estar vacías.")
    return x, x_rec


def mse(x, x_rec):
    """Error medio cuadrático según la fórmula del enunciado (con raíz cuadrada)."""
    x, x_rec = _validar_vectores(x, x_rec)
    return float(np.sqrt(np.mean((x - x_rec) ** 2)))


def potencia_tiempo(x):
    """Potencia promedio de la señal calculada en el dominio del tiempo."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("La señal está vacía.")
    return float(np.mean(x ** 2))


def potencia_parseval(a0, a, b):
    """Potencia de la señal reconstruida a partir de sus coeficientes (Parseval)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a0 ** 2 + 0.5 * np.sum(a ** 2 + b ** 2))


def _un_periodo(muestras=MUESTRAS_POR_PERIODO):
    """Vector de tiempo de un periodo (puntos medios, evita caer justo en una discontinuidad)."""
    dt = T / muestras
    return (np.arange(muestras) + 0.5) * dt


def calcular_metricas(nombre, N, a0=None, a=None, b=None, muestras=MUESTRAS_POR_PERIODO):
    """
    Métricas de la señal `nombre` reconstruida con N armónicos, evaluadas en un periodo.
    Si no se pasan los coeficientes, se calculan los analíticos.
    """
    N = validar_armonicos(N)
    if a0 is None:
        a0, a, b = coeficientes_analiticos(nombre, N)
    t = _un_periodo(muestras)
    x = obtener_senal(nombre, t)
    x_rec = reconstruir(t, a0, a, b)
    return {
        "N": N,
        "mse": mse(x, x_rec),
        "potencia_original": potencia_tiempo(x),
        "potencia_reconstruida": potencia_tiempo(x_rec),
        "potencia_parseval": potencia_parseval(a0, a, b),
    }


def tabla_resultados(nombre, muestras=MUESTRAS_POR_PERIODO):
    """Lista de métricas para cada N permitido (tabla pedida para la sustentación)."""
    n_max = max(ARMONICOS_PERMITIDOS)
    a0, a_max, b_max = coeficientes_analiticos(nombre, n_max)   # se calcula una sola vez
    filas = []
    for N in ARMONICOS_PERMITIDOS:
        filas.append(calcular_metricas(nombre, N, a0, a_max[:N], b_max[:N], muestras))
    return filas




def exportar_csv(filas, ruta, decimal=","):
    
    
    
    separador = ";" if decimal == "," else ","

    def num(valor, decimales):
        return f"{valor:.{decimales}f}".replace(".", decimal)

    encabezado = ["N armonicos", "Error medio cuadratico", "Potencia original",
                  "Potencia reconstruida (tiempo)", "Potencia reconstruida (Parseval)"]
    try:
        with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
            escritor = csv.writer(archivo, delimiter=separador)
            escritor.writerow(encabezado)
            for f in filas:
                escritor.writerow([f["N"], num(f["mse"], 8), num(f["potencia_original"], 6),
                                   num(f["potencia_reconstruida"], 6), num(f["potencia_parseval"], 6)])
    
    except OSError as error:
        raise OSError(f"No se pudo guardar el archivo '{ruta}': {error}")


# ------------------------------------------------------- Prueba rápida del módulo
"""
 al ejecutar metricas.py se crean 3 archivos csv que contienen las metricas para cada señal 
 y además se muestran las gráficas que representan la variación de la potencia y del  error RMSE cuando
 N crece , N = número de armónicos
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(2, len(SENALES), figsize=(14, 7))
    for i, nombre in enumerate(SENALES):
        filas = tabla_resultados(nombre)
        letra = chr(ord("A") + i)
        exportar_csv(filas, f"resultados_senal_{letra}.csv")

        print(f"\n{nombre}")
        print(f"{'N':>5} {'MSE':>12} {'P original':>12} {'P recon.':>12} {'P Parseval':>12}")
        for f in filas:
            print(f"{f['N']:>5} {f['mse']:>12.6f} {f['potencia_original']:>12.6f} "
                  f"{f['potencia_reconstruida']:>12.6f} {f['potencia_parseval']:>12.6f}")

        Ns = [f["N"] for f in filas]
        # MSE vs N
        ax = ejes[0, i]
        ax.loglog(Ns, [f["mse"] for f in filas], "o-")
        ax.set_title(f"{nombre}\nError vs. N")
        ax.set_xlabel("N (armónicos)")
        ax.set_ylabel("MSE")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        # Potencia vs N
        ax = ejes[1, i]
        ax.semilogx(Ns, [f["potencia_reconstruida"] for f in filas], "o-", label="Reconstruida")
        ax.axhline(filas[0]["potencia_original"], color="red", linestyle="--", label="Original")
        ax.set_title("Potencia vs. N")
        ax.set_xlabel("N (armónicos)")
        ax.set_ylabel("Potencia")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.show()