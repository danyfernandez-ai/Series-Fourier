"""
fourier.py
Cálculo de los coeficientes de la serie de Fourier y reconstrucción de la señal.

Forma trigonométrica usada (a0 = valor medio):

    x(t) = a0 + sum_{n=1}^{N} [ a_n cos(n w0 t) + b_n sin(n w0 t) ]

    a0  = (1/T) * integral_0^T x(t) dt
    a_n = (2/T) * integral_0^T x(t) cos(n w0 t) dt
    b_n = (2/T) * integral_0^T x(t) sin(n w0 t) dt

Coeficientes analíticos (n >= 1), con amplitud A = 1 y las convenciones de signals.py:

    Cuadrada (impar):          a0 = 0, a_n = 0,            b_n = 4A/(n*pi)        si n es impar, 0 si es par
    Triangular (par):          a0 = 0, a_n = 8A/(n*pi)^2   si n es impar, 0 si es par,   b_n = 0
    Diente de sierra (impar):  a0 = 0, a_n = 0,            b_n = 2A(-1)^(n+1)/(n*pi)

Solo se usan operaciones básicas de NumPy (sin np.fft ni funciones de integración).
"""

import numpy as np


from Core.signals import AMPLITUD, F0, OMEGA_0, SENALES, T, obtener_senal
from Core.signals import senal_cuadrada, senal_diente_sierra, senal_triangular

# Valores de N permitidos por el enunciado (RF3), en orden creciente
ARMONICOS_PERMITIDOS = (1, 2, 3, 5, 10, 20, 50, 100, 150, 200)


def validar_armonicos(N):
    """Verifica que N sea uno de los valores permitidos y lo devuelve como entero."""
    try:
        valor = float(N)
    except (TypeError, ValueError):
        raise ValueError(f"N debe ser un número entero, se recibió: {N!r}")
    if valor != int(valor) or int(valor) not in ARMONICOS_PERMITIDOS:
        raise ValueError(f"N debe ser uno de {ARMONICOS_PERMITIDOS}, se recibió: {N!r}")
    return int(valor)


def _validar_nombre(nombre):
    if nombre not in SENALES:
        raise ValueError(f"Señal desconocida: '{nombre}'. Opciones: {list(SENALES)}")


def coeficientes_analiticos(nombre, N):
    """
    Coeficientes (a0, a, b) deducidos a mano para la señal indicada.
    a y b son arreglos de longitud N; a[n-1] y b[n-1] corresponden al armónico n.
    """
    _validar_nombre(nombre)
    N = validar_armonicos(N)
    n = np.arange(1, N + 1)
    impares = (n % 2 == 1)
    a0 = 0.0
    a = np.zeros(N)
    b = np.zeros(N)

    funcion = SENALES[nombre]
    if funcion is senal_cuadrada:
        b[impares] = 4.0 * AMPLITUD / (np.pi * n[impares])
    elif funcion is senal_triangular:
        a[impares] = 8.0 * AMPLITUD / (np.pi * n[impares]) ** 2
    elif funcion is senal_diente_sierra:
        b = 2.0 * AMPLITUD * (-1.0) ** (n + 1) / (np.pi * n)
    return a0, a, b


def coeficientes_numericos(nombre, N, muestras=20000):
    """
    Coeficientes calculados con integración numérica propia (regla del punto medio).
    Sirve para verificar las fórmulas analíticas.
    """
    _validar_nombre(nombre)
    N = validar_armonicos(N)
    dt = T / muestras
    t = (np.arange(muestras) + 0.5) * dt          # puntos medios de cada subintervalo
    x = obtener_senal(nombre, t)

    a0 = np.sum(x) * dt / T
    a = np.zeros(N)
    b = np.zeros(N)
    for n in range(1, N + 1):
        a[n - 1] = (2.0 / T) * np.sum(x * np.cos(n * OMEGA_0 * t)) * dt
        b[n - 1] = (2.0 / T) * np.sum(x * np.sin(n * OMEGA_0 * t)) * dt
    return a0, a, b


def reconstruir(t, a0, a, b):
    """Suma de la serie de Fourier truncada a len(a) armónicos, evaluada en t."""
    t = np.asarray(t, dtype=float)
    x_rec = np.full_like(t, a0)
    for n in range(1, len(a) + 1):
        x_rec += a[n - 1] * np.cos(n * OMEGA_0 * t) + b[n - 1] * np.sin(n * OMEGA_0 * t)
    return x_rec


def espectro_lineas(a0, a, b):
    """
    Espectro de líneas de la señal reconstruida.
    Devuelve (n, amplitudes): n = 0, 1, ..., N (múltiplos de la frecuencia fundamental f0)
    y la amplitud |a0| para n = 0 y sqrt(a_n^2 + b_n^2) para n >= 1.
    """
    n = np.arange(0, len(a) + 1)
    amplitudes = np.zeros(len(a) + 1)
    amplitudes[0] = abs(a0)
    amplitudes[1:] = np.sqrt(a ** 2 + b ** 2)
    return n, amplitudes


# ------------------------------------------------------- Prueba rápida del módulo
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1) Verificación: analítico vs numérico
    print("Diferencia máxima |analítico - numérico| con N = 10")
    for nombre in SENALES:
        a0_t, a_t, b_t = coeficientes_analiticos(nombre, 10)
        a0_n, a_n, b_n = coeficientes_numericos(nombre, 10)
        err = max(abs(a0_t - a0_n), np.max(np.abs(a_t - a_n)), np.max(np.abs(b_t - b_n)))
        print(f"  {nombre:28s} {err:.2e}")

    # 2) Reconstrucción con varios N
    t = np.linspace(-T, T, 4000)
    Ns = (1, 3, 10, 50)
    fig, ejes = plt.subplots(len(SENALES), 1, figsize=(9, 9), sharex=True)
    for eje, nombre in zip(ejes, SENALES):
        eje.plot(t, obtener_senal(nombre, t), "k", linewidth=2, label="Original")
        for N in Ns:
            a0, a, b = coeficientes_analiticos(nombre, N)
            eje.plot(t, reconstruir(t, a0, a, b), linewidth=1, label=f"N = {N}")
        eje.set_title(nombre)
        eje.set_xlim(-T, T)
        eje.set_ylim(-1.5 * AMPLITUD, 1.5 * AMPLITUD)
        eje.set_xticks(np.arange(-T, T + T / 2, T / 2))
        eje.set_yticks(np.arange(-AMPLITUD, AMPLITUD + 0.5, 0.5))
        eje.axhline(0, color="black", linewidth=1.2)
        eje.axvline(0, color="black", linewidth=1.2)
        eje.set_ylabel("x(t)")
        eje.grid(True, linestyle="--", alpha=0.5)
        eje.legend(loc="upper right", ncol=5, fontsize=8)
    ejes[-1].set_xlabel("t [s]")
    plt.tight_layout()
    plt.show()