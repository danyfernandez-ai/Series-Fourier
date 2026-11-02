"""
signals.py
Definición de las señales periódicas originales del proyecto:
    Señal A: cuadrada
    Señal B: triangular
    Señal C: diente de sierra



Solo se usan operaciones básicas (módulo, valor absoluto, np.where), sin funciones
especiales de alto nivel
"""

import numpy as np

# ---------------------------------------------------------------- Constantes
T = 2                     # periodo de las señales [s]
AMPLITUD = 1.0               # amplitud de las señales
F0 = 1.0 / T                 # frecuencia fundamental [Hz]
OMEGA_0 = 2.0 * np.pi * F0   # frecuencia angular fundamental [rad/s]


def _validar_tiempo(t):
    """Convierte t a arreglo de floats y verifica que sea un vector válido."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        raise ValueError("El vector de tiempo está vacío.")
    if not np.all(np.isfinite(t)):
        raise ValueError("El vector de tiempo contiene valores no válidos (NaN o inf).")
    return t


def senal_cuadrada(t):
    """Señal A: onda cuadrada de periodo T y amplitud 1."""
    t = _validar_tiempo(t)
    fase = t % T                                  # posición dentro del periodo [0, T)
    return np.where(fase < T / 2.0, AMPLITUD, -AMPLITUD)


def senal_triangular(t):
    """Señal B: onda triangular de periodo T y amplitud 1 (máximo en t = 0)."""
    t = _validar_tiempo(t)
    fase = ((t + T / 2.0) % T) - T / 2.0          # posición en [-T/2, T/2)
    return AMPLITUD * (1.0 - 4.0 * np.abs(fase) / T)


def senal_diente_sierra(t):
    """Señal C: onda diente de sierra de periodo T y amplitud 1 (x(0) = 0)."""
    t = _validar_tiempo(t)
    fase = (t + T / 2.0) % T                      # posición en [0, T)
    return AMPLITUD * (2.0 * fase / T - 1.0)


# Diccionario para que la interfaz seleccione la señal por nombre
SENALES = {
    "Señal A: Cuadrada": senal_cuadrada,
    "Señal B: Triangular": senal_triangular,
    "Señal C: Diente de sierra": senal_diente_sierra,
}


def obtener_senal(nombre, t):
    """Evalúa la señal seleccionada en el vector de tiempo t."""
    if nombre not in SENALES:
        raise ValueError(f"Señal desconocida: '{nombre}'. Opciones: {list(SENALES)}")
    return SENALES[nombre](t)


# ------------------------------------------------------- Prueba rápida del módulo
# permite visualizar las gráficas de las señales definidas anteriormente
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.linspace(-2 * T, 2 * T, 4000)
    fig, ejes = plt.subplots(len(SENALES), 1, figsize=(9, 8), sharex=True)
    for eje, (nombre, funcion) in zip(ejes, SENALES.items()):
        eje.plot(t, funcion(t), linewidth=1.8)
        eje.set_title(nombre)
        # Límites fijos de los ejes
        eje.set_xlim(-2 * T, 2 * T)
        eje.set_ylim(-1.5 * AMPLITUD, 1.5 * AMPLITUD)
        # Marcas legibles: cada T/2 en x y cada 0.5 en y
        eje.set_xticks(np.arange(-2 * T, 2 * T + T / 2, T / 2))
        eje.set_yticks(np.arange(-AMPLITUD, AMPLITUD + 0.5, 0.5))
        # Ejes x e y resaltados en cero
        eje.axhline(0, color="black", linewidth=1.2)
        eje.axvline(0, color="black", linewidth=1.2)
        eje.set_ylabel("x(t)")
        eje.grid(True, linestyle="--", alpha=0.5)
    ejes[-1].set_xlabel("t [s]")
    plt.tight_layout()
    plt.show()