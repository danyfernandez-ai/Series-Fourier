
# Reconstrucción de señales con series de Fourier

Aplicación con interfaz gráfica que reconstruye señales periódicas (cuadrada, triangular y diente de sierra) mediante series de Fourier, y analiza el efecto del número de armónicos sobre el error de reconstrucción.

**Universidad del Cauca** · Facultad de Ingeniería Electrónica y Telecomunicaciones · Señales y Sistemas


## Funcionalidades

- Selección de la señal: cuadrada, triangular o diente de sierra (T = 2 s, amplitud 1).
- Número de armónicos N: 1, 2, 3, 5, 10, 20, 50, 100, 150 y 200.
- Intervalo temporal de visualización configurable.
- Gráficas: señal original, reconstruida, error |x(t) − x̂(t)| y espectro de líneas.
- Métricas: error medio cuadrático, potencia original y potencia reconstruida.
- Exportación de la tabla de resultados a CSV.

## Requisitos

- Python 3.10 o superior (con Tkinter, incluido en la instalación estándar)
- NumPy y Matplotlib
- Instalación de dependencias y librerias : pip install -r requirements.txt

## Instalación y ejecución

```
Descargar el archivo ZIP de proyecto o clonar el repositorio , y ejecutar el archivo main.py
haciendo doble clic sobre él o también, abrir una terminal en la raíz del proyecto y ejecutar: python main.py 
```

## Estructura del proyecto

- `main.py`: punto de entrada de la aplicación.
- `Core/signals.py`: definición de las señales originales.
- `Core/fourier.py`: coeficientes de Fourier, reconstrucción y espectro.
- `Core/metricas.py`: error medio cuadrático, potencias y tabla de resultados.
- `Interfaz/interfaz.py`: interfaz gráfica (Tkinter + Matplotlib).

## Notas

El cálculo se hizo sin funciones de alto nivel (sin FFT ni funciones de señales de SciPy). El error medio cuadrático sigue la fórmula del enunciado, que incluye raíz cuadrada.





