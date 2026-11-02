"""
interfaz.py
Interfaz gráfica (Tkinter + Matplotlib) para la reconstrucción de señales con series de Fourier.

RF1: interfaz interactiva con actualización inmediata.
RF2: selección de señal (A cuadrada, B triangular, C diente de sierra).
RF3: número de armónicos N (valores del enunciado) e intervalo temporal.
RF4: gráficas de señal original, reconstruida, error |x - x_rec| y espectro de líneas.
RF5: métricas MSE, potencia original y potencia reconstruida.
RNF1: los errores de entrada se muestran en pantalla sin cerrar la aplicación.

La parte de cálculo y la parte de dibujo son funciones independientes de Tkinter,
de modo que se pueden probar por separado.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Core.fourier import (ARMONICOS_PERMITIDOS, coeficientes_analiticos, espectro_lineas,
                     reconstruir, validar_armonicos)
from Core.metricas import calcular_metricas, exportar_csv, tabla_resultados
from Core.signals import AMPLITUD, F0, SENALES, T, obtener_senal

# ---------------------------------------------------------------- Constantes
INTERVALO_INICIAL = (-T, T)       # intervalo temporal por defecto [s]
N_INICIAL = 10
MAX_PERIODOS = 20                 # ancho máximo del intervalo visualizado
MAX_ABS_TIEMPO = 1000 * T         # evita pérdida de precisión con tiempos enormes
MUESTRAS_MIN = 4000
MUESTRAS_MAX = 100000
MUESTRAS_POR_CICLO = 30           # para que el armónico más alto se vea bien


# ====================================================================== Lógica
def parsear_intervalo(texto_min, texto_max):
    """Convierte los textos del intervalo a números y valida que sea un intervalo razonable."""
    try:
        t_min = float(texto_min.strip().replace(",", "."))
        t_max = float(texto_max.strip().replace(",", "."))
    except ValueError:
        raise ValueError("El intervalo temporal debe contener números (por ejemplo -1 y 1).")
    if not (np.isfinite(t_min) and np.isfinite(t_max)):
        raise ValueError("El intervalo temporal debe contener valores finitos.")
    if t_min >= t_max:
        raise ValueError("El tiempo inicial debe ser menor que el tiempo final.")
    if abs(t_min) > MAX_ABS_TIEMPO or abs(t_max) > MAX_ABS_TIEMPO:
        raise ValueError(f"Los tiempos deben estar entre -{MAX_ABS_TIEMPO:g} y {MAX_ABS_TIEMPO:g} s.")
    if (t_max - t_min) > MAX_PERIODOS * T:
        raise ValueError(f"El intervalo no puede abarcar más de {MAX_PERIODOS} periodos "
                         f"({MAX_PERIODOS * T:g} s).")
    return t_min, t_max


def calcular(nombre, N, t_min, t_max):
    """Calcula señal original, reconstruida, error, espectro y métricas para la interfaz."""
    N = validar_armonicos(N)
    periodos = (t_max - t_min) / T
    muestras = int(min(max(MUESTRAS_MIN, MUESTRAS_POR_CICLO * N * periodos), MUESTRAS_MAX))
    t = np.linspace(t_min, t_max, muestras)

    a0, a, b = coeficientes_analiticos(nombre, N)
    x = obtener_senal(nombre, t)
    x_rec = reconstruir(t, a0, a, b)
    n, amplitudes = espectro_lineas(a0, a, b)
    return {
        "N": N, "t": t, "x": x, "x_rec": x_rec, "error": np.abs(x - x_rec),
        "n": n, "amplitudes": amplitudes,
        "metricas": calcular_metricas(nombre, N, a0, a, b),
    }


# ===================================================================== Dibujo
def _ejes_tiempo(ax, titulo, t_min, t_max, y_min, y_max):
    """Formato común de las gráficas temporales: límites fijos, ejes en cero y cuadrícula."""
    ax.set_title(titulo, fontsize=10)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("t [s]")
    ax.grid(True, linestyle="--", alpha=0.5)


def dibujar_graficas(fig, nombre, datos, t_min, t_max):
    """Dibuja las cuatro gráficas del RF4 en la figura `fig`."""
    fig.clear()
    t, x, x_rec = datos["t"], datos["x"], datos["x_rec"]
    lim = 1.5 * AMPLITUD
    ax1, ax2, ax3, ax4 = (fig.add_subplot(2, 2, k) for k in (1, 2, 3, 4))

    # a) Señal original
    ax1.plot(t, x, color="tab:blue", linewidth=1.6)
    _ejes_tiempo(ax1, f"Original — {nombre}", t_min, t_max, -lim, lim)
    ax1.set_ylabel("x(t)")

    # b) Señal reconstruida (con la original como referencia tenue)
    ax2.plot(t, x, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="Original")
    ax2.plot(t, x_rec, color="tab:red", linewidth=1.4, label=f"Reconstruida (N = {datos['N']})")
    _ejes_tiempo(ax2, "Señal reconstruida", t_min, t_max, -lim, lim)
    ax2.set_ylabel("x̂(t)")
    ax2.legend(loc="upper right", fontsize=8)

    # c) Señal de error |x(t) - x̂(t)|
    error = datos["error"]
    tope = max(float(np.max(error)) * 1.1, 1e-12)
    ax3.plot(t, error, color="tab:green", linewidth=1.2)
    _ejes_tiempo(ax3, "Error  |x(t) − x̂(t)|", t_min, t_max, 0.0, tope)
    ax3.set_ylabel("Error")

    # d) Espectro de líneas en función de la frecuencia fundamental
    n, amp = datos["n"], datos["amplitudes"]
    marcador = "C0o" if len(n) <= 60 else " "
    ax4.stem(n, amp, linefmt="C0-", markerfmt=marcador, basefmt="k-")
    ax4.set_title(f"Espectro de líneas (f0 = {F0:g} Hz)", fontsize=10)
    ax4.set_xlim(-0.5, n[-1] + 0.5)
    ax4.set_ylim(0, max(float(np.max(amp)) * 1.15, 1e-12))
    ax4.set_xlabel("n  (frecuencia = n · f0)")
    ax4.set_ylabel("Amplitud")
    ax4.grid(True, linestyle="--", alpha=0.5)


# ================================================================== Interfaz
class AplicacionFourier:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Reconstrucción de señales con series de Fourier")
        raiz.geometry("1280x780")
        raiz.minsize(1000, 640)

        self.var_senal = tk.StringVar(value=list(SENALES)[0])
        self.var_N = tk.StringVar(value=str(N_INICIAL))
        self.var_tmin = tk.StringVar(value=f"{INTERVALO_INICIAL[0]:g}")
        self.var_tmax = tk.StringVar(value=f"{INTERVALO_INICIAL[1]:g}")
        self.var_mse = tk.StringVar(value="—")
        self.var_p_orig = tk.StringVar(value="—")
        self.var_p_rec = tk.StringVar(value="—")
        self.var_p_parseval = tk.StringVar(value="—")
        self.var_estado = tk.StringVar(value="")

        self._crear_panel_control()
        self._crear_area_graficas()
        raiz.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.actualizar()

    # ---------------------------------------------------------- construcción
    def _crear_panel_control(self):
        panel = ttk.Frame(self.raiz, padding=10)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        marco = ttk.LabelFrame(panel, text="Parámetros", padding=10)
        marco.pack(fill=tk.X)

        ttk.Label(marco, text="Señal:").pack(anchor="w")
        combo_senal = ttk.Combobox(marco, textvariable=self.var_senal, values=list(SENALES),
                                   state="readonly", width=28)
        combo_senal.pack(fill=tk.X, pady=(0, 8))
        combo_senal.bind("<<ComboboxSelected>>", self.actualizar)

        ttk.Label(marco, text="Número de armónicos N:").pack(anchor="w")
        combo_n = ttk.Combobox(marco, textvariable=self.var_N, values=[str(n) for n in ARMONICOS_PERMITIDOS],
                               state="readonly", width=10)
        combo_n.pack(anchor="w", pady=(0, 8))
        combo_n.bind("<<ComboboxSelected>>", self.actualizar)

        ttk.Label(marco, text=f"Intervalo temporal [s]   (T = {T:g} s):").pack(anchor="w")
        fila = ttk.Frame(marco)
        fila.pack(anchor="w", pady=(0, 8))
        entrada_min = ttk.Entry(fila, textvariable=self.var_tmin, width=8)
        entrada_max = ttk.Entry(fila, textvariable=self.var_tmax, width=8)
        entrada_min.pack(side=tk.LEFT)
        ttk.Label(fila, text="  a  ").pack(side=tk.LEFT)
        entrada_max.pack(side=tk.LEFT)
        for entrada in (entrada_min, entrada_max):
            entrada.bind("<Return>", self.actualizar)

        botones = ttk.Frame(marco)
        botones.pack(fill=tk.X)
        ttk.Button(botones, text="Actualizar", command=self.actualizar).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(botones, text="Restablecer", command=self.restablecer).pack(side=tk.LEFT)

        metricas = ttk.LabelFrame(panel, text="Métricas (sobre un periodo)", padding=10)
        metricas.pack(fill=tk.X, pady=12)
        for texto, variable in (("Error medio cuadrático (MSE):", self.var_mse),
                                ("Potencia señal original:", self.var_p_orig),
                                ("Potencia señal reconstruida:", self.var_p_rec),
                                ("Potencia reconstruida (Parseval):", self.var_p_parseval)):
            ttk.Label(metricas, text=texto).pack(anchor="w")
            ttk.Label(metricas, textvariable=variable, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))

        ttk.Button(panel, text="Exportar tabla de resultados (CSV)",
                   command=self.exportar_tabla).pack(fill=tk.X)

        self.etiqueta_estado = ttk.Label(panel, textvariable=self.var_estado, foreground="red",
                                         wraplength=260, justify=tk.LEFT)
        self.etiqueta_estado.pack(fill=tk.X, pady=10)

    def _crear_area_graficas(self):
        self.figura = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figura, master=self.raiz)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- acciones
    def actualizar(self, _evento=None):
        """Recalcula y redibuja. Cualquier error se muestra en pantalla sin cerrar la app."""
        try:
            t_min, t_max = parsear_intervalo(self.var_tmin.get(), self.var_tmax.get())
            nombre = self.var_senal.get()
            datos = calcular(nombre, self.var_N.get(), t_min, t_max)
        except ValueError as error:
            self.var_estado.set(f"⚠ {error}")
            return
        except Exception as error:   # error inesperado: se informa, la aplicación sigue activa
            self.var_estado.set(f"⚠ Error inesperado: {error}")
            return

        dibujar_graficas(self.figura, nombre, datos, t_min, t_max)
        self.canvas.draw_idle()
        m = datos["metricas"]
        self.var_mse.set(f"{m['mse']:.6f}")
        self.var_p_orig.set(f"{m['potencia_original']:.6f}")
        self.var_p_rec.set(f"{m['potencia_reconstruida']:.6f}")
        self.var_p_parseval.set(f"{m['potencia_parseval']:.6f}")
        self.var_estado.set("")

    def restablecer(self):
        self.var_senal.set(list(SENALES)[0])
        self.var_N.set(str(N_INICIAL))
        self.var_tmin.set(f"{INTERVALO_INICIAL[0]:g}")
        self.var_tmax.set(f"{INTERVALO_INICIAL[1]:g}")
        self.actualizar()

    def exportar_tabla(self):
        """Genera la tabla N vs MSE vs potencia de la señal seleccionada y la guarda en CSV."""
        nombre = self.var_senal.get()
        letra = chr(ord("A") + list(SENALES).index(nombre))
        ruta = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile=f"resultados_senal_{letra}.csv",
                                            filetypes=[("Archivo CSV", "*.csv")])
        if not ruta:
            return
        try:
            exportar_csv(tabla_resultados(nombre), ruta)
        except Exception as error:
            messagebox.showerror("Error al exportar", str(error))
            return
        messagebox.showinfo("Tabla exportada", f"Resultados guardados en:\n{ruta}")

    def _cerrar(self):
        self.raiz.quit()
        self.raiz.destroy()

    def _cerrar(self):
        self.raiz.quit()
        self.raiz.destroy()


def main():
    raiz = tk.Tk()
    AplicacionFourier(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()