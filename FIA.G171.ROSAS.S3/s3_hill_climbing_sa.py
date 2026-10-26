"""S3 · Búsqueda local: Hill Climbing y Simulated Annealing (2.3)

Problema: ruta mínima de un repartidor que visita 21 ciudades del centro de México una sola vez
y regresa al punto de partida (TSP). Se implementan:
  · Hill Climbing de máxima pendiente (con y sin reinicios aleatorios)
  · Simulated Annealing (recocido simulado)
Ambos usan la misma vecindad: el movimiento 2-opt (invertir un tramo de la ruta).

  Función objetivo: longitud total del recorrido en km (se MINIMIZA)
  Estado          : permutación de las 21 ciudades
  Vecindad        : todas las rutas que resultan de invertir un tramo (2-opt)
  Parámetros SA   : T0 (temperatura inicial), eta (factor de enfriamiento), iteraciones

Ejecutar: python s3_hill_climbing_sa.py
Requiere: pip install numpy matplotlib
Genera:   figuras/convergencia.png, figuras/comparacion.png, figuras/mejor_ruta.png y resultados_s3.json
"""
import itertools
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # genera PNG sin abrir ventana
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
os.makedirs(FIG, exist_ok=True)

# ======================================================================
# 1. Datos (coordenadas aproximadas) y matriz de distancias
# ======================================================================
CIUDADES = {
    "CDMX": (19.43, -99.13), "Texcoco": (19.51, -98.88), "Chalco": (19.26, -98.90),
    "Amecameca": (19.13, -98.77), "Atlautla": (19.02, -98.77), "Cuautla": (18.81, -98.95),
    "Cuernavaca": (18.92, -99.23), "Puebla": (19.04, -98.21), "Tlaxcala": (19.31, -98.24),
    "Apizaco": (19.41, -98.14), "Pachuca": (20.10, -98.76), "Tulancingo": (20.08, -98.37),
    "Toluca": (19.28, -99.66), "Querétaro": (20.59, -100.39), "San Juan del Río": (20.39, -99.99),
    "Tula": (20.05, -99.34), "Tepeji": (19.90, -99.35), "Cholula": (19.06, -98.30),
    "Atlixco": (18.91, -98.43), "Izúcar": (18.60, -98.47), "Tehuacán": (18.46, -97.39),
}
NOMBRES = list(CIUDADES)
SINUOSIDAD = 1.25   # distancia por carretera estimada = línea recta x 1.25


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def matriz_distancias(nombres):
    n = len(nombres)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = haversine(CIUDADES[nombres[i]], CIUDADES[nombres[j]]) * SINUOSIDAD
    return D


# ======================================================================
# 2. Función objetivo, vecindad 2-opt y utilidades
# ======================================================================
def longitud(ruta, D):
    return sum(D[ruta[i], ruta[(i + 1) % len(ruta)]] for i in range(len(ruta)))


def delta_2opt(ruta, i, j, D):
    """Cambio de longitud al invertir el tramo ruta[i..j] (solo se recalculan 2 aristas)."""
    n = len(ruta)
    a, b = ruta[i - 1], ruta[i]
    c, d = ruta[j], ruta[(j + 1) % n]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


def aplicar_2opt(ruta, i, j):
    return ruta[:i] + ruta[i:j + 1][::-1] + ruta[j + 1:]


def ruta_aleatoria(n, rng):
    r = list(range(n))
    rng.shuffle(r)
    return r


# ======================================================================
# 3. Hill Climbing de máxima pendiente
# ======================================================================
def hill_climbing(D, rng, ruta=None):
    """Evalúa TODA la vecindad 2-opt, se mueve al mejor vecino y se detiene en un óptimo local."""
    n = len(D)
    ruta = ruta or ruta_aleatoria(n, rng)
    hist = [longitud(ruta, D)]
    evals = 0
    while True:
        mejor_d, mejor_mov = 0.0, None
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d = delta_2opt(ruta, i, j, D)
                evals += 1
                if d < mejor_d - 1e-9:
                    mejor_d, mejor_mov = d, (i, j)
        if mejor_mov is None:
            break                                   # óptimo local: ningún vecino mejora
        ruta = aplicar_2opt(ruta, *mejor_mov)
        hist.append(hist[-1] + mejor_d)
    return ruta, hist[-1], hist, evals


def hill_climbing_reinicios(D, rng, reinicios=20):
    mejor, mejor_l, curva, evals = None, math.inf, [], 0
    for _ in range(reinicios):
        r, l, _, e = hill_climbing(D, rng)
        evals += e
        if l < mejor_l:
            mejor, mejor_l = r, l
        curva.append(mejor_l)
    return mejor, mejor_l, curva, evals


# ======================================================================
# 4. Simulated Annealing
# ======================================================================
def simulated_annealing(D, rng, T0=50.0, eta=0.9995, iteraciones=20000):
    """Acepta siempre los vecinos mejores y los peores con probabilidad exp(-delta/T).
    La temperatura baja en cada iteración: T <- T * eta."""
    n = len(D)
    ruta = ruta_aleatoria(n, rng)
    actual = longitud(ruta, D)
    mejor, mejor_l = ruta[:], actual
    T = T0
    curva_mejor, curva_actual = [mejor_l], [actual]
    for _ in range(iteraciones):
        i, j = sorted(rng.sample(range(1, n), 2))
        d = delta_2opt(ruta, i, j, D)
        if d < 0 or rng.random() < math.exp(-d / T):
            ruta = aplicar_2opt(ruta, i, j)
            actual += d
            if actual < mejor_l - 1e-9:
                mejor, mejor_l = ruta[:], actual
        T *= eta
        curva_mejor.append(mejor_l)
        curva_actual.append(actual)
    return mejor, mejor_l, curva_mejor, curva_actual


# ======================================================================
# 5. Verificación: óptimo exacto en una instancia pequeña (Held-Karp)
# ======================================================================
def optimo_exacto(D):
    """Programación dinámica de Held-Karp: óptimo del TSP para n pequeño (aquí n = 12)."""
    n = len(D)
    INF = float("inf")
    dp = {(1 << k, k): D[0, k] for k in range(1, n)}
    for tam in range(2, n):
        for sub in itertools.combinations(range(1, n), tam):
            mask = sum(1 << k for k in sub)
            for k in sub:
                prev = mask ^ (1 << k)
                dp[(mask, k)] = min(dp[(prev, m)] + D[m, k] for m in sub if m != k)
    full = sum(1 << k for k in range(1, n))
    return min(dp[(full, k)] + D[k, 0] for k in range(1, n))


# ======================================================================
# 6. Experimentos
# ======================================================================
def experimento(D, corridas=30):
    res = {"aleatorio": [], "hc": [], "hc_reinicios": [], "sa": []}
    tiempos = {k: [] for k in res}
    curvas_sa = []
    for s in range(corridas):
        rng = random.Random(s)
        t = time.perf_counter(); r = ruta_aleatoria(len(D), rng); res["aleatorio"].append(longitud(r, D)); tiempos["aleatorio"].append(time.perf_counter() - t)
        t = time.perf_counter(); res["hc"].append(hill_climbing(D, random.Random(s))[1]); tiempos["hc"].append(time.perf_counter() - t)
        t = time.perf_counter(); res["hc_reinicios"].append(hill_climbing_reinicios(D, random.Random(s))[1]); tiempos["hc_reinicios"].append(time.perf_counter() - t)
        t = time.perf_counter(); _, l, cm, _ = simulated_annealing(D, random.Random(s)); res["sa"].append(l); tiempos["sa"].append(time.perf_counter() - t)
        curvas_sa.append(cm)
    return res, tiempos, np.array(curvas_sa)


def sensibilidad_eta(D, etas=(0.99, 0.999, 0.9995, 0.9999), corridas=15):
    filas = []
    for e in etas:
        f = [simulated_annealing(D, random.Random(s), eta=e)[1] for s in range(corridas)]
        filas.append({"eta": e, "T_final": 50.0 * e ** 20000, "media": float(np.mean(f)), "desv": float(np.std(f)), "mejor": float(np.min(f))})
    return filas


# ======================================================================
# 7. Gráficas
# ======================================================================
VINO, ORO, GRIS, TINTA = "#7A1F3D", "#C9A227", "#8C8C8C", "#2B2024"


def graficas(D, res, curvas_sa, mejor_ruta, hc_hist, hc_reinicios_curva, sa_actual):
    # (a) convergencia
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(curvas_sa.shape[1])
    ax[0].fill_between(x, curvas_sa.min(0), curvas_sa.max(0), color=ORO, alpha=0.3, label="mín–máx (30 corridas)")
    ax[0].plot(x, curvas_sa.mean(0), color=VINO, lw=2.2, label="SA: media")
    ax[0].set_title("SA · mejor longitud encontrada", fontsize=10, fontweight="bold", loc="left")
    ax[0].set_xlabel("Iteración"); ax[0].set_ylabel("Longitud (km)"); ax[0].legend()
    ax[1].plot(sa_actual, color=GRIS, lw=0.7, label="solución actual")
    ax[1].set_title("SA · la solución actual sube y baja (acepta peores)", fontsize=10, fontweight="bold", loc="left")
    ax[1].set_xlabel("Iteración"); ax[1].set_ylabel("Longitud (km)")
    ax[2].step(range(len(hc_hist)), hc_hist, color=VINO, lw=2, where="post", label="HC (una corrida)")
    ax[2].set_title("Hill Climbing · cada paso mejora hasta el óptimo local", fontsize=10, fontweight="bold", loc="left")
    ax[2].set_xlabel("Paso"); ax[2].set_ylabel("Longitud (km)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "convergencia.png"), dpi=150); plt.close(fig)
    # (b) comparación
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    etiquetas = ["Hill Climbing", "HC + 20 reinicios", "Simulated Annealing"]
    datos = [res["hc"], res["hc_reinicios"], res["sa"]]
    bp = ax.boxplot(datos, patch_artist=True, tick_labels=etiquetas, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#B56A82", ORO, VINO]):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    for m in bp["medians"]:
        m.set_color("black")
    ax.set_ylabel("Longitud de la ruta (km)")
    ax.set_title("Comparación en 30 corridas (menor es mejor)\nLa ruta aleatoria (≈3,150 km) queda fuera de la escala", fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "comparacion.png"), dpi=150); plt.close(fig)
    # (c) mejor ruta
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    pts = [CIUDADES[NOMBRES[i]] for i in mejor_ruta] + [CIUDADES[NOMBRES[mejor_ruta[0]]]]
    ax.plot([p[1] for p in pts], [p[0] for p in pts], "-", color=VINO, lw=2.2, zorder=1)
    for k, i in enumerate(mejor_ruta):
        la, lo = CIUDADES[NOMBRES[i]]
        ax.scatter(lo, la, s=95, color="white", edgecolor=VINO, linewidth=2, zorder=3)
        ax.text(lo, la, str(k + 1), ha="center", va="center", fontsize=7, fontweight="bold", color=VINO, zorder=4)
        ax.annotate(NOMBRES[i], (lo, la), textcoords="offset points", xytext=(6, 6), fontsize=7.5, color=TINTA)
    ax.set_title(f"Mejor ruta de SA · {longitud(mejor_ruta, D):.1f} km", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mejor_ruta.png"), dpi=150); plt.close(fig)


def main():
    D = matriz_distancias(NOMBRES)
    n = len(NOMBRES)
    res, tiempos, curvas_sa = experimento(D)
    print(f"=== TSP con {n} ciudades · 30 corridas por método ===")
    print(f"{'Método':22}{'Mejor':>9}{'Media':>9}{'Desv.':>8}{'Peor':>9}{'Tiempo (s)':>12}")
    nombres = {"aleatorio": "Ruta aleatoria", "hc": "Hill Climbing", "hc_reinicios": "HC + 20 reinicios", "sa": "Simulated Annealing"}
    for k, v in res.items():
        print(f"{nombres[k]:22}{min(v):>9.1f}{np.mean(v):>9.1f}{np.std(v):>8.1f}{max(v):>9.1f}{np.mean(tiempos[k]):>12.3f}")

    # corrida de ejemplo para las gráficas
    _, _, hc_hist, _ = hill_climbing(D, random.Random(0))
    _, _, hc_curva, _ = hill_climbing_reinicios(D, random.Random(0))
    mejor_sa, l_sa, _, sa_actual = simulated_annealing(D, random.Random(int(np.argmin(res["sa"]))))
    print("Mejor ruta SA:", " -> ".join(NOMBRES[i] for i in mejor_sa), f"-> {l_sa:.1f} km")

    sens = sensibilidad_eta(D)
    print("\nSensibilidad al factor de enfriamiento eta (T0=50, 20,000 iteraciones, 15 corridas)")
    for f in sens:
        print(f"eta={f['eta']:<7} T_final={f['T_final']:8.4f}  media={f['media']:.1f}  desv={f['desv']:.1f}  mejor={f['mejor']:.1f}")

    # verificación con el óptimo exacto en las primeras 12 ciudades
    sub = NOMBRES[:12]
    Ds = matriz_distancias(sub)
    opt = optimo_exacto(Ds)
    sa12 = [simulated_annealing(Ds, random.Random(s), iteraciones=8000)[1] for s in range(15)]
    hc12 = [hill_climbing(Ds, random.Random(s))[1] for s in range(15)]
    print(f"\nVerificación con 12 ciudades: óptimo exacto (Held-Karp) = {opt:.1f} km")
    print(f"  SA llegó al óptimo en {sum(abs(x - opt) < 1e-6 for x in sa12)}/15 corridas; HC en {sum(abs(x - opt) < 1e-6 for x in hc12)}/15")

    graficas(D, res, curvas_sa, mejor_sa, hc_hist, hc_curva, sa_actual)
    with open(os.path.join(AQUI, "resultados_s3.json"), "w", encoding="utf-8") as f:
        json.dump({"res": res, "tiempos": {k: float(np.mean(v)) for k, v in tiempos.items()}, "sens": sens,
                   "mejor_sa": [NOMBRES[i] for i in mejor_sa], "l_sa": l_sa, "hc_pasos": len(hc_hist) - 1,
                   "verif": {"n": 12, "opt": opt, "sa_ok": int(sum(abs(x - opt) < 1e-6 for x in sa12)),
                             "hc_ok": int(sum(abs(x - opt) < 1e-6 for x in hc12))}}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
