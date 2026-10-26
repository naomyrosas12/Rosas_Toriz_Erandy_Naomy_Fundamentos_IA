"""S6 · Algoritmo genético (DEAP) para programar los turnos semanales de un equipo de trabajo

Caso empresarial: un negocio con servicio extendido (por ejemplo, una clínica o una tienda) debe
asignar a 8 empleados un turno por día durante una semana. El algoritmo genético busca el horario
que cumple la cobertura mínima y reduce las violaciones de las reglas laborales.

  Cromosoma : lista de 56 genes (8 empleados x 7 días); cada gen es 0=descanso, 1=mañana, 2=tarde, 3=noche
  Fitness   : penalización total a MINIMIZAR (0 = horario perfecto)
  Operadores: selección por torneo, cruce por empleado, mutación uniforme y elitismo
  Corrida   : población de 100 individuos y 30 generaciones

Equipo: Bernal Najera Alexandra · Marin Flor Julissa · Rosas Toriz Erandy Naomy · Suarez Melonie Magdalena
Ejecutar: python s6_algoritmo_genetico.py
Requiere: pip install deap numpy matplotlib
Genera:   figuras/convergencia.png, figuras/horario_optimo.png, figuras/sensibilidad.png y resultados_s6.json
"""
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # genera PNG sin abrir ventana
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
os.makedirs(FIG, exist_ok=True)

# ======================================================================
# 1. Datos del problema
# ======================================================================
EMPLEADOS = ["Ana", "Beto", "Carla", "Diego", "Elena", "Fabián", "Gaby", "Hugo"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
TURNOS = {0: "Descanso", 1: "Mañana", 2: "Tarde", 3: "Noche"}
NE, ND = len(EMPLEADOS), len(DIAS)
LARGO = NE * ND                                  # genes del cromosoma
COBERTURA = {1: 2, 2: 2, 3: 1}                   # personas mínimas por turno y día
MAX_DIAS, MIN_DIAS, MAX_NOCHES = 5, 4, 3         # días trabajados por semana y noches por semana
SIN_NOCHE = {2, 5}                               # Carla y Fabián no pueden trabajar de noche
PESOS = {"cobertura": 10, "sobrecobertura": 2, "dias_max": 8, "dias_min": 3,
         "noche_manana": 6, "sin_noche": 15, "noches_max": 5, "equidad_noches": 1}

# Parámetros del algoritmo genético
POBLACION, GENERACIONES = 100, 30
CXPB, MUTPB, INDPB, TORNEO, ELITE = 0.8, 0.3, 0.04, 3, 2


def matriz(ind):
    """Convierte el cromosoma plano en una matriz empleados x días."""
    return np.array(ind, dtype=int).reshape(NE, ND)


# ======================================================================
# 2. Función de aptitud (fitness): suma ponderada de violaciones
# ======================================================================
def penalizaciones(ind):
    m = matriz(ind)
    p = dict.fromkeys(PESOS, 0.0)
    for d in range(ND):
        for t, minimo in COBERTURA.items():
            n = int(np.sum(m[:, d] == t))
            p["cobertura"] += max(0, minimo - n)               # faltan personas
            p["sobrecobertura"] += max(0, n - (minimo + 1))    # sobran personas
    trabajados = (m > 0).sum(axis=1)
    p["dias_max"] = float(np.sum(np.maximum(0, trabajados - MAX_DIAS)))
    p["dias_min"] = float(np.sum(np.maximum(0, MIN_DIAS - trabajados)))
    p["noche_manana"] = float(np.sum((m[:, :-1] == 3) & (m[:, 1:] == 1)))   # sin descanso suficiente
    p["sin_noche"] = float(sum(np.sum(m[e] == 3) for e in SIN_NOCHE))
    noches = (m == 3).sum(axis=1)
    p["noches_max"] = float(np.sum(np.maximum(0, noches - MAX_NOCHES)))
    p["equidad_noches"] = float(np.var(noches))
    return p


def aptitud(ind):
    p = penalizaciones(ind)
    return (sum(PESOS[k] * v for k, v in p.items()),)


# ======================================================================
# 3. Operadores genéticos
# ======================================================================
def cruce_por_empleado(a, b):
    """Intercambia filas completas (el horario de un empleado) entre dos padres."""
    for e in range(NE):
        if random.random() < 0.5:
            i = e * ND
            a[i:i + ND], b[i:i + ND] = b[i:i + ND], a[i:i + ND]
    return a, b


if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMin)


def crear_toolbox(indpb=INDPB):
    tb = base.Toolbox()
    tb.register("gen", lambda: random.choices([0, 1, 2, 3], weights=[3, 3, 3, 1])[0])
    tb.register("individuo", tools.initRepeat, creator.Individuo, tb.gen, LARGO)
    tb.register("poblacion", tools.initRepeat, list, tb.individuo)
    tb.register("evaluate", aptitud)
    tb.register("select", tools.selTournament, tournsize=TORNEO)
    tb.register("mate", cruce_por_empleado)
    tb.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=indpb)
    return tb


# ======================================================================
# 4. Ciclo evolutivo (con elitismo)
# ======================================================================
def evolucionar(semilla=42, n=POBLACION, ngen=GENERACIONES, indpb=INDPB):
    random.seed(semilla)
    tb = crear_toolbox(indpb)
    pop = tb.poblacion(n=n)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)
    hist = []

    def registrar(g):
        f = [i.fitness.values[0] for i in pop]
        hist.append({"gen": g, "mejor": min(f), "promedio": float(np.mean(f)), "peor": max(f)})

    registrar(0)
    for g in range(1, ngen + 1):
        elite = [tb.clone(i) for i in tools.selBest(pop, ELITE)]
        hijos = [tb.clone(i) for i in tb.select(pop, n - ELITE)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if random.random() < CXPB:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for h in hijos:
            if random.random() < MUTPB:
                tb.mutate(h)
                del h.fitness.values
        for h in hijos:
            if not h.fitness.valid:
                h.fitness.values = tb.evaluate(h)
        pop[:] = hijos + elite
        registrar(g)
    mejor = tools.selBest(pop, 1)[0]
    return mejor, hist


def busqueda_aleatoria(semilla, evaluaciones):
    """Línea base: mismo número de evaluaciones que el AG, pero sin evolución."""
    random.seed(semilla)
    tb = crear_toolbox()
    return min(tb.evaluate(tb.individuo())[0] for _ in range(evaluaciones))


# ======================================================================
# 5. Experimentos y gráficas
# ======================================================================
VERDE, VERDE2, LIMA, CREMA, TINTA = "#14532D", "#3F8F5A", "#A3C93A", "#FAF7F0", "#1B2A22"
COLORES_TURNO = {0: "#E8EDE9", 1: "#F2C94C", 2: "#F2994A", 3: "#2D4A7A"}


def grafica_convergencia(hist, corridas):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    g = [h["gen"] for h in hist]
    ax[0].plot(g, [h["peor"] for h in hist], color="#C0392B", lw=1.4, label="Peor")
    ax[0].plot(g, [h["promedio"] for h in hist], color=LIMA, lw=2, label="Promedio")
    ax[0].plot(g, [h["mejor"] for h in hist], color=VERDE, lw=2.6, label="Mejor")
    ax[0].set_title("Convergencia de una corrida (semilla 42)", fontsize=11, fontweight="bold", loc="left")
    ax[0].set_xlabel("Generación"); ax[0].set_ylabel("Penalización (menor es mejor)"); ax[0].legend()
    M = np.array([[h["mejor"] for h in c] for c in corridas])
    x = np.arange(M.shape[1])
    ax[1].fill_between(x, M.min(0), M.max(0), color=LIMA, alpha=0.35, label="mínimo–máximo")
    ax[1].plot(x, M.mean(0), color=VERDE, lw=2.6, label=f"media de {len(corridas)} corridas")
    ax[1].set_title("Mejor penalización por generación (10 semillas)", fontsize=11, fontweight="bold", loc="left")
    ax[1].set_xlabel("Generación"); ax[1].legend()
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "convergencia.png"), dpi=150); plt.close(fig)


def grafica_horario(mejor, etiqueta):
    m = matriz(mejor)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for e in range(NE):
        for d in range(ND):
            ax.add_patch(plt.Rectangle((d, NE - 1 - e), 0.94, 0.92, color=COLORES_TURNO[m[e, d]]))
            txt = {0: "", 1: "M", 2: "T", 3: "N"}[m[e, d]]
            ax.text(d + 0.47, NE - 1 - e + 0.46, txt, ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if m[e, d] == 3 else TINTA)
    ax.set_xlim(0, ND); ax.set_ylim(0, NE)
    ax.set_xticks([d + 0.47 for d in range(ND)]); ax.set_xticklabels(DIAS)
    ax.set_yticks([NE - 1 - e + 0.46 for e in range(NE)]); ax.set_yticklabels(EMPLEADOS)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES_TURNO[k]) for k in (1, 2, 3, 0)]
    ax.legend(handles, ["Mañana (M)", "Tarde (T)", "Noche (N)", "Descanso"], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    ax.set_title(f"Mejor horario encontrado · {etiqueta}", fontsize=11, fontweight="bold", loc="left", pad=22)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "horario_optimo.png"), dpi=150); plt.close(fig)


def grafica_sensibilidad(sens):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = [f"{s['indpb']}" for s in sens]
    b = ax.bar(x, [s["media"] for s in sens], yerr=[s["desv"] for s in sens], color=[VERDE2, VERDE, LIMA][:len(sens)],
               capsize=5, width=0.55)
    ax.bar_label(b, fmt="%.1f", padding=8, fontsize=9)
    ax.set_xlabel("Probabilidad de mutación por gen (indpb)"); ax.set_ylabel("Penalización final (media ± desv.)")
    ax.set_title("Sensibilidad a la tasa de mutación (10 corridas cada una)", fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sensibilidad.png"), dpi=150); plt.close(fig)


def main():
    mejor, hist = evolucionar(42)
    corridas = [evolucionar(s)[1] for s in range(10)]
    finales = [c[-1]["mejor"] for c in corridas]
    evals = POBLACION + GENERACIONES * (POBLACION - ELITE)
    aleatorio = [busqueda_aleatoria(s, evals) for s in range(10)]
    pen = penalizaciones(mejor)

    print("=== Algoritmo genético (DEAP) · turnos de 8 empleados, 7 días ===")
    print(f"Población={POBLACION}  generaciones={GENERACIONES}  cxpb={CXPB}  mutpb={MUTPB}  indpb={INDPB}  torneo={TORNEO}  élite={ELITE}")
    print(f"Corrida semilla 42: penalización inicial (mejor) = {hist[0]['mejor']:.1f} -> final = {hist[-1]['mejor']:.1f}")
    print(f"10 corridas AG       : media = {np.mean(finales):.1f}  desv = {np.std(finales):.1f}  mejor = {min(finales):.1f}  peor = {max(finales):.1f}")
    print(f"10 búsquedas aleatorias ({evals} evaluaciones): media = {np.mean(aleatorio):.1f}  mejor = {min(aleatorio):.1f}")
    print("Desglose de la mejor solución (valor sin ponderar):")
    for k, v in pen.items():
        print(f"  {k:15} {v:6.2f}  (peso {PESOS[k]})")

    sens = []
    for indpb in (0.01, 0.04, 0.10):
        f = [evolucionar(s, indpb=indpb)[1][-1]["mejor"] for s in range(10)]
        sens.append({"indpb": indpb, "media": float(np.mean(f)), "desv": float(np.std(f))})
        print(f"indpb={indpb}: media final = {np.mean(f):.1f} ± {np.std(f):.1f}")

    grafica_convergencia(hist, corridas)
    grafica_horario(mejor, f"penalización {hist[-1]['mejor']:.1f}")
    grafica_sensibilidad(sens)

    with open(os.path.join(AQUI, "resultados_s6.json"), "w", encoding="utf-8") as f:
        json.dump({"hist": hist, "finales": finales, "aleatorio": aleatorio, "evaluaciones": evals,
                   "desglose": pen, "sens": sens, "mejor": [int(g) for g in mejor],
                   "inicial_mejor": hist[0]["mejor"]}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
