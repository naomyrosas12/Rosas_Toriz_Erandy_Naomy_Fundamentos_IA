"""S4 · Búsqueda ciega: BFS (anchura) y DFS (profundidad) en un laberinto y en el 8-puzzle (2.4)

Se resuelven dos problemas con los mismos algoritmos y se comparan tiempo, nodos expandidos,
longitud del camino y tamaño máximo de la frontera (memoria):
  1) Laberinto en cuadrícula con ciclos (estado = celda; 4 acciones: arriba, derecha, abajo, izquierda)
  2) 8-puzzle (estado = acomodo de las fichas; la acción mueve el hueco)

Representación del grafo: implícita. Cada problema define vecinos(estado) y los algoritmos
generan el grafo conforme lo exploran. Ambos algoritmos usan un conjunto de visitados.

Ejecutar: python s4_bfs_dfs.py
Requiere: pip install numpy matplotlib
Genera:   figuras/laberinto_bfs_dfs.png, figuras/comparacion_tamanos.png y resultados_s4.json
"""
import json
import os
import random
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")  # genera PNG sin abrir ventana
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
os.makedirs(FIG, exist_ok=True)


# ======================================================================
# 1. Algoritmos de búsqueda ciega (genéricos: sirven para cualquier problema)
# ======================================================================
def reconstruir(padre, meta):
    camino = [meta]
    while padre[camino[-1]] is not None:
        camino.append(padre[camino[-1]])
    return camino[::-1]


def bfs(inicio, es_meta, vecinos):
    """Búsqueda en anchura: frontera = COLA (FIFO). Encuentra el camino con menos pasos."""
    t0 = time.perf_counter()
    padre = {inicio: None}                 # sirve también como conjunto de visitados
    frontera = deque([inicio])
    expandidos, max_frontera, orden = 0, 1, []
    while frontera:
        u = frontera.popleft()
        expandidos += 1
        orden.append(u)
        if es_meta(u):
            break
        for v in vecinos(u):
            if v not in padre:
                padre[v] = u
                frontera.append(v)
        max_frontera = max(max_frontera, len(frontera))
    else:
        return None
    camino = reconstruir(padre, u)
    return {"camino": camino, "pasos": len(camino) - 1, "expandidos": expandidos, "max_frontera": max_frontera,
            "tiempo_ms": (time.perf_counter() - t0) * 1000, "orden": orden}


def dfs(inicio, es_meta, vecinos):
    """Búsqueda en profundidad: frontera = PILA (LIFO). Encuentra un camino, no necesariamente el más corto."""
    t0 = time.perf_counter()
    padre, pila = {}, [(inicio, None)]
    expandidos, max_frontera, orden = 0, 1, []
    while pila:
        u, p = pila.pop()
        if u in padre:
            continue
        padre[u] = p
        expandidos += 1
        orden.append(u)
        if es_meta(u):
            break
        for v in reversed(vecinos(u)):     # reversed: se explora primero el primer vecino
            if v not in padre:
                pila.append((v, u))
        max_frontera = max(max_frontera, len(pila))
    else:
        return None
    camino = reconstruir(padre, u)
    return {"camino": camino, "pasos": len(camino) - 1, "expandidos": expandidos, "max_frontera": max_frontera,
            "tiempo_ms": (time.perf_counter() - t0) * 1000, "orden": orden}


# ======================================================================
# 2. Problema 1: laberinto
# ======================================================================
def generar_laberinto(celdas, semilla, ciclos=0.10):
    """Laberinto de celdas x celdas. 1 = pared, 0 = pasillo. Se talla con DFS aleatorio y luego se
    derriban algunas paredes ('ciclos') para que existan varios caminos posibles."""
    rng = random.Random(semilla)
    n = 2 * celdas + 1
    g = np.ones((n, n), dtype=int)
    pila = [(1, 1)]
    g[1, 1] = 0
    while pila:
        r, c = pila[-1]
        vec = [(r + dr, c + dc, dr, dc) for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
               if 0 < r + dr < n - 1 and 0 < c + dc < n - 1 and g[r + dr, c + dc] == 1]
        if vec:
            nr, nc, dr, dc = rng.choice(vec)
            g[r + dr // 2, c + dc // 2] = 0
            g[nr, nc] = 0
            pila.append((nr, nc))
        else:
            pila.pop()
    paredes = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
               if g[r, c] == 1 and ((g[r - 1, c] == 0 and g[r + 1, c] == 0) or (g[r, c - 1] == 0 and g[r, c + 1] == 0))]
    for r, c in rng.sample(paredes, int(len(paredes) * ciclos)):
        g[r, c] = 0
    return g


def problema_laberinto(g):
    n = g.shape[0]
    inicio, meta = (1, 1), (n - 2, n - 2)

    def vecinos(s):
        r, c = s
        return [(r + dr, c + dc) for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)) if g[r + dr, c + dc] == 0]
    return inicio, (lambda s: s == meta), vecinos


# ======================================================================
# 3. Problema 2: 8-puzzle
# ======================================================================
META_8 = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def vecinos_8(s):
    h = s.index(0)
    r, c = divmod(h, 3)
    res = []
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            l = list(s)
            j = nr * 3 + nc
            l[h], l[j] = l[j], l[h]
            res.append(tuple(l))
    return res


def revolver_8(pasos, semilla):
    """Parte de la meta y aplica movimientos aleatorios (sin deshacer el anterior): estado siempre resoluble."""
    rng = random.Random(semilla)
    s, anterior = META_8, None
    for _ in range(pasos):
        opciones = [v for v in vecinos_8(s) if v != anterior]
        anterior, s = s, rng.choice(opciones)
    return s


# ======================================================================
# 4. Experimentos
# ======================================================================
def resumen(r):
    return {k: r[k] for k in ("pasos", "expandidos", "max_frontera", "tiempo_ms")}


def comparar_tamanos(tamanos=(10, 20, 40, 60, 80), semillas=range(5)):
    filas = []
    for celdas in tamanos:
        acum = {"bfs": [], "dfs": []}
        for s in semillas:
            g = generar_laberinto(celdas, s)
            ini, meta, vec = problema_laberinto(g)
            acum["bfs"].append(resumen(bfs(ini, meta, vec)))
            acum["dfs"].append(resumen(dfs(ini, meta, vec)))
        fila = {"celdas": celdas}
        for alg, lst in acum.items():
            for k in lst[0]:
                fila[f"{alg}_{k}"] = float(np.mean([x[k] for x in lst]))
        filas.append(fila)
    return filas


# ======================================================================
# 5. Gráficas
# ======================================================================
NAVY, AMARILLO, AZUL, GRIS = "#0B1D3A", "#F4C20D", "#2F6FDE", "#D5DBE5"


def dibujar_laberinto(ax, g, r, titulo):
    img = np.where(g == 1, 0.0, 1.0)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ex = np.array(r["orden"])
    ax.scatter(ex[:, 1], ex[:, 0], s=4, color=AZUL, alpha=0.35, marker="s")
    cam = np.array(r["camino"])
    ax.plot(cam[:, 1], cam[:, 0], color="#E4572E", lw=2.2)
    n = g.shape[0]
    ax.scatter([1], [1], s=60, color="#2E9E5B", zorder=5); ax.scatter([n - 2], [n - 2], s=60, color="#D64545", zorder=5)
    ax.set_title(titulo, fontsize=10, fontweight="bold", loc="left"); ax.axis("off")


def graficas(g, rb, rd, filas):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    dibujar_laberinto(ax[0], g, rb, f"BFS · camino de {rb['pasos']} pasos · {rb['expandidos']} nodos expandidos")
    dibujar_laberinto(ax[1], g, rd, f"DFS · camino de {rd['pasos']} pasos · {rd['expandidos']} nodos expandidos")
    fig.text(0.5, 0.02, "Azul = celdas expandidas · naranja = camino solución · verde = inicio · rojo = meta", ha="center", fontsize=9, color="#52606D")
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(os.path.join(FIG, "laberinto_bfs_dfs.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    x = [f["celdas"] for f in filas]
    for a, (k, tit) in zip(ax.flat, [("pasos", "Longitud del camino (pasos)"), ("expandidos", "Nodos expandidos"),
                                     ("max_frontera", "Tamaño máximo de la frontera (memoria)"), ("tiempo_ms", "Tiempo (ms)")]):
        a.plot(x, [f[f"bfs_{k}"] for f in filas], "o-", color=AZUL, lw=2, label="BFS")
        a.plot(x, [f[f"dfs_{k}"] for f in filas], "s-", color="#E4572E", lw=2, label="DFS")
        a.set_title(tit, fontsize=10, fontweight="bold", loc="left"); a.set_xlabel("celdas por lado del laberinto"); a.legend()
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "comparacion_tamanos.png"), dpi=150); plt.close(fig)


def main():
    # ----- Laberinto de 20 x 20 celdas -----
    g = generar_laberinto(20, semilla=3)
    ini, meta, vec = problema_laberinto(g)
    rb, rd = bfs(ini, meta, vec), dfs(ini, meta, vec)
    libres = int((g == 0).sum())
    print(f"=== Laberinto 20x20 celdas (matriz {g.shape[0]}x{g.shape[1]}, {libres} celdas libres) ===")
    print(f"{'':6}{'Pasos':>8}{'Expandidos':>12}{'Frontera máx.':>15}{'Tiempo (ms)':>13}")
    for nombre, r in (("BFS", rb), ("DFS", rd)):
        print(f"{nombre:6}{r['pasos']:>8}{r['expandidos']:>12}{r['max_frontera']:>15}{r['tiempo_ms']:>13.3f}")

    # ----- 8-puzzle -----
    inicio8 = revolver_8(22, semilla=5)
    r8b = bfs(inicio8, lambda s: s == META_8, vecinos_8)
    r8d = dfs(inicio8, lambda s: s == META_8, vecinos_8)
    print(f"\n=== 8-puzzle · inicio {inicio8} ===")
    for nombre, r in (("BFS", r8b), ("DFS", r8d)):
        print(f"{nombre:6}{r['pasos']:>8}{r['expandidos']:>12}{r['max_frontera']:>15}{r['tiempo_ms']:>13.1f}")
    print("Solución óptima (BFS):", r8b["pasos"], "movimientos; el espacio completo tiene 181,440 estados alcanzables")

    # ----- Escalamiento -----
    filas = comparar_tamanos()
    print("\n=== Escalamiento (promedio de 5 laberintos por tamaño) ===")
    print(f"{'celdas':>7}{'BFS pasos':>11}{'DFS pasos':>11}{'BFS exp.':>10}{'DFS exp.':>10}{'BFS front.':>12}{'DFS front.':>12}")
    for f in filas:
        print(f"{f['celdas']:>7}{f['bfs_pasos']:>11.0f}{f['dfs_pasos']:>11.0f}{f['bfs_expandidos']:>10.0f}{f['dfs_expandidos']:>10.0f}"
              f"{f['bfs_max_frontera']:>12.0f}{f['dfs_max_frontera']:>12.0f}")

    graficas(g, rb, rd, filas)
    quitar = lambda r: {k: v for k, v in r.items() if k not in ("orden", "camino")}
    with open(os.path.join(AQUI, "resultados_s4.json"), "w", encoding="utf-8") as f:
        json.dump({"laberinto": {"libres": libres, "matriz": g.shape[0], "bfs": quitar(rb), "dfs": quitar(rd)},
                   "puzzle": {"inicio": list(inicio8), "bfs": quitar(r8b), "dfs": quitar(r8d)},
                   "escala": filas}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
