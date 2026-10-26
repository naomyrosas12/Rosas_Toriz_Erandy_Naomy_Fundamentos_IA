"""S5 · A* y Dijkstra en una red de transporte (proyecto en equipo + dashboard)

Compara A* y Dijkstra sobre:
  1) Una red carretera regional (21 ciudades del centro de México, coordenadas aproximadas).
  2) Redes aleatorias de 100 a 2,500 nodos para ver cómo escala cada algoritmo.
Además comprueba si la heurística (distancia en línea recta) es admisible y consistente,
y muestra qué pasa cuando se usa una heurística NO admisible.

Equipo: Bernal Najera Alexandra · Marin Flor Julissa · Rosas Toriz Erandy Naomy · Suarez Melonie Magdalena
Ejecutar: python s5_astar_dijkstra.py
Requiere: pip install networkx numpy matplotlib
Genera:   figuras/dashboard_red_regional.png y figuras/dashboard_escalamiento.png
"""
import heapq
import json
import math
import os
import random
import time

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")  # genera PNG sin abrir ventana
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
os.makedirs(FIG, exist_ok=True)

# ======================================================================
# 1. Datos: red carretera regional (coordenadas y conexiones aproximadas)
# ======================================================================
CIUDADES = {  # nombre: (latitud, longitud)
    "CDMX": (19.43, -99.13), "Texcoco": (19.51, -98.88), "Chalco": (19.26, -98.90),
    "Amecameca": (19.13, -98.77), "Atlautla": (19.02, -98.77), "Cuautla": (18.81, -98.95),
    "Cuernavaca": (18.92, -99.23), "Puebla": (19.04, -98.21), "Tlaxcala": (19.31, -98.24),
    "Apizaco": (19.41, -98.14), "Pachuca": (20.10, -98.76), "Tulancingo": (20.08, -98.37),
    "Toluca": (19.28, -99.66), "Querétaro": (20.59, -100.39), "San Juan del Río": (20.39, -99.99),
    "Tula": (20.05, -99.34), "Tepeji": (19.90, -99.35), "Cholula": (19.06, -98.30),
    "Atlixco": (18.91, -98.43), "Izúcar": (18.60, -98.47), "Tehuacán": (18.46, -97.39),
}
CARRETERAS = [
    ("CDMX", "Texcoco"), ("CDMX", "Chalco"), ("CDMX", "Toluca"), ("CDMX", "Cuernavaca"),
    ("CDMX", "Pachuca"), ("CDMX", "Tepeji"), ("Tepeji", "Tula"), ("Tula", "San Juan del Río"),
    ("San Juan del Río", "Querétaro"), ("Tula", "Pachuca"), ("Texcoco", "Pachuca"),
    ("Texcoco", "Tlaxcala"), ("Texcoco", "Chalco"), ("Chalco", "Amecameca"), ("Amecameca", "Atlautla"),
    ("Atlautla", "Cuautla"), ("Chalco", "Puebla"), ("Amecameca", "Puebla"), ("Cuautla", "Cuernavaca"),
    ("Cuautla", "Izúcar"), ("Izúcar", "Atlixco"), ("Atlixco", "Puebla"), ("Atlixco", "Cholula"),
    ("Cholula", "Puebla"), ("Puebla", "Tlaxcala"), ("Tlaxcala", "Apizaco"), ("Apizaco", "Puebla"),
    ("Apizaco", "Tulancingo"), ("Pachuca", "Tulancingo"), ("Puebla", "Tehuacán"), ("Izúcar", "Tehuacán"),
    ("Toluca", "Cuernavaca"), ("Toluca", "Tepeji"),
]
SINUOSIDAD = 1.25   # la carretera mide ~25 % más que la línea recta (nunca menos)


def haversine(a, b):
    """Distancia en km sobre la esfera entre dos puntos (lat, lon)."""
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def euclidiana(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def red_regional():
    G = nx.Graph(dist=haversine)
    for n, p in CIUDADES.items():
        G.add_node(n, pos=p)
    for u, v in CARRETERAS:
        G.add_edge(u, v, weight=round(haversine(CIUDADES[u], CIUDADES[v]) * SINUOSIDAD, 1))
    return G


def red_aleatoria(n, k=5, semilla=0):
    """Red de n nodos en un cuadrado de 100x100 km; cada nodo se une a sus k vecinos más cercanos.
    El peso de cada arista es la distancia euclidiana por un factor de sinuosidad >= 1."""
    rng = np.random.default_rng(semilla)
    pts = rng.uniform(0, 100, size=(n, 2))
    G = nx.Graph(dist=euclidiana)
    for i, p in enumerate(pts):
        G.add_node(i, pos=(p[0], p[1]))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    for i in range(n):
        for j in np.argsort(d[i])[1:k + 1]:
            G.add_edge(i, int(j), weight=float(d[i, j] * rng.uniform(1.0, 1.5)))
    mayor = max(nx.connected_components(G), key=len)
    return G.subgraph(mayor).copy()


# ======================================================================
# 2. Algoritmos: Dijkstra y A* (misma estructura, solo cambia la prioridad)
# ======================================================================
def buscar(G, origen, destino, h=None):
    """Búsqueda de costo uniforme (Dijkstra) si h es None; A* si se da una heurística h(nodo).
    Prioridad = g + h. Devuelve camino, costo, nodos expandidos, tiempo y orden de expansión."""
    t0 = time.perf_counter()
    g = {origen: 0.0}
    padre = {}
    abiertos = [((h(origen) if h else 0.0), 0.0, origen)]
    cerrados = set()
    orden = []
    while abiertos:
        _, gu, u = heapq.heappop(abiertos)
        if u in cerrados:
            continue                      # entrada vieja de la cola
        cerrados.add(u)
        orden.append(u)
        if u == destino:
            break
        for v, datos in G[u].items():
            nuevo = gu + datos["weight"]
            if nuevo < g.get(v, math.inf):
                g[v] = nuevo
                padre[v] = u
                heapq.heappush(abiertos, (nuevo + (h(v) if h else 0.0), nuevo, v))
    camino = [destino]
    while camino[-1] != origen:
        camino.append(padre[camino[-1]])
    return {"camino": camino[::-1], "costo": g[destino], "expandidos": len(orden),
            "tiempo_ms": (time.perf_counter() - t0) * 1000, "orden": orden}


def dijkstra(G, o, d):
    return buscar(G, o, d)


def astar(G, o, d, escala=1.0):
    """A* con h(n) = escala x distancia en línea recta al destino. escala = 1 es admisible."""
    pos, f = nx.get_node_attributes(G, "pos"), G.graph["dist"]
    return buscar(G, o, d, lambda n: escala * f(pos[n], pos[d]))


# ======================================================================
# 3. Admisibilidad y consistencia de la heurística
# ======================================================================
def verificar_heuristica(G, destino, escala=1.0):
    """Compara h(n) con el costo real h*(n) (Dijkstra desde el destino) y revisa la consistencia."""
    pos, f = nx.get_node_attributes(G, "pos"), G.graph["dist"]
    h = {n: escala * f(pos[n], pos[destino]) for n in G}
    real = nx.single_source_dijkstra_path_length(G, destino)
    viol_adm = [n for n in G if h[n] > real[n] + 1e-9]
    viol_con = [(u, v) for u, v, d in G.edges(data=True)
                if h[u] > d["weight"] + h[v] + 1e-9 or h[v] > d["weight"] + h[u] + 1e-9]
    ratio = max(h[n] / real[n] for n in G if real[n] > 0)
    return {"h": h, "real": real, "viol_adm": len(viol_adm), "viol_con": len(viol_con),
            "max_ratio": ratio, "nodos": G.number_of_nodes(), "aristas": G.number_of_edges()}


# ======================================================================
# 4. Experimentos
# ======================================================================
def caso_regional(origen="Atlautla", destino="Querétaro"):
    G = red_regional()
    return G, dijkstra(G, origen, destino), astar(G, origen, destino), verificar_heuristica(G, destino)


def escalamiento(tamanos=(100, 300, 600, 1000, 1500, 2500), pares=30):
    """Promedia nodos expandidos y tiempo sobre 'pares' consultas aleatorias por tamaño."""
    filas = []
    for n in tamanos:
        G = red_aleatoria(n, semilla=n)
        nodos = list(G.nodes)
        rnd = random.Random(n)
        d_exp, a_exp, d_t, a_t = [], [], [], []
        for _ in range(pares):
            o, d = rnd.sample(nodos, 2)
            rd, ra = dijkstra(G, o, d), astar(G, o, d)
            assert abs(rd["costo"] - ra["costo"]) < 1e-6      # A* admisible = mismo costo óptimo
            d_exp.append(rd["expandidos"]); a_exp.append(ra["expandidos"])
            d_t.append(rd["tiempo_ms"]); a_t.append(ra["tiempo_ms"])
        filas.append({"n": G.number_of_nodes(), "dij_exp": np.mean(d_exp), "ast_exp": np.mean(a_exp),
                      "dij_ms": np.mean(d_t), "ast_ms": np.mean(a_t),
                      "reduccion": 100 * (1 - np.mean(a_exp) / np.mean(d_exp))})
    return filas


def heuristica_no_admisible(escalas=(1.0, 1.5, 2.0, 3.0), n=600, pares=60):
    """Multiplica h por un factor > 1 (la vuelve no admisible) y mide sobrecosto y nodos expandidos."""
    G = red_aleatoria(n, semilla=7)
    nodos = list(G.nodes)
    filas = []
    for e in escalas:
        rnd = random.Random(11)
        sobre, exp, malas = [], [], 0
        for _ in range(pares):
            o, d = rnd.sample(nodos, 2)
            ra, opt = astar(G, o, d, escala=e), dijkstra(G, o, d)
            extra = 100 * (ra["costo"] / opt["costo"] - 1)
            sobre.append(extra); exp.append(ra["expandidos"]); malas += extra > 1e-6
        filas.append({"escala": e, "sobrecosto": float(np.mean(sobre)), "expandidos": float(np.mean(exp)),
                      "no_optimas": malas, "pares": pares, "admisible": verificar_heuristica(G, nodos[0], e)["viol_adm"] == 0})
    return filas


# ======================================================================
# 5. Dashboard (matplotlib)
# ======================================================================
CORAL, GRAFITO, AZUL, GRIS = "#E4572E", "#1F2933", "#2E86AB", "#9AA5B1"


def dashboard_regional(G, rd, ra, ver, origen, destino):
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 1], hspace=0.38, wspace=0.28, top=0.84, bottom=0.07)
    fig.text(0.04, 0.95, "Dashboard · A* vs Dijkstra en la red carretera regional", fontsize=17, fontweight="bold", color=GRAFITO)
    red = 100 * (1 - ra["expandidos"] / rd["expandidos"])
    fig.text(0.04, 0.905, f"{origen} → {destino}  ·  costo óptimo {ra['costo']:.1f} km (ambos)  ·  Dijkstra expandió "
             f"{rd['expandidos']} nodos  ·  A* expandió {ra['expandidos']}  ·  reducción {red:.0f} %", fontsize=10.5, color="#52606D")
    pos = {n: (p[1], p[0]) for n, p in nx.get_node_attributes(G, "pos").items()}   # x = lon, y = lat
    ax = fig.add_subplot(gs[:, :2])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CBD2D9", width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#F0F4F8", edgecolors=GRIS, node_size=150)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=rd["orden"], node_color="none", edgecolors=AZUL, linewidths=2.2, node_size=330)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ra["orden"], node_color=CORAL, alpha=0.85, node_size=120)
    ruta = list(zip(ra["camino"][:-1], ra["camino"][1:]))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=ruta, edge_color=GRAFITO, width=3.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, verticalalignment="bottom",
                            labels={n: n for n in G}, font_color=GRAFITO)
    ax.set_title("Mapa: nodos expandidos (aro azul = Dijkstra, relleno naranja = A*, línea negra = ruta)",
                 fontsize=9.5, loc="left")
    ax.axis("off")
    ax2 = fig.add_subplot(gs[0, 2])
    b = ax2.bar(["Dijkstra", "A*"], [rd["expandidos"], ra["expandidos"]], color=[AZUL, CORAL], width=0.55)
    ax2.bar_label(b, fontsize=10, fontweight="bold")
    ax2.set_title("Nodos expandidos", fontsize=10, loc="left"); ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax3 = fig.add_subplot(gs[1, 2])
    h, real = [ver["h"][n] for n in G], [ver["real"][n] for n in G]
    ax3.scatter(real, h, color=CORAL, s=28, zorder=3)
    lim = max(real) * 1.05
    ax3.plot([0, lim], [0, lim], "--", color=GRAFITO, lw=1)
    ax3.fill_between([0, lim], [0, lim], lim, color="#FDE8E2", alpha=0.7, zorder=0)
    ax3.text(lim * 0.05, lim * 0.85, "zona NO admisible\n(h > costo real)", fontsize=8, color=CORAL)
    ax3.set_xlim(0, lim); ax3.set_ylim(0, lim)
    ax3.set_xlabel("costo real h*(n) (km)", fontsize=9); ax3.set_ylabel("heurística h(n) (km)", fontsize=9)
    ax3.set_title(f"Admisibilidad: {ver['viol_adm']} violaciones de {ver['nodos']} nodos", fontsize=10, loc="left")
    ax3.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIG, "dashboard_red_regional.png"), dpi=150)
    plt.close(fig)


def dashboard_escalamiento(esc, noadm):
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.6))
    fig.suptitle("Dashboard · Escalamiento de A* y Dijkstra y efecto de una heurística no admisible",
                 fontsize=14, fontweight="bold", color=GRAFITO, x=0.04, ha="left")
    n = [f["n"] for f in esc]
    a = axs[0, 0]; a.plot(n, [f["dij_exp"] for f in esc], "o-", color=AZUL, label="Dijkstra"); a.plot(n, [f["ast_exp"] for f in esc], "s-", color=CORAL, label="A*")
    a.set_title("Nodos expandidos (promedio de 30 consultas)", fontsize=10, loc="left"); a.set_xlabel("nodos de la red"); a.legend()
    a = axs[0, 1]; a.plot(n, [f["dij_ms"] for f in esc], "o-", color=AZUL, label="Dijkstra"); a.plot(n, [f["ast_ms"] for f in esc], "s-", color=CORAL, label="A*")
    a.set_title("Tiempo de ejecución (ms)", fontsize=10, loc="left"); a.set_xlabel("nodos de la red"); a.legend()
    a = axs[1, 0]; b = a.bar([str(x) for x in n], [f["reduccion"] for f in esc], color=CORAL); a.bar_label(b, fmt="%.0f %%", fontsize=9)
    a.set_title("Reducción de nodos expandidos de A* frente a Dijkstra", fontsize=10, loc="left"); a.set_xlabel("nodos de la red"); a.set_ylim(0, 100)
    a = axs[1, 1]; x = [f"×{f['escala']}" for f in noadm]
    b = a.bar(x, [f["sobrecosto"] for f in noadm], color=[AZUL if f["admisible"] else CORAL for f in noadm])
    a.bar_label(b, fmt="%.1f %%", fontsize=9)
    a.set_title("Sobrecosto de la ruta según el factor de la heurística", fontsize=10, loc="left"); a.set_xlabel("h(n) multiplicada por (azul = admisible)")
    for a in axs.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "dashboard_escalamiento.png"), dpi=150)
    plt.close(fig)


def main():
    origen, destino = "Atlautla", "Querétaro"
    G, rd, ra, ver = caso_regional(origen, destino)
    print(f"=== Red regional: {G.number_of_nodes()} ciudades, {G.number_of_edges()} carreteras ===")
    print(f"{'':10}{'Costo (km)':>12}{'Expandidos':>12}{'Tiempo (ms)':>13}")
    for nombre, r in (("Dijkstra", rd), ("A*", ra)):
        print(f"{nombre:10}{r['costo']:>12.1f}{r['expandidos']:>12}{r['tiempo_ms']:>13.3f}")
    print("Ruta:", " -> ".join(ra["camino"]))
    print(f"Heurística haversine: violaciones de admisibilidad = {ver['viol_adm']}, de consistencia = {ver['viol_con']}, "
          f"max h/h* = {ver['max_ratio']:.2f}")
    # verificación independiente con NetworkX
    ref = nx.dijkstra_path_length(G, origen, destino)
    print(f"Verificación con NetworkX: costo = {ref:.1f} km -> coincide: {abs(ref - ra['costo']) < 1e-6}")

    esc = escalamiento()
    print("\n=== Escalamiento (30 consultas por tamaño) ===")
    print(f"{'nodos':>6}{'Dij exp':>9}{'A* exp':>8}{'Dij ms':>9}{'A* ms':>8}{'reducción':>11}")
    for f in esc:
        print(f"{f['n']:>6}{f['dij_exp']:>9.0f}{f['ast_exp']:>8.0f}{f['dij_ms']:>9.2f}{f['ast_ms']:>8.2f}{f['reduccion']:>10.0f}%")

    noadm = heuristica_no_admisible()
    print("\n=== Heurística no admisible (red de ~600 nodos, 60 consultas) ===")
    for f in noadm:
        print(f"h x{f['escala']:<4} admisible={str(f['admisible']):5}  sobrecosto medio = {f['sobrecosto']:.2f} %  "
              f"rutas no óptimas = {f['no_optimas']}/{f['pares']}  nodos expandidos = {f['expandidos']:.0f}")

    dashboard_regional(G, rd, ra, ver, origen, destino)
    dashboard_escalamiento(esc, noadm)

    # guarda los resultados para reutilizarlos en el informe
    limpio = lambda r: {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()}
    with open(os.path.join(AQUI, "resultados_s5.json"), "w", encoding="utf-8") as f:
        json.dump({"dijkstra": {k: v for k, v in rd.items() if k != "orden"} | {"orden": rd["orden"]},
                   "astar": {k: v for k, v in ra.items() if k != "orden"} | {"orden": ra["orden"]},
                   "viol_adm": ver["viol_adm"], "viol_con": ver["viol_con"], "max_ratio": float(ver["max_ratio"]),
                   "escalamiento": [limpio(x) for x in esc], "no_admisible": [limpio(x) for x in noadm]},
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
