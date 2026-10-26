# S5 · A* vs Dijkstra en una red de transporte

Actividad Parcial 2 · Fundamentos de Inteligencia Artificial · Grupo 171 · Universidad Politécnica de Atlautla

**Equipo:** Bernal Najera Alexandra · Marin Flor Julissa · Rosas Toriz Erandy Naomy · Suarez Melonie Magdalena

Compara los algoritmos A* y Dijkstra para encontrar la ruta de menor costo en una red carretera regional
(21 ciudades del centro de México, de Atlautla a Querétaro) y en redes aleatorias de 100 a 2,500 nodos.
Incluye un dashboard con métricas, una tabla comparativa y el análisis de admisibilidad de la heurística.

## Contenido

| Archivo | Descripción |
|---|---|
| `s5_astar_dijkstra.py` | Código completo: red, algoritmos, experimentos y dashboards |
| `figuras/dashboard_red_regional.png` | Dashboard de la red regional |
| `figuras/dashboard_escalamiento.png` | Dashboard de escalamiento y heurística no admisible |
| `resultados_s5.json` | Resultados numéricos de la última ejecución |
| `FIA_G171_Equipo_S5.pdf` | Informe de la actividad |

## Cómo ejecutarlo

1. Instalar Python 3.10 o superior.
2. Instalar las librerías: `pip install networkx numpy matplotlib`
3. Ejecutar: `python s5_astar_dijkstra.py`
4. Revisar la salida en la terminal, las imágenes en `figuras/` y los datos en `resultados_s5.json`.

## Qué hace el programa

- Construye la red regional con las coordenadas de las ciudades; el costo de cada carretera es la distancia en línea recta × 1.25.
- Ejecuta Dijkstra y A* (heurística: distancia en línea recta) y registra costo, nodos expandidos y tiempo.
- Verifica el costo con NetworkX y comprueba que la heurística sea admisible y consistente.
- Mide el escalamiento en redes aleatorias (30 consultas por tamaño) y el efecto de multiplicar la heurística por 1.5, 2 y 3.

## Nota sobre los datos

Las coordenadas y las conexiones son aproximadas y los costos son estimados; no son mediciones reales por carretera.
