# S6 · Algoritmo genético para programar turnos de personal

Actividad Parcial 2 · Fundamentos de Inteligencia Artificial · Grupo 171 · Universidad Politécnica de Atlautla

**Equipo:** Bernal Najera Alexandra · Marin Flor Julissa · Rosas Toriz Erandy Naomy · Suarez Melonie Magdalena

Algoritmo genético implementado con DEAP que asigna turnos (mañana, tarde, noche o descanso) a 8 empleados durante
7 días, cumpliendo la cobertura mínima y reduciendo las violaciones de reglas laborales. Se ejecuta con una población
de 100 individuos durante 30 generaciones.

## Contenido

| Archivo | Descripción |
|---|---|
| `s6_algoritmo_genetico.py` | Código completo: cromosoma, fitness, operadores, experimentos y gráficas |
| `figuras/convergencia.png` | Curvas de convergencia |
| `figuras/horario_optimo.png` | Mejor horario encontrado |
| `figuras/sensibilidad.png` | Efecto de la tasa de mutación |
| `resultados_s6.json` | Resultados numéricos de la última ejecución |
| `FIA_G171_Equipo_S6.pdf` | Informe de la actividad |

## Cómo replicar los resultados paso a paso

1. Instalar Python 3.10 o superior.
2. Instalar las librerías: `pip install deap numpy matplotlib`
3. Descargar esta carpeta y abrir una terminal dentro de ella.
4. Ejecutar: `python s6_algoritmo_genetico.py` (tarda menos de un minuto).
5. Revisar la salida en la terminal, las imágenes en `figuras/` y los datos en `resultados_s6.json`.

Las semillas aleatorias están fijas (0 a 9 y 42), por lo que se obtienen los mismos resultados.

## Diseño del algoritmo

| Componente | Diseño |
|---|---|
| Cromosoma | 56 genes (8 empleados × 7 días); gen = 0 descanso, 1 mañana, 2 tarde, 3 noche |
| Fitness | Penalización total por violaciones de reglas (se minimiza) |
| Selección | Torneo de tamaño 3 |
| Cruce | Por empleado (intercambio de filas completas), probabilidad 0.8 |
| Mutación | Probabilidad 0.3 por individuo; 0.04 por gen |
| Elitismo | Los 2 mejores pasan sin cambios |

## Cómo probar otro caso

Cambiar en el código las listas `EMPLEADOS` y `DIAS`, el diccionario `COBERTURA`, el conjunto `SIN_NOCHE` o los `PESOS` de la función de aptitud.

## Nota sobre los datos

Los empleados, la cobertura y las restricciones son un caso de ejemplo, no datos de una empresa real.
