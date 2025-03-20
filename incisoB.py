##PRACTICA 3:
import tracemalloc
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

laberinto = np.array([
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
])

nodo_raiz = (0, 3)
meta = (19, 19)
movimientos = [(-1, 0), (0, 1), (0, -1), (1, 0)]

def dfs(laberinto, nodo_raiz, meta):
    tracemalloc.start()
    start_time = time.time()
    
    pila = [(nodo_raiz, [])]
    filas, columnas = laberinto.shape
    nodos_visitados = np.zeros((filas, columnas))
    considerados = []
    
    while pila:
        nodo_actual, camino = pila.pop()
        considerados.append(nodo_actual)
        
        if nodo_actual == meta:
            memory_used = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return camino + [nodo_actual], considerados, time.time() - start_time, memory_used
        
        nodos_visitados[nodo_actual] = 1
        for direccion in movimientos:
            new_pos = (nodo_actual[0] + direccion[0], nodo_actual[1] + direccion[1])
            if 0 <= new_pos[0] < filas and 0 <= new_pos[1] < columnas:
                if nodos_visitados[new_pos] == 0 and laberinto[new_pos] == 1:
                    pila.append((new_pos, camino + [nodo_actual]))
    
    tracemalloc.stop()
    return None, considerados, time.time() - start_time, tracemalloc.get_traced_memory()[1]

def bfs(laberinto, nodo_raiz, meta):
    tracemalloc.start()
    start_time = time.time()
    
    cola = deque([(nodo_raiz, [])])
    filas, columnas = laberinto.shape
    nodos_visitados = np.zeros((filas, columnas))
    considerados = []
    
    while cola:
        nodo_actual, camino = cola.popleft()
        considerados.append(nodo_actual)
        
        if nodo_actual == meta:
            memory_used = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return camino + [nodo_actual], considerados, time.time() - start_time, memory_used
        
        nodos_visitados[nodo_actual] = 1
        for direccion in movimientos:
            new_pos = (nodo_actual[0] + direccion[0], nodo_actual[1] + direccion[1])
            if 0 <= new_pos[0] < filas and 0 <= new_pos[1] < columnas:
                if nodos_visitados[new_pos] == 0 and laberinto[new_pos] == 1:
                    cola.append((new_pos, camino + [nodo_actual]))
    
    tracemalloc.stop()
    return None, considerados, time.time() - start_time, tracemalloc.get_traced_memory()[1]

def graficar_laberinto(laberinto, considerados, camino, title):
    plt.imshow(laberinto, cmap='gray')
    if considerados:
        for i in considerados:
            plt.plot(i[1], i[0], 'o', color='blue', markersize=2)
    if camino:
        for j in camino:
            plt.plot(j[1], j[0], 'o', color='red', markersize=2)
    plt.title(title)
    plt.show()

# Ejecutar y medir DFS
camino_dfs, considerados_dfs, tiempo_dfs, memoria_dfs = dfs(laberinto, nodo_raiz, meta)
graficar_laberinto(laberinto, considerados_dfs, camino_dfs, 'DFS')

# Ejecutar y medir BFS
camino_bfs, considerados_bfs, tiempo_bfs, memoria_bfs = bfs(laberinto, nodo_raiz, meta)
graficar_laberinto(laberinto, considerados_bfs, camino_bfs, 'BFS')

# Mostrar resultados
print(f"DFS: Tiempo = {tiempo_dfs:.6f} s, Memoria = {memoria_dfs} bytes")
print(f"BFS: Tiempo = {tiempo_bfs:.6f} s, Memoria = {memoria_bfs} bytes")