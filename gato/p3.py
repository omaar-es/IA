import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
import time
from matplotlib.animation import FuncAnimation

# Laberinto 16x16 con 1 = pared, 0 = camino
laberinto = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

punto_inicial = (1, 1)
meta = (13, 1)

# Movimientos en 4 direcciones
movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Movimientos en 8 direcciones (incluyendo diagonales)
movimientos_a_estrella = [(-1,1), (-1,-1), (-1,0), (1,-1),(0,1), (1,1), (1,0), (0,-1)]


def BFS(laberinto, punto_inicial, meta):
    #inicializar la cola
    cola = [(punto_inicial, [])]
    filas= np.shape(laberinto)[0]
    columnas = np.shape(laberinto)[1]

    ##lista de nodos visitados
    nodos_visitados = np.zeros((filas, columnas))

    #lista para guardad trayectorias
    considerados = []

    while cola:
        ##Tomamos la última posicion de la pila
        nodo_actual, camino = cola[0]
        ##Imprimimos el nodo actual
        #print("Nodo actual: ", nodo_actual)

        ##Popear
        cola=cola[1:]

        ##guardar
        considerados += [nodo_actual]

        #evaluar si el nodo_actual es la solucion(meta)

        if nodo_actual == meta:
          return camino + [nodo_actual], considerados

        nodos_visitados [nodo_actual[0], nodo_actual[1]]=1
        for direccion in movimientos:
          #evaluar si el nodo vecino es ub¿n obstaculo, está dentro de los limites o si ya lo visitamos previamente
          new_pos = (nodo_actual[0] + direccion[0], nodo_actual[1] + direccion[1])
          if new_pos[0] >= 0 and new_pos[0] < filas and new_pos[1] >= 0 and new_pos[1] < columnas:
            if laberinto[new_pos[0], new_pos[1]] == 0 and nodos_visitados[new_pos[0], new_pos[1]] == 0:
              cola += [(new_pos, camino + [nodo_actual])]
    return None,considerados


def DFS(laberinto, punto_inicial, meta):
    # Inicializar la pila
    pila = [(punto_inicial, [])]
    filas, columnas = laberinto.shape  # Obtener dimensiones correctamente

    # Lista de nodos visitados
    nodos_visitados = np.zeros((filas, columnas), dtype=int)

    # Lista para guardar trayectorias consideradas
    considerados = []

    while pila:
        # Extraer el último elemento de la pila (LIFO)
        nodo_actual, camino = pila[-1]
        pila = pila[:-1]

        # Imprimir el nodo actual
       # print("Nodo actual:", nodo_actual)

        # Guardar nodos considerados
        considerados += [nodo_actual]

        # Evaluar si el nodo_actual es la solución (meta)
        if nodo_actual == meta:
            return camino + [nodo_actual], considerados

        nodos_visitados[nodo_actual[0], nodo_actual[1]] = 1

        for direccion in movimientos:
            # Calcular nueva posición
            new_pos = (nodo_actual[0] + direccion[0], nodo_actual[1] + direccion[1])

            # Evaluar si está dentro de los límites y si es un camino válido
            if 0 <= new_pos[0] < filas and 0 <= new_pos[1] < columnas:
                if laberinto[new_pos[0], new_pos[1]] == 0 and nodos_visitados[new_pos[0], new_pos[1]] == 0:
                  pila += [(new_pos, camino + [nodo_actual])]

    return None, considerados

def heuristica(nodo_actual, objetivo):
    return (abs(objetivo[0]-nodo_actual[0]) + abs(objetivo[1]-nodo_actual[1]))


def a_estrella(laberinto, punto_inicial, meta):
    lista_abierta = [(punto_inicial, 0, heuristica(punto_inicial, meta), [])]
    filas, columnas = laberinto.shape
    lista_cerrada = np.zeros((filas, columnas))
    considerados = []

    while lista_abierta:
        menor_f = lista_abierta[0][2]
        nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[0]
        indice_menor_f = 0

        for i in range(1, len(lista_abierta)):
            if lista_abierta[i][2] < menor_f:
                menor_f = lista_abierta[i][2]
                nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[i]
                indice_menor_f = i

        # <-- Este bloque estaba mal indentado:
        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f+1:]
        considerados += [nodo_actual]

        if nodo_actual == meta:
            return camino_actual + [nodo_actual], considerados

        lista_cerrada[nodo_actual[0], nodo_actual[1]] = 1

        for direccion in movimientos_a_estrella:
            nueva_posicion = (nodo_actual[0] + direccion[0], nodo_actual[1] + direccion[1])
            if 0 <= nueva_posicion[0] < filas and 0 <= nueva_posicion[1] < columnas:
                if laberinto[nueva_posicion[0], nueva_posicion[1]] == 0 and lista_cerrada[nueva_posicion[0], nueva_posicion[1]] == 0:
                    g_nuevo = g_actual + (14 if abs(direccion[0]) == abs(direccion[1]) else 10)
                    f_nuevo = g_nuevo + heuristica(nueva_posicion, meta)

                    ya_esta = False
                    for nodo, g, f, camino in lista_abierta:
                        if nodo == nueva_posicion and g <= g_nuevo:
                            ya_esta = True
                            break

                    if not ya_esta:
                        lista_abierta += [(nueva_posicion, g_nuevo, f_nuevo, camino_actual + [nueva_posicion])]

    return None, considerados


def animar_algoritmo(maze, camino, considerados, titulo, ax, velocidad=100):
    ax.clear()
    ax.imshow(maze, cmap='binary')
    ax.set_title(titulo)
    ax.axis('off')

    puntos_considerados = []
    puntos_camino = []

    def actualizar(frame):
        if frame < len(considerados):
            # Dibujar solo el nodo considerado actual
            i = considerados[frame]
            ax.plot(i[1], i[0], 'o', color='blue', markersize=2)
        elif frame - len(considerados) < len(camino):
            # Dibujar solo el nodo del camino actual
            i = camino[frame - len(considerados)]
            ax.plot(i[1], i[0], 'o', color='red', markersize=4)

    total_frames = len(considerados) + len(camino)
    return FuncAnimation(fig, actualizar, frames=total_frames, interval=velocidad, repeat=False)


def dijkstra(laberinto, punto_inicial, meta):
    # Cada entrada es: (posición, distancia acumulada, camino hasta aquí)
    pendientes = [(punto_inicial, 0, [])]
    
    filas, columnas = laberinto.shape
    visitados = np.zeros((filas, columnas))
    considerados = []

    while pendientes:
        # Buscar nodo con menor distancia acumulada (g)
        indice_menor = 0
        for i in range(1, len(pendientes)):
            if pendientes[i][1] < pendientes[indice_menor][1]:
                indice_menor = i
        
        nodo_actual, dist_actual, camino_actual = pendientes[indice_menor]
        pendientes = pendientes[:indice_menor] + pendientes[indice_menor+1:]
        considerados += [nodo_actual]

        if nodo_actual == meta:
            return camino_actual + [nodo_actual], considerados

        visitados[nodo_actual[0], nodo_actual[1]] = 1

        for dx, dy in movimientos:
            nueva_pos = (nodo_actual[0] + dx, nodo_actual[1] + dy)

            if 0 <= nueva_pos[0] < filas and 0 <= nueva_pos[1] < columnas:
                if laberinto[nueva_pos[0], nueva_pos[1]] == 0 and visitados[nueva_pos[0], nueva_pos[1]] == 0:
                    
                    # Verificamos si ya existe ese nodo en pendientes con menor o igual costo
                    mejor_dist_existente = False
                    for nodo, dist, _ in pendientes:
                        if nodo == nueva_pos and dist <= dist_actual + 1:
                            mejor_dist_existente = True
                            break
                    
                    if not mejor_dist_existente:
                        nuevo_camino = camino_actual + [nodo_actual]
                        pendientes += [(nueva_pos, dist_actual + 1, nuevo_camino)]

    return None, considerados


# Ejecutar los algoritmos y recolectar resultados
resultados = []


tracemalloc.start()
start_time = time.time()
camino, considerados = DFS(laberinto, punto_inicial, meta)
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
titulo = f"DFS\nTiempo: {end_time - start_time:.8f}s\nMemoria: {peak/10**6:.8f}MB"
resultados += [(camino, considerados, titulo)]

tracemalloc.start()
start_time = time.time()
camino, considerados = BFS(laberinto, punto_inicial, meta)
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
titulo = f"BFS\nTiempo: {end_time - start_time:.8f}s\nMemoria: {peak/10**6:.8f}MB"
resultados += [(camino, considerados, titulo)]

tracemalloc.start()
start_time = time.time()
camino, considerados = dijkstra(laberinto, punto_inicial, meta)
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
titulo = f"Dijkstra\nTiempo: {end_time - start_time:.4f}s\nMemoria: {peak/10**6:.4f}MB"
resultados += [(camino, considerados, titulo)]

tracemalloc.start()
start_time = time.time()
camino, considerados = a_estrella(laberinto, punto_inicial, meta)
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
titulo = f"A*\nTiempo: {end_time - start_time:.4f}s\nMemoria: {peak/10**6:.4f}MB"
resultados += [(camino, considerados, titulo)]


# Mostrar animaciones para los algoritmos
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
animaciones = []

# Cambia el valor de `velocidad` para ajustar la rapidez de la animación
velocidad_animacion = 200  # Más alto = más lento, más bajo = más rápido

for ax, (camino, considerados, titulo) in zip(axs.flatten(), resultados):
    anim = animar_algoritmo(laberinto, camino, considerados, titulo, ax, velocidad=velocidad_animacion)
    animaciones.append(anim)

plt.tight_layout()
plt.show()
