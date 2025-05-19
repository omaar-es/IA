import numpy as np
import matplotlib.pyplot as plt
import random

# laberinto 16x16 con 1 representando paredes y 0 representando camino
laberinto = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

punto_inicial = (1, 1)
meta = (5, 10)

# Tipos de movimientos
movimientos = [(-1, 1), (-1, -1), (-1, 0), (1, -1), (0, 1), (1, 1), (1, 0), (0, -1)]


def heuristica(nodo_actual, objetivo):
    return (abs(objetivo[0] - nodo_actual[0]) + abs(objetivo[1] - nodo_actual[1]))


def desplegar_laberinto_subplot(ax, maze, camino=None, considerados=None, titulo=""):
    ax.imshow(maze, cmap='binary')
    if considerados:
        for i in considerados:
            ax.plot(i[1], i[0], 'o', color='blue')
    if camino:
        for i in camino:
            ax.plot(i[1], i[0], 'o', color='red')
    ax.set_title(titulo)
    ax.axis('off')


def a_estrella(laberinto, punto_inicial, meta):
    lista_abierta = [(punto_inicial, 0, heuristica(punto_inicial, meta), [])]
    filas, columnas = laberinto.shape
    lista_cerrada = np.zeros((filas, columnas))
    considerados = []
    nodos_considerados = 0

    while lista_abierta:
        menor_f = lista_abierta[0][2]
        nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[0]
        indice_menor_f = 0

        for i in range(1, len(lista_abierta)):
            if lista_abierta[i][2] < menor_f:
                menor_f = lista_abierta[i][2]
                nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[i]
                indice_menor_f = i

        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f + 1:]
        considerados += [nodo_actual]
        nodos_considerados += 1

        if nodo_actual == meta:
            energia_camino = g_actual  # Guardar la energía del camino
            return camino_actual + [nodo_actual], considerados, nodos_considerados, len(camino_actual + [nodo_actual]), energia_camino

        lista_cerrada[nodo_actual[0], nodo_actual[1]] = 1

        for direccion in movimientos:
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
    return None, considerados, nodos_considerados, 0, 0


fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Crear una cuadrícula de 2x2 subplots

for i in range(4):
    random.shuffle(movimientos)
    camino, considerados, nodos_considerados, nodos_camino, energia_camino = a_estrella(laberinto, punto_inicial, meta)
    print(f"Ejecucion numero: {i + 1}")
    print(f"El numero de nodos considerados fue de: {nodos_considerados - nodos_camino}")
    print(f"El numero de nodos en el camino fue de: {nodos_camino}")
    print(f"La energia del camino fue de: {energia_camino}")
    
    # Seleccionar el subplot correspondiente
    ax = axs[i // 2, i % 2]
    titulo = f"Ejecución {i + 1}\nNodos: {nodos_camino}, Energía: {energia_camino}"
    desplegar_laberinto_subplot(ax, laberinto, camino, considerados, titulo)

plt.tight_layout()  # Ajustar el diseño para evitar superposición
plt.show()
