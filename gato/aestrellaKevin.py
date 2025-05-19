import numpy as np
import matplotlib.pyplot as plt

def heuristica(nodo, meta):
    """
    Calcula la heurística (distancia Manhattan multiplicada por 10).
    """
    dx = abs(nodo[0] - meta[0])
    dy = abs(nodo[1] - meta[1])
    return 10 * (dx + dy)

# Movimientos en 4 direcciones (arriba, derecha, izquierda, abajo)
movimientos = [(-1, 0), (0, 1), (0, -1), (1, 0), (-1,-1), (-1, 1), (1, -1), (1, 1)]

def astar_animado(laberinto, inicio, meta, ax):
    """
    Algoritmo A* animado. Usa operaciones de concatenación y slicing
    en lugar de métodos como append, pop o add.
    
    Parámetros:
      laberinto: np.array donde 1 indica camino libre y 0 obstáculo.
      inicio: tupla (fila, columna) nodo inicial.
      meta: tupla (fila, columna) objetivo.
      ax: objeto Axes para la animación.
      
    Retorna:
      (camino, considerados) siendo:
        camino: lista de nodos (tuplas) del camino encontrado.
        considerados: lista de nodos evaluados en el proceso.
    """
    # Cada elemento de lista_abierta es (nodo, g, f, camino_recorrido)
    lista_abierta = [(inicio, 0, heuristica(inicio, meta), [inicio])]
    filas, columnas = laberinto.shape
    # Matriz para marcar nodos visitados (similares a lista cerrada)
    visitados = np.zeros((filas, columnas))
    considerados = []

    while len(lista_abierta) > 0:
        # Buscar el nodo con menor f en lista_abierta sin usar pop
        nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[0]
        indice_menor_f = 0
        for i in range(1, len(lista_abierta)):
            if lista_abierta[i][2] < f_actual:
                nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[i]
                indice_menor_f = i

        # Eliminar el nodo seleccionado usando slicing
        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f+1:]
        # Agregar el nodo actual a la lista de considerados (sin append)
        considerados = considerados + [nodo_actual]
        if nodo_actual != inicio:
            ax.plot(nodo_actual[1], nodo_actual[0], 'o', color='blue')
            plt.pause(0.01)

        # Si se alcanzó la meta, retornar camino y considerados
        if nodo_actual == meta:
            return camino_actual + [nodo_actual], considerados

        # Marcar nodo actual como visitado
        visitados[nodo_actual[0], nodo_actual[1]] = 1

        # Evaluar vecinos en las 4 direcciones
        for m in movimientos:
            nuevo = (nodo_actual[0] + m[0], nodo_actual[1] + m[1])
            if 0 <= nuevo[0] < filas and 0 <= nuevo[1] < columnas:
                # Solo se procesan los nodos no visitados y con valor 1 (camino libre)
                if visitados[nuevo[0], nuevo[1]] == 0 and laberinto[nuevo[0], nuevo[1]] == 1:
                    visitados[nuevo[0], nuevo[1]] = 1  # Marcar al agregar
                    g_nuevo = g_actual + 10  # Costo uniforme para movimiento recto
                    f_nuevo = g_nuevo + heuristica(nuevo, meta)
                    
                    # Verificar si ya existe un camino mejor para el mismo nodo en lista_abierta
                    existe_mejor_camino = False
                    for (nodo, g_val, f_val, cam) in lista_abierta:
                        if nodo == nuevo and g_val <= g_nuevo:
                            existe_mejor_camino = True
                            break
                    if not existe_mejor_camino:
                        # Se construye el nuevo camino sin usar append
                        nuevo_camino = camino_actual + [nodo_actual]
                        lista_abierta = lista_abierta + [(nuevo, g_nuevo, f_nuevo, nuevo_camino)]
    return None, considerados

def animar_solucion_astar(laberinto, inicio, meta):
    """
    Configura la figura y eje para la animación.
    Se dibuja el laberinto (con imshow en escala de grises), se ejecuta el A* animado
    y se dibuja el camino final en rojo.
    """
    fig, ax = plt.subplots()
    ax.imshow(laberinto, cmap='gray')
    
    camino, considerados = astar_animado(laberinto, inicio, meta, ax)
    
    # Animar el camino final en rojo
    if camino is not None:
        for nodo in camino:
            ax.plot(nodo[1], nodo[0], 'o', color='red')
            plt.pause(0.05)
    plt.show()

# --- BLOQUE PRINCIPAL ---
np.random.seed(42)
laberinto = np.random.choice([0, 1], size=(20, 20), p=[0.3, 0.7])
# Asegurar que el inicio y la meta sean caminos libres
laberinto[0, 0] = 1
laberinto[19, 19] = 1

inicio = (0, 0)
meta = (19, 19)

animar_solucion_astar(laberinto, inicio, meta)
