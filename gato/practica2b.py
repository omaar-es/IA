import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import math
from math import radians, sin, cos, sqrt, atan2

# Configuración OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

# Descargar el grafo de Eixample, Barcelona
G = ox.graph_from_place("Eixample, Barcelona, España", network_type='drive')

# Seleccionar origen y destino usando coordenadas correctas (longitud, latitud)
orig = ox.distance.nearest_nodes(G, X=2.163, Y=41.387)  # Longitud, Latitud
dest = ox.distance.nearest_nodes(G, X=2.173, Y=41.393)

# Mostrar mapa con puntos de origen y destino
fig, ax = ox.plot_graph(G, node_size=0, edge_linewidth=0.5, bgcolor='white', show=False)
ax.scatter(G.nodes[orig]['x'], G.nodes[orig]['y'], c='green', s=100, label='Origen', zorder=5)
ax.scatter(G.nodes[dest]['x'], G.nodes[dest]['y'], c='red', s=100, label='Destino', zorder=5)
ax.legend()
plt.title("Eixample: Origen (verde) y Destino (rojo)")
plt.show()

# Heurística corregida usando Haversine para metros
def heuristica(u, v):
    # Obtener coordenadas de los nodos (longitud, latitud)
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    
    # Convertir grados a radianes
    lat1, lon1 = radians(y1), radians(x1)
    lat2, lon2 = radians(y2), radians(x2)
    
    # Fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radio_tierra = 6371000  # Radio de la Tierra en metros
    return radio_tierra * c

# Algoritmo A* corregido
def a_estrella_grafo(G, inicio, meta):
    lista_abierta = [(0 + heuristica(inicio, meta), inicio, 0, [])]  # (f, nodo, g, camino)
    lista_cerrada = []

    while lista_abierta:
        # Buscar el nodo con menor f en lista_abierta
        menor_f = lista_abierta[0][0]
        indice_menor_f = 0
        for i in range(1, len(lista_abierta)):
            if lista_abierta[i][0] < menor_f:
                menor_f = lista_abierta[i][0]
                indice_menor_f = i
        f_actual, nodo_actual, g_actual, camino_actual = lista_abierta[indice_menor_f]
        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f+1:]

        if nodo_actual == meta:
            return camino_actual + [nodo_actual]

        if nodo_actual in lista_cerrada:
            continue
        lista_cerrada = lista_cerrada + [nodo_actual]  # Agrega nodo_actual a la lista

        for vecino in G.neighbors(nodo_actual):
            if vecino in lista_cerrada:
                continue
            
            # Calcular nuevo costo g
            longitud_arista = G[nodo_actual][vecino][0]['length']
            g_nuevo = g_actual + longitud_arista
            f_nuevo = g_nuevo + heuristica(vecino, meta)
            
            # Verificar si ya existe un mejor camino en lista_abierta
            existe_mejor = False
            for i, (f_existente, n_existente, g_existente, _) in enumerate(lista_abierta):
                if n_existente == vecino and g_existente <= g_nuevo:
                    existe_mejor = True
                    break
            
            if not existe_mejor:
                lista_abierta = lista_abierta + [(f_nuevo, vecino, g_nuevo, camino_actual + [nodo_actual])]

    return None

# Ejecutar A* y mostrar resultados
ruta = a_estrella_grafo(G, orig, dest)
if ruta is not None:
    # Convertir la ruta a una lista de nodos en orden secuencial
    ruta_ordenada = []
    for i in range(len(ruta)-1):
        ruta_ordenada.extend([ruta[i], ruta[i+1]])
    
    # Crear figura combinada
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_linewidth=0.5,
        bgcolor='white',
        show=False
    )
    
    # Dibujar ruta segmento por segmento
    for u, v in zip(ruta[:-1], ruta[1:]):
        if G.has_edge(u, v):
            x = [G.nodes[u]['x'], G.nodes[v]['x']]
            y = [G.nodes[u]['y'], G.nodes[v]['y']]
            ax.plot(x, y, 'r-', linewidth=4, zorder=5)
    
    # Mostrar puntos origen/destino
    ax.scatter(G.nodes[orig]['x'], G.nodes[orig]['y'], c='green', s=200, zorder=6)
    ax.scatter(G.nodes[dest]['x'], G.nodes[dest]['y'], c='blue', s=200, zorder=6)
    
    plt.title("Ruta óptima encontrada por A*")
    plt.show()