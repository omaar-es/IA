import numpy as np
import matplotlib.pyplot as plt
import math
import time


#laberinto 16x16 con 1 representando paredes y 0 representando camino
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


punto_inicial=(1,1)
meta= (5,10)


##Tipos de movimientos
movimientos = [(-1,1), (-1,-1), (-1,0), (1,-1),(0,1), (1,1), (1,0), (0,-1)]

def heuristica(nodo_actual, objetivo, heuristica_probada):
  if heuristica_probada == 1:
    return (abs(objetivo[0]-nodo_actual[0]) + abs(objetivo[1]-nodo_actual[1]))
  else:
    return math.sqrt((objetivo[0] - nodo_actual[0])**2 + (objetivo[1] - nodo_actual[1])**2)



def desplegar_laberinto(maze, camino = None, considerados = None):
  plt.imshow(maze, cmap='binary')
  if considerados:
    for i in considerados:
      plt.plot(i[1], i[0], 'o',color ='blue')
  if camino:
    for i in camino:
      plt.plot(i[1], i[0], 'o', color='red')
  plt.show()

def a_estrella(laberinto, punto_inicial, meta, heuristica_probada):
    lista_abierta = [(punto_inicial, 0, heuristica(punto_inicial, meta, heuristica_probada), [])]
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
        considerados+=[nodo_actual]

        if nodo_actual == meta:
            return camino_actual + [nodo_actual], considerados

        lista_cerrada[nodo_actual[0], nodo_actual[1]] = 1

        for direccion in movimientos:
            nueva_posicion = (nodo_actual[0] + direccion[0], nodo_actual[1] + direccion[1])
            if 0 <= nueva_posicion[0] < filas and 0 <= nueva_posicion[1] < columnas:
                if laberinto[nueva_posicion[0], nueva_posicion[1]] == 0 and lista_cerrada[nueva_posicion[0], nueva_posicion[1]] == 0:
                    g_nuevo = g_actual + (14 if abs(direccion[0]) == abs(direccion[1]) else 10)
                    f_nuevo = g_nuevo + heuristica(nueva_posicion, meta, heuristica_probada)

                    ya_esta = False
                    for nodo, g, f, camino in lista_abierta:
                        if nodo == nueva_posicion and g <= g_nuevo:
                            ya_esta = True
                            break

                    if not ya_esta:
                        lista_abierta += [(nueva_posicion, g_nuevo, f_nuevo, camino_actual + [nueva_posicion])]
    return None, considerados


start_time = time.time() #Inicio de medicion de tiempo
camino, considerados = a_estrella(laberinto, punto_inicial, meta,1)
end_time = time.time()
manhattan= end_time - start_time

print(f"Tiempo de ejecución con distancia manhattan: {manhattan}")
desplegar_laberinto(laberinto, camino, considerados)


start_time = time.time() #Inicio de medicion de tiempo
camino, considerados = a_estrella(laberinto, punto_inicial, meta,2)
end_time = time.time()
euclideana = end_time - start_time


print(f"Tiempo de ejecución con distancia euclideana: {euclideana}")
desplegar_laberinto(laberinto, camino, considerados)

if manhattan<euclideana:
  print(f" manhattan es mas rapido")
else:
  print(f"Euclideana es mas rapido")
