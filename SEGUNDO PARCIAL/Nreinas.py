import numpy as np

def n_reinas_iterativo(n):
    tablero = [-1] * n
    fila = 0

    while fila >= 0:
        tablero[fila] += 1  # Probar siguiente columna en la fila actual

        while tablero[fila] < n:
            # Verificar si es una posición válida
            validez = True
            for i in range(fila):
                if (tablero[i] == tablero[fila] or
                    tablero[i] - i == tablero[fila] - fila or
                    tablero[i] + i == tablero[fila] + fila):
                    validez = False
                    break

            if validez:
                break  # Posición válida, salimos del while para continuar con la siguiente fila
            else:
                tablero[fila] += 1  # Probar siguiente columna

        if tablero[fila] < n:
            if fila == n - 1:
                imprimir_tablero(tablero, n)  # Solución encontrada
                # Podemos seguir buscando más soluciones, si se desea
                tablero[fila] = -1
                fila -= 1
            else:
                fila += 1
                tablero[fila] = -1  # Preparar la siguiente fila
        else:
            # Retroceso (backtracking)
            tablero[fila] = -1
            fila -= 1

def imprimir_tablero(tablero, n):
    tablerito = np.zeros((n, n))
    for i in range(n):
        if tablero[i] != -1:
            tablerito[i, tablero[i]] = 1
    print(tablerito)
    print()  # Línea en blanco entre soluciones

# Ejecutar el algoritmo iterativo
n_reinas_iterativo(4)
