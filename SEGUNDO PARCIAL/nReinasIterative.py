import numpy as np

def n_reinas_iterativo(n):
    tablero = [-1] * n
    fila = 0

    while fila >= 0:
        tablero[fila] += 1 
        #prueba poner una reina en la fila 
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
                fila=-1
            else:
                fila += 1
                tablero[fila] = -1  # Preparar la siguiente fila
        else:
            # Retroceso (backtracking)
            print("haciendo retroceso, moviendo la fila hacia atras", fila)
            #imprimir_tablero(tablero, n)
            tablero[fila] = -1 #reinicia la fila
            fila -= 1 #retrocede una columna
    
            

def imprimir_tablero(tablero, n):
    tablerito = np.zeros((n, n))
    for i in range(n):
        if tablero[i] != -1:
            tablerito[i, tablero[i]] = 1
    print(tablerito)
    print()  # Línea en blanco entre soluciones

# Ejecutar el algoritmo iterativo
print("Ingresa el numero de reinas: ")
num=int(input())

n_reinas_iterativo(num)

