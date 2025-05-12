import numpy as np
def n_reinas(n):
    tablero=[-1]*n ##arreglo unidimensional
    fila=0;
    while fila < n:
        for col in range(n):
            i=0
            ##checa 
            while i < fila:
                ##recorre las posibles soluciones 0...fila_actual
                if tablero[i] == col or tablero[i] - i == col - fila or tablero[i] + i == col + fila:
                validez=False
                break
                i+=1
            if validez:
                tablero[fila] = col
                ##imprimir_tablero(tablero, n)
                if resolvedor(tablero, fila+1):
                return True
                print(f"Backtracking: Retirando la reina de la fila {fila}, columna {col}")
                tablero[fila] = -1
        fila+=1
    if fila == n: #Ya se recorrieron todas las posibles filas del tablero (caso base)
        imprimir_tablero(tablero,n)
    else: 
        print("No hay solucion")

def imprimir_tablero(tablero, n):
    tablerito = np.zeros((n,n))
    for i in range(n):
      for j in range(n):
        if tablero[i] == j:
          tablerito[i, j] = 1
    print(tablerito)

n_reinas(4)