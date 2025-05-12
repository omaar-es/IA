import numpy as np
def n_reinas(n):
  def resolvedor(tablero, fila):
    if fila == n: #Ya se recorrieron todas las posibles filas del tablero (caso base)
      imprimir_tablero(tablero,n)
      return True
    for col in range(n):
      validez=True  #es true cuando la posicion de la reina es la correcta
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
    return False

  def imprimir_tablero(tablero, n):
    tablerito = np.zeros((n,n))
    for i in range(n):
      for j in range(n):
        if tablero[i] == j:
          tablerito[i, j] = 1
    print(tablerito)


  tablero=[-1]*n ##arreglo unidimensional
  if not resolvedor(tablero, 0):
    print("No hay solucion")


n_reinas(4)