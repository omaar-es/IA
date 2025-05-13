def resolver_sudoku(tablero):
    fila, col = encontrar_vacia(tablero)
    if fila is None:
        return True  # Ya está resuelto

    for num in range(1, 10):  # Números del 1 al 9
        if es_valido(tablero, fila, col, num):
            tablero[fila][col] = num

            if resolver_sudoku(tablero):  # Llamada recursiva
                return True

            tablero[fila][col] = 0  # Backtracking

    return False  # Si ningún número es válido, se hace backtrack
def encontrar_vacia(tablero):
    for i in range(9):
        for j in range(9):
            if tablero[i][j] == 0:
                return i, j
    return None, None

def es_valido(tablero, fila, col, num):
    # Verificar fila
    if num in tablero[fila]:
        return False

    # Verificar columna
    for i in range(9):
        if tablero[i][col] == num:
            return False

    # Verificar subcuadro 3x3
    inicio_fila = (fila // 3) * 3
    inicio_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if tablero[inicio_fila + i][inicio_col + j] == num:
                return False

    return True

def imprimir_tablero(tablero):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(tablero[i][j] if tablero[i][j] != 0 else ".", end=" ")
        print()


tablero = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

print("Sudoku original:")
imprimir_tablero(tablero)

if resolver_sudoku(tablero):
    print("\nSudoku resuelto:")
    imprimir_tablero(tablero)
else:
    print("No se pudo resolver.")
