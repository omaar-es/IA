def A_estrella(laberinto, punto_inicial, meta):
  lista_abierta = [(punto_inicial, 0, heuristica(punto_inicial, meta))]
  ## nodo, g, f, camino

  filas = np.shape(laberinto)[0]
  columnas = np.shape(laberinto)[1]

  lista_cerrada = np.zero((fila,columnas))

  considerados = []

  while len(lista_abierta) > 0:
    ##En la primera iteración, se inicializa
    ##menor_f como el resultado directo guardado en lista_abierta

    menor_f = lista_abierta [0][2]
    nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[0]
    indice_menor_f = 0

    for i in range (1,len(lista_abierta)):
      ##Si el valor de f en el nodo i-esimo
      ## es menor a el valor de f del nodo actual
      if lista_abierta[i][2] < f_actual:
        ##Guardamos el valor mas pequeño de f y despues
        ##actualizamos todos los valores con el valor-i
        menor_f = lista_abierta[i][2]
        nodo_actual, g_actual, f_actual, camino_actual = lista_abierta[i]
        inidice_menor_f = i

        ##Eliminacion del nodo de menor energia en la
        ##lista abierta

    lista_abierta = lista_abierta[: indice_menor_f] + lista_abierta[indice_menor_f+1:]
    considerados += [nodo_actual]

    if nodo_actual == meta:
      return camino + [nodo_actual], considerados

    lista_cerrada[nodo_actual[0], nodo_actual[1]] = 1

    ##Evaluar a los nodos vecinos
    for direccion in movimientos:
      nueva_posicion = (nodo_actual[0]+direccion[0], nodo_actual[1]+direccion[0])
      ## Ver que el vecino (nueva posicion)este dentro del laberinto
      if ((0 <= nueva_posicion[0] < filas)):
        if ((laberinto [nueva_posicion[0]])):
          if(abs(direccion[0]) == abs (direccion[1])):
            g_nuevo = g_actual+14
          else:
            g_nuevo = g_actual+10
          f_nuevo = g_nuevo + heuristica(nueva_posicion, meta)

          ##Comprobar si cambia de padre o no
          bandera_padre = False
          for nodo,g,f,camino in lista_abierta:
            if g < g_nuevo:
              bandera_padre = True
              break
          if bandera_padre == False:
            lista_abierta += [(nueva_posicion, g_nuevo, f_nuevo, camino_actual + [nueva_posicion])]


  return  None, considerados


##Ejecutar
camino, considerados = A_estrella(laberinto, punto_inicial,meta)
desplegar_laberinto(camino, meta, )