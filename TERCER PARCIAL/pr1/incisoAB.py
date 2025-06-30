import numpy as np

# Funciones de activación
def escalon(z):
  if z >= 0:
    return 1
  else:
    return 0

def relu(z):
  if z >= 0:
    return z
  else:
    return 0

def sigmoide(z):
  return 1 / (1 + np.exp(-z))

def tanh_act(z):
  return np.tanh(z)

def bipolar(z):
  if z > 0:
    return 1
  elif z == 0:
    return 0
  else:
    return -1

# Elegir compuerta
tipo = input("Seleccione el tipo de compuerta (AND / OR): ").strip().upper()
while tipo != "AND" and tipo != "OR":
  tipo = input("Entrada inválida. Elija AND o OR: ").strip().upper()

# Elegir número de entradas
n = int(input("Ingrese el número de características (entre 2 y 5): "))
while n < 2 or n > 5:
  n = int(input("Número inválido. Ingrese entre 2 y 5: "))

# Elegir función de activación
print("\nSeleccione la función de activación:")
print("1. Escalón")
print("2. ReLU")
print("3. Sigmoide")
print("4. Tanh")
print("5. Bipolar")

opcion = int(input("Ingrese una opción (1-5): "))
while opcion not in [1, 2, 3, 4, 5]:
  opcion = int(input("Opción inválida. Ingrese una opción (1-5): "))

if opcion == 1:
  funcion_activacion = escalon
elif opcion == 2:
  funcion_activacion = relu
elif opcion == 3:
  funcion_activacion = sigmoide
elif opcion == 4:
  funcion_activacion = tanh_act
elif opcion == 5:
  funcion_activacion = bipolar

# Parámetros
lr = 0.2
epochs = 100

# Generar combinaciones binarias
filas = 2 ** n
X = np.zeros((filas, n), dtype=int)
for i in range(filas):
  num = i
  for j in range(n - 1, -1, -1):
    X[i][j] = num % 2
    num = num // 2

# Salidas deseadas Yd
Yd = np.zeros(filas, dtype=int)
for i in range(filas):
  if tipo == "AND":
    es_uno = 1
    for bit in X[i]:
      if bit == 0:
        es_uno = 0
        break
    Yd[i] = es_uno
  elif tipo == "OR":
    es_uno = 0
    for bit in X[i]:
      if bit == 1:
        es_uno = 1
        break
    Yd[i] = es_uno

# Inicializar pesos aleatorios
W = np.random.rand(n)
w0 = np.random.rand()
x0 = 1
yobt = np.zeros_like(Yd)

# Entrenamiento
for i in range(epochs):
  for j in range(filas):
    z = w0 * x0 + np.sum(W * X[j])
    y = funcion_activacion(z)
    if opcion in [3, 4]:  # sigmoide y tanh necesitan redondeo para comparar
      y = round(y)
    yobt[j] = y
    if y != Yd[j]:
      error = y - Yd[j]
      w0 = w0 - lr * error
      for k in range(n):
        W[k] = W[k] - lr * error * X[j][k]

# Mostrar tabla
print("\nTabla de verdad aprendida:")
for i in range(filas):
  entrada = " ".join(str(val) for val in X[i])
  print(f"Entrada: {entrada} | Yd: {Yd[i]} | Yobt: {yobt[i]}")
