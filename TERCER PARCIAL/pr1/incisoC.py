import numpy as np

# Función de activación escalón
def escalon(z):
  if z >= 0:
    return 1
  else:
    return 0

# Datos de entrada (X) y salida deseada (Yd)
X = np.array([
  [0, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 1, 1],
  [0, 1, 0, 1],
  [0, 1, 1, 0],
  [1, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 1, 1, 1]
])

Yd = np.array([0, 0, 1, 1, 1, 0, 1, 1])
yobt = np.zeros_like(Yd)

# Parámetros
lr = 0.2
epochs = 100
x0 = 1

# Inicialización de pesos aleatorios entre 0 y 1
W = np.random.rand(4)
w0 = np.random.rand()

# Entrenamiento
for i in range(epochs):
  for j in range(len(Yd)):
    z = w0 * x0 + np.sum(W * X[j])
    y = escalon(z)
    yobt[j] = y
    if y != Yd[j]:
      error = y - Yd[j]
      w0 = w0 - lr * error
      for k in range(4):
        W[k] = W[k] - lr * error * X[j][k]

# Resultados finales
print("\nResultados finales:")
print("Entradas     Yd      Yobt")
for i in range(len(Yd)):
  entrada = " ".join(str(x) for x in X[i])
  print(entrada, "  |  ", Yd[i], "  |  ", yobt[i])


# ----------------- Fase de operación ---------------------
print("\n--- Fase de operación ---")
x1 = int(input("Ingrese x1: "))
x2 = int(input("Ingrese x2: "))
x3 = int(input("Ingrese x3: "))
x4 = int(input("Ingrese x4: "))

X_op = np.array([x1, x2, x3, x4])
z_op = w0*x0 + np.sum(W * X_op)
y_op = escalon(z_op)
print("El valor de Y es:", y_op)