import numpy as np

def activacion(z):
    if z < 0.5:
        return 0
    elif z < 1.5:
        return 1
    else:
        return 2


# Entradas y salidas deseadas
X = np.array([
  [-1, 0.1],
  [-0.9, 0.7],
  [0.8, 0.1],
  [0.9, 0.9],
  [0.7, 0.2],
  [0.85, 0.95]
])

Yd = np.array([0, 0, 1, 2, 1, 1])
yobt = np.zeros_like(Yd)

# Inicialización de pesos
W = np.random.rand(2)
w0 = np.random.rand()


# Parámetros
lr = 0.2
epochs = 100

# Entrenamiento
for epoca in range(epochs):
  for i in range(len(Yd)):
    z = w0  + W[0]*X[i][0] + W[1]*X[i][1]
    y = activacion(z)
    yobt[i] = y
    if y != Yd[i]:
      error = y - Yd[i]
      w0 = w0 - lr * error
      W[0] = W[0] - lr * error * X[i][0]
      W[1] = W[1] - lr * error * X[i][1]


# Resultados finales
print("\nResultados finales:")
print("Entradas     Yd      Yobt")

for i in range(len(Yd)):
  entrada = " ".join(str(x) for x in X[i])
  print(entrada, "  |  ", Yd[i], "  |  ", yobt[i])

# Fase de operación0.8

print("\n--- Fase de operación ---")
x1 = float(input("Ingresa el valor de x1: "))
x2 = float(input("Ingresa el valor de x2: "))
z = w0  + W[0]*x1 + W[1]*x2
y_op = activacion(z)
print("El valor de Y es:", y_op)
