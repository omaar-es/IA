import numpy as np

# Función de activación
def escalon(z):
    if z>=0:
        return 1
    else:
        return 0

# Hiperparámetros
lr = 0.2
w0 = -2
x0 = 1
W = np.array([0.1, 0.7])
epochs = 9

# Datos de entrenamiento
Yd = np.array([0, 0, 0, 1])
Yobt = np.zeros_like(Yd)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# ENTRENAMIENTO
for i in range(epochs):
    print(f"Época {i}: ", Yobt)
    print("Yobt = ", Yobt)
    print("W = ", W)
    print("w0 = ", w0)
    for j in range(len(Yd)):
        z = w0 * x0 + np.sum(W * X[j, :])
        Yobt[j] = escalon(z)
        # Actualización de pesos
        if Yobt[j] != Yd[j]:
            error = Yobt[j] - Yd[j]
            w0 = w0 - lr * error
            W = W - lr * error * X[j, :]
            print(f"Nuevos pesos w0 = {w0} y W = {W}")

# Resultado final
print("Yobt = ", Yobt)
print("W = ", W)
print("w0 = ", w0)

# OPERACIÓN
X1 = int(input("Introduce el valor de X1: "))
X2 = int(input("Introduce el valor de X2: "))
X_op = np.array([X1, X2])
z = w0 * x0 + np.sum(W * X_op)
Yobt_op = escalon(z)
print("El valor de Y es: ", Yobt_op)
