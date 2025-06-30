import numpy as np

# 1. Función sigmoide y su derivada
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 2. Datos para la compuerta AND
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])  # shape: (4, 2)

y = np.array([
    [0],
    [0],
    [0],
    [1]
])  # shape: (4, 1)

# 3. Inicialización de pesos y bias
np.random.seed(1)
w = np.random.randn(2, 1)  # 2 entradas → 1 salida
b = np.random.randn(1)

# 4. Hiperparámetros
learning_rate = 0.1
epochs = 10000

# 5. Entrenamiento (Gradient Descent - todos los datos en cada paso)
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, w) + b  # shape: (4,1)
    a = sigmoid(z)

    # Cálculo del error
    error = a - y
    loss = np.mean(error ** 2)

    # Backpropagation
    dz = error * sigmoid_derivative(z)         # derivada del ECM respecto a z
    dw = np.dot(X.T, dz) / len(X)              # promedio de gradientes
    db = np.mean(dz)                           # promedio del gradiente del bias

    # Gradient Descent: actualización de pesos
    w -= learning_rate * dw
    b -= learning_rate * db

    # Mostrar pérdida ocasionalmente
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

# 6. Pruebas
print("\nResultados finales:")
for x_test in X:
    z = np.dot(x_test, w) + b
    a = sigmoid(z)
    print(f"Entrada: {x_test}, Salida predicha: {a.round(2)}")
