import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = np.array ([1,2,3,4,5,6,7,8,9])
y = np.array ([2,5,8,10,14,5,19,20,24.5])

# Inicialización
m = 0
b = 0
learning_rate = 0.01
max_epochs = 10000  # Límite superior
tolerancia = 0.0001

n = len(x)
ecm_anterior = float('inf')  # Comienza en infinito

# Entrenamiento
for epoch in range(max_epochs):
    y_pred = m * x + b
    error = y - y_pred
    ecm_actual = np.mean(error**2)  # Error cuadrático medio

    # Revisión del criterio de paro dinámico
    if abs(ecm_actual - ecm_anterior) < tolerancia:
        print(f"Parando en la época {epoch}, diferencia mínima alcanzada.")
        break

    ecm_anterior = ecm_actual

    # Gradientes
    dm = (-2/n) * np.sum(x * error)
    db = (-2/n) * np.sum(error)

    # Actualización
    m -= learning_rate * dm
    b -= learning_rate * db

# Resultados finales
print(f"Épocas ejecutadas: {epoch + 1}")
print(f"Pendiente (m): {m}")
print(f"Intersección (b): {b}")
print(f"ECM final: {ecm_actual}")

# Graficar
plt.scatter(x, y, label='Datos reales')
plt.plot(x, m * x + b, color='red', label='Línea ajustada')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


print("Escribe la x: ")
xp = float(input())
yp=m*xp+b
print("valor correspondiente: ", yp)