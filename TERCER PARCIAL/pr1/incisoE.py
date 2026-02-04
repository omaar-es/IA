import numpy as np
import matplotlib.pyplot as plt

def escalon(z):
    return 1 if z >= 0 else 0


def agregar_ruido(patron, prob_ruido=0.01):
    """Devuelve una copia del patrón con ruido binario"""
    ruido = np.random.rand(*patron.shape) < prob_ruido
    return np.bitwise_xor(patron, ruido.astype(int))


# Clase 0  =>  Letra "F"
# Clase 1  =>  Número "4"

# --- Letra F (clase 0) ---

letra_F = np.array([
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],  
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
], dtype=int)

numero_5 = np.array([
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
  [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0]
], dtype=int)

# Conjunto de entrenamiento
X  = []  # entradas
Yd = []  # salidas deseadas

fig, axs = plt.subplots(2, 5, figsize=(10, 4))
fig.suptitle("Muestras de entrenamiento (con ruido)")

# 5 letras F 
for i in range(5):
    patron = agregar_ruido(letra_F, prob_ruido=0.08)
    axs[0, i].imshow(patron, cmap="Greys")
    axs[0, i].axis("off")
    X+=[patron.flatten()]
    Yd+=[0]

# 5 números 5 
for i in range(5):
    patron = agregar_ruido(numero_5, prob_ruido=0.08)
    axs[1, i].imshow(patron, cmap="Greys")
    axs[1, i].axis("off")
    X+=[patron.flatten()]
    Yd+=[1]

plt.tight_layout()
plt.show()

# Preparar datos
X  = np.array(X)         # 10 × 256
Yd = np.array(Yd)        # 10
yobt = np.zeros_like(Yd) # resultados deseados

# Hiperparámetros
np.random.seed(0)
W  = np.random.rand(256)  # pesos aleatorios entre 0 y 1
w0 = np.random.rand()    # sesgo
lr = 0.2
epochs = 1000

for _ in range(epochs):
    for i in range(len(Yd)):
        z = w0 + np.sum(W * X[i])
        y = escalon(z)
        yobt[i] = y
        if y != Yd[i]:
            error = y - Yd[i]
            w0 = w0 - lr * error
            for j in range(len(W)):
                W[j] = W[j] - lr * error * X[i][j]

print("\n--- Resultados del entrenamiento ---")
print("Y deseada :", Yd)
print("Y obtenida:", yobt)
print("Precisión :", np.mean(yobt == Yd) * 100, "%")

# ------------------------------
# Fase de operación
# ------------------------------
# Nuevas muestras con algo de ruido
nueva_F  = agregar_ruido(letra_F,  prob_ruido=0.14)
nuevo_5  = agregar_ruido(numero_5, prob_ruido=0.14)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(nueva_F, cmap="Greys")
plt.title("F")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(nuevo_5, cmap="Greys")
plt.title("5")
plt.axis("off")
plt.tight_layout()
plt.show()

# Clasificación
z_F = w0 + np.sum(W * nueva_F.flatten())
z_5 = w0 + np.sum(W * nuevo_5.flatten())
cl_F = escalon(z_F)
cl_5 = escalon(z_5)

print("\n--- Clasificación de nuevas muestras ---")
print("F   clasificada como:", cl_F, "  (esperado 0)")
print("5   clasificada como:", cl_5, "  (esperado 1)")
