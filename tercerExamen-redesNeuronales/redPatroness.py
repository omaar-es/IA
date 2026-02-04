import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Generación de patrones base
# =============================
def generate_rectangle():
    img = np.zeros((100, 100))
    img[20:80, 20:80] = 1.0
    return img

def generate_circle():
    img = np.zeros((100, 100))
    center = (50, 50)
    for i in range(100):
        for j in range(100):
            if (i - center[0])**2 + (j - center[1])**2 <= 400:  # Radio ~30 píxeles
                img[i, j] = 1.0
    return img

def generate_triangle():
    img = np.zeros((100, 100))
    for i in range(40, 80):
        width = i - 40
        img[i, 50 - width:50 + width + 1] = 1.0
    return img

def generate_cross():
    img = np.zeros((100, 100))
    img[40:61, :] = 1.0  # Barra horizontal
    img[:, 40:61] = 1.0  # Barra vertical
    return img

# =============================
# 2. Generar conjunto de entrenamiento con ruido
# =============================
def add_noise(img, intensity=0.3):
    noise = np.random.uniform(-intensity, intensity, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)

# Crear 5 ejemplos ruidosos por cada clase (20 imágenes total)
pattern_types = [generate_rectangle, generate_circle, generate_triangle, generate_cross]
class_names = ["Rectángulo", "Círculo", "Triángulo", "Cruz"]

# Almacenar dataset de entrenamiento
X_train = []  # Imágenes de entrenamiento
y_train = []  # Etiquetas

plt.figure(figsize=(15, 10))

for class_idx, pattern_fn in enumerate(pattern_types):
    for sample_idx in range(5):  # 5 muestras por clase
        # Crear patrón base
        base_img = pattern_fn()
        
        # Agregar ruido
        noisy_img = add_noise(base_img, intensity=0.4)
        
        # Guardar en dataset
        X_train.append(noisy_img.flatten())  # Aplanar imagen a vector 1D
        y_train.append(class_idx)
        
        # Mostrar en una cuadrícula
        plt.subplot(4, 5, class_idx * 5 + sample_idx + 1)
        plt.imshow(noisy_img, cmap='gray', interpolation='nearest')
        plt.title(f'Clase: {class_names[class_idx]}')
        plt.axis('off')

plt.suptitle('Conjunto de entrenamiento (20 imágenes)', fontsize=16)
plt.tight_layout()
plt.show()

# Convertir a arrays numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

# =============================
# 3. Definir la red neuronal (desde cero) con Early Stopping
# =============================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar pesos con valores pequeños aleatorios
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.loss_history = []  # Historial de pérdida
        self.best_loss = float('inf')  # Mejor pérdida encontrada
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward(self, X):
        # Capa oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Capa de salida
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def compute_loss(self, y, output):
        # Loss de entropía cruzada
        m = y.shape[0]
        log_likelihood = -np.log(output[np.arange(m), y])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backprop(self, X, y, learning_rate):
        m = y.shape[0]
        
        # Derivada de la función de pérdida con respecto a z2
        dz2 = self.a2.copy()
        dz2[np.arange(m), y] -= 1
        dz2 /= m
        
        # Gradientes para pesos y sesgos de capa de salida
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Derivada de la función de pérdida con respecto a a1
        da1 = np.dot(dz2, self.W2.T)
        
        # Derivada de la función de pérdida con respecto a z1
        dz1 = da1 * (self.z1 > 0)  # Derivada de ReLU (1 si z1 > 0, 0 caso contrario)
        
        # Gradientes para pesos y sesgos de capa oculta
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Actualización de parámetros
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, max_epochs, learning_rate, patience=10, tol=1e-4, verbose=True):
        """Entrenamiento con early stopping"""
        patience_counter = 0
        best_weights = None
        
        print(f"Entrenando con Early Stopping (Paciencia={patience}, Tolerancia={tol:.6f})")
        for epoch in range(max_epochs):
            # Paso forward
            output = self.forward(X)
            
            # Cálculo de pérdida
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
            
            # Paso backward
            self.backprop(X, y, learning_rate)
            
            # Comprobar si hay mejora en la pérdida
            if loss < self.best_loss - tol:
                self.best_loss = loss
                patience_counter = 0
                # Guardar los mejores pesos encontrados
                best_weights = {
                    'W1': self.W1.copy(),
                    'b1': self.b1.copy(),
                    'W2': self.W2.copy(),
                    'b2': self.b2.copy()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n¡Early stopping en época {epoch+1}!")
                    print(f"La pérdida no mejoró en las últimas {patience} épocas.")
                    print(f"Mejor pérdida: {self.best_loss:.4f}")
                    
                    # Restaurar los mejores pesos
                    if best_weights is not None:
                        self.W1 = best_weights['W1']
                        self.b1 = best_weights['b1']
                        self.W2 = best_weights['W2']
                        self.b2 = best_weights['b2']
                    break
            
            # Mostrar progreso
            if verbose and epoch % 100 == 0:
                accuracy = np.sum(y == self.predict(X)) / len(y)
                print(f"Época {epoch}/{max_epochs}: Pérdida={loss:.4f}, Accuracy={accuracy*100:.2f}%")
        else:
            print(f"\nEntrenamiento completado en {max_epochs} épocas")
            print(f"Pérdida final: {loss:.4f} (Mejor pérdida: {self.best_loss:.4f})")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    

# =============================
# 4. Entrenamiento de la red con Early Stopping
# =============================
input_size = 100 * 100  # 10,000 pixeles
hidden_size = 50         # Neuronas en capa oculta
output_size = 4          # 4 clases (rectángulo, círculo, triángulo, cruz)

# Inicializar red neuronal
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Entrenar la red con early stopping
print("\nIniciando entrenamiento...\n")
nn.train(X_train, y_train, 
         max_epochs=10000, 
         learning_rate=0.01,
         patience=100,    # Épocas sin mejora después de las cuales se detiene
         tol=0.001)       # Tolerancia para considerar mejora significativa


# =============================
# 5. Fase de operación (prueba) con imágenes ruidosas
# =============================
# Generar nuevas imágenes de prueba (una por patrón con alto nivel de ruido)
test_images = []
test_classes = []
plt.figure(figsize=(10, 5))

print("\nGenerando imágenes de prueba con ruido...\n")
for i, pattern_fn in enumerate(pattern_types):
    base_img = pattern_fn()
    noisy_img = add_noise(base_img, intensity=0.60)  # Más ruido que entrenamiento
    test_images.append(noisy_img.flatten())
    test_classes.append(i)
    
    plt.subplot(1, 4, i+1)
    plt.imshow(noisy_img, cmap='gray', interpolation='nearest')
    plt.title(f'Prueba {class_names[i]}\n(Con ruido intenso)')
    plt.axis('off')

plt.suptitle('Imágenes de Operación con Alto Ruido (fase operativa)', fontsize=16)
plt.tight_layout()
plt.show()

# Convertir a formato adecuado
X_test = np.array(test_images)

# Realizar predicciones
predictions = nn.predict(X_test)

# Resultados detallados
plt.figure(figsize=(12, 4))

print("\n===== RESULTADOS FINALES DE CLASIFICACIÓN =====")
for i, pred_class in enumerate(predictions):
    actual_class = test_classes[i]
    is_correct = actual_class == pred_class
    color = 'green' if is_correct else 'red'
    result = "CORRECTO" if is_correct else "INCORRECTO"
    
    # Mostrar resultados en terminal
    print(f"\nImagen {i+1} ({class_names[actual_class]})")
    print(f"  Etiqueta real: {class_names[actual_class]}")
    print(f"  Predicción: {class_names[pred_class]} → {result}")
    print("-" * 50)
    
    # Mostrar imagen con resultado
    plt.subplot(1, 4, i+1)
    plt.imshow(test_images[i].reshape(100, 100), cmap='gray')  # Convertir vector a matriz 100x100
    plt.title(f'Real: {class_names[actual_class]}\nPredicción: {class_names[pred_class]}', color=color)
    plt.axis('off')

plt.suptitle('Resultados de Clasificación en Fase Operativa', fontsize=16)
plt.tight_layout()
plt.show()

print("\nRESUMEN FINAL:")
accuracy = np.mean(np.array(test_classes) == np.array(predictions)) * 100
print(f"Accuracy en pruebas operativas: {accuracy:.2f}%")
