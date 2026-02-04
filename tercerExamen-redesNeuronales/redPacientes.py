import numpy as np
import csv
import os
import math

class RedNeuronal:
    """
    Implementación minimalista de una red neuronal con backpropagation
    para clasificación multiclase, usando solo numpy y funciones básicas.
    """
    
    def __init__(self):
        """Configuración inicial de la red neuronal"""
        # Hiperparámetros configurables
        self.tasa_aprendizaje = 0.1
        self.max_epocas = 1000
        self.tolerancia = 0.001  # Para early stopping
        self.bias = True  # Usar bias en las capas
        self.num_neuronas_ocultas = 10
        
        # Variables internas
        self.pesos1 = None  # Capa entrada -> oculta
        self.pesos2 = None  # Capa oculta -> salida
        self.bias1 = None   # Bias capa oculta
        self.bias2 = None   # Bias capa salida
        
    def _funcion_sigmoide(self, x):
        """Función de activación sigmoide"""
        return 1 / (1 + np.exp(-x))
    
    def _derivada_sigmoide(self, x):
        """Derivada de la función sigmoide"""
        return x * (1 - x)
    
    def _funcion_softmax(self, x):
        """Función softmax para clasificación multiclase"""
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def _inicializar_pesos(self, tam_entrada, tam_salida, tam_oculta):
        """Inicializa los pesos y biases de la red"""
        np.random.seed(42)  # Para reproducibilidad
        
        # Pesos capa entrada -> oculta
        self.pesos1 = np.random.randn(tam_entrada, tam_oculta) * 0.1
        
        # Pesos capa oculta -> salida
        self.pesos2 = np.random.randn(tam_oculta, tam_salida) * 0.1
        
        # Inicializar biases si está configurado
        if self.bias:
            self.bias1 = np.zeros((1, tam_oculta))
            self.bias2 = np.zeros((1, tam_salida))
        else:
            self.bias1 = np.zeros((1, tam_oculta))
            self.bias2 = np.zeros((1, tam_salida))
    
    def _propagacion_adelante(self, X):
        """Realiza la propagación hacia adelante"""
        # Capa oculta
        self.suma_oculta = np.dot(X, self.pesos1) + self.bias1
        self.activacion_oculta = self._funcion_sigmoide(self.suma_oculta)
        
        # Capa de salida
        self.suma_salida = np.dot(self.activacion_oculta, self.pesos2) + self.bias2
        self.activacion_salida = self._funcion_softmax(self.suma_salida)
        
        return self.activacion_salida
    
    def _propagacion_atras(self, X, y, salida):
        """Realiza la propagación hacia atrás y actualiza pesos"""
        # Error en la capa de salida
        error_salida = y - salida
        
        # Error en la capa oculta
        error_oculta = np.dot(error_salida, self.pesos2.T)
        delta_oculta = error_oculta * self._derivada_sigmoide(self.activacion_oculta)
        
        # Actualizar pesos y biases
        self.pesos2 += np.dot(self.activacion_oculta.T, error_salida) * self.tasa_aprendizaje
        self.pesos1 += np.dot(X.T, delta_oculta) * self.tasa_aprendizaje
        
        if self.bias:
            self.bias2 += np.sum(error_salida, axis=0, keepdims=True) * self.tasa_aprendizaje
            self.bias1 += np.sum(delta_oculta, axis=0, keepdims=True) * self.tasa_aprendizaje
    
    def _error_cruzada_entropia(self, y_real, y_pred):
        """Calcula la función de pérdida (cross-entropy)"""
        epsilon = 1e-10  # Para evitar log(0)
        return -np.mean(y_real * np.log(y_pred + epsilon))
    
    def _precision_clasificacion(self, y_real, y_pred):
        """Calcula la precisión de clasificación"""
        predicciones = np.argmax(y_pred, axis=1)
        real = np.argmax(y_real, axis=1)
        return np.mean(predicciones == real)
    
    def entrenar(self, X_entrenamiento, y_entrenamiento, X_validacion=None, y_validacion=None):
        """
        Entrena la red neuronal con early stopping
        
        Parámetros:
            X_entrenamiento: Datos de entrenamiento
            y_entrenamiento: Etiquetas de entrenamiento (one-hot)
            X_validacion: Datos para validación (opcional)
            y_validacion: Etiquetas para validación (opcional)
        """
        if X_validacion is None or y_validacion is None:
            X_validacion, y_validacion = X_entrenamiento, y_entrenamiento
        
        n_entrada = X_entrenamiento.shape[1]
        n_salida = y_entrenamiento.shape[1]
        
        self._inicializar_pesos(n_entrada, n_salida, self.num_neuronas_ocultas)
        
        mejor_error = float('inf')
        epocas_sin_mejora = 0
        
        print("\nConfiguración de la Red:")
        print(f"- Tasa de aprendizaje: {self.tasa_aprendizaje}")
        print(f"- Máximo de épocas: {self.max_epocas}")
        print(f"- Tolerancia para early stopping: {self.tolerancia}")
        print(f"- Neuronas en capa oculta: {self.num_neuronas_ocultas}")
        print(f"- Uso de bias: {'Sí' if self.bias else 'No'}")
        print("\nComenzando entrenamiento...\n")
        
        for epoca in range(self.max_epocas):
            # Propagación hacia adelante
            salida = self._propagacion_adelante(X_entrenamiento)
            
            # Propagación hacia atrás
            self._propagacion_atras(X_entrenamiento, y_entrenamiento, salida)
            
            # Calcular métricas
            if epoca % 100 == 0:
                perdida = self._error_cruzada_entropia(y_entrenamiento, salida)
                precision = self._precision_clasificacion(y_entrenamiento, salida)
                
                # Validación para early stopping
                salida_valid = self._propagacion_adelante(X_validacion)
                perdida_valid = self._error_cruzada_entropia(y_validacion, salida_valid)
                
                print(f"Época {epoca}: Pérdida={perdida:.4f} (entrenamiento), {perdida_valid:.4f} (validación) | Precisión={precision:.4f}")
                
                # Early stopping
                if perdida_valid < mejor_error - self.tolerancia:
                    mejor_error = perdida_valid
                    epocas_sin_mejora = 0
                else:
                    epocas_sin_mejora += 1
                    
                if epocas_sin_mejora >= 5:  # Paciencia de 5 épocas
                    print(f"\nEarly stopping en época {epoca} - Sin mejora significativa")
                    break
                
        print("\nEntrenamiento completado")
    
    def predecir(self, X):
        """Realiza predicciones para nuevos datos"""
        salida = self._propagacion_adelante(X)
        return np.argmax(salida, axis=1)

def cargar_datos(nombre_archivo):
    """Carga los datos desde un archivo CSV"""
    datos = []
    etiquetas = []
    
    with open(nombre_archivo, 'r') as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            # Convertir y almacenar características
            datos.append([
                float(fila['Temperatura']),
                float(fila['Presion']),
                float(fila['Glucosa']),
                int(fila['DolorCabeza']),
                int(fila['Sed'])
            ])
            # Almacenar etiquetas
            etiquetas.append(fila['Diagnostico'])
    
    return np.array(datos), np.array(etiquetas)

def normalizar_datos(datos):
    """Normaliza los datos al rango [0, 1]"""
    minimos = np.min(datos, axis=0)
    maximos = np.max(datos, axis=0)
    return (datos - minimos) / (maximos - minimos + 1e-10), minimos, maximos

def codificar_etiquetas(etiquetas):
    """Codifica las etiquetas a one-hot encoding"""
    categorias = {'Gripe': 0, 'Normal': 1, 'Hipertension': 2, 'Deshidratacion': 3, 'Diabetes': 4}
    y_encoded = np.array([categorias[e] for e in etiquetas])
    
    # Convertir a one-hot
    y_onehot = np.zeros((y_encoded.size, len(categorias)))
    y_onehot[np.arange(y_encoded.size), y_encoded] = 1
    
    return y_onehot, categorias

def obtener_entrada_usuario(minimos, maximos):
    """Obtiene y normaliza los parámetros del paciente"""
    print("\nIngrese los parámetros del paciente:")
    temperatura = float(input("Temperatura (C°): "))
    presion = float(input("Presión arterial (mmHg): "))
    glucosa = float(input("Nivel de glucosa (mg/dL): "))
    dolor_cabeza = int(input("Dolor de cabeza (0 - No, 1 - Sí): "))
    sed = int(input("Sed (0 - No, 1 - Sí): "))
    
    # Normalizar
    entrada = np.array([temperatura, presion, glucosa, dolor_cabeza, sed])
    entrada_normalizada = (entrada - minimos) / (maximos - minimos + 1e-10)
    
    return entrada_normalizada.reshape(1, -1)

def mostrar_diagnostico(indice, categorias):
    """Muestra el diagnóstico como texto"""
    diagnostico = [k for k, v in categorias.items() if v == indice][0]
    print(f"\nDiagnóstico: {diagnostico.upper()}")
    print("Explicación:")
    if diagnostico == 'gripe':
        print("- Síntomas de infección respiratoria (fiebre, dolor de cabeza)")
    elif diagnostico == 'normal':
        print("- Todos los parámetros dentro de rangos saludables")
    elif diagnostico == 'hipertension':
        print("- Presión arterial elevada detectada")
    elif diagnostico == 'deshidratacion':
        print("- Signos de deshidratación (sed, posible presión baja)")
    elif diagnostico == 'diabetes':
        print("- Niveles de glucosa elevados detectados")
    print()

def main():
    # Cargar y preparar datos
    print("Cargando datos de pacientes...")
    datos, etiquetas = cargar_datos('datos_pacientes.csv')
    X_normalizado, minimos, maximos = normalizar_datos(datos)
    y_onehot, categorias = codificar_etiquetas(etiquetas)
    
    # Dividir en entrenamiento y validación (80-20)
    indices = np.random.permutation(len(X_normalizado))
    tam_entrenamiento = int(0.8 * len(indices))
    
    X_entrenamiento = X_normalizado[indices[:tam_entrenamiento]]
    y_entrenamiento = y_onehot[indices[:tam_entrenamiento]]
    X_validacion = X_normalizado[indices[tam_entrenamiento:]]
    y_validacion = y_onehot[indices[tam_entrenamiento:]]
    
    # Crear y entrenar red neuronal
    red = RedNeuronal()
    red.entrenar(X_entrenamiento, y_entrenamiento, X_validacion, y_validacion)
    
    # Evaluar en conjunto de validación
    predicciones = red.predecir(X_validacion)
    precision = np.mean(predicciones == np.argmax(y_validacion, axis=1))
    print(f"\nPrecisión en conjunto de validación: {precision:.4f}")
    
    # Modo interactivo
    print("\nModo de diagnóstico interactivo (presione Ctrl+C para salir)")
    while True:
        try:
            entrada = obtener_entrada_usuario(minimos, maximos)
            prediccion = red.predecir(entrada)[0]
            mostrar_diagnostico(prediccion, categorias)
        except ValueError:
            print("Error: Por favor ingrese valores válidos.")
        except KeyboardInterrupt:
            print("\nSaliendo del programa...")
            break

if __name__ == "__main__":
    main()
