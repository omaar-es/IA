#Perceptrón
import numpy as np

## Función de activación
def escalon(z):
  if z>=0:
    return 1
  else:
    return 0

def escalon_modifi(z):
  if z>=.5:
    return 1
  else:
    return 0

def sigmoide (z):
  return 1/(1+np.exp(-z))

def tanh(z):
  return np.tanh(z)

def bipolar(z):
  if z>0:
    return 1
  elif z==0:
    return 0
  else:
    return -1

def relu(z):
  return max(0,z);
## Definición de los hiperparámetros

## lr toma valores entre 0 y 1 siempre que no esté dividido entre m
lr = .2

## Los pesos, se recomienda inicializarlos entre 0 y 1 pero de forma aleatoria
## Para la práctica todos los pesos deben ser inicializados de manera aleatoria
w0 = -2
x0 = 1

W = np.array([0.1, 0.7])

## Epocas máximas
epochs = 20

## Problema a resolver
#Salidas
Yd = np.array([0, 1, 1, 1])
yobt = np.zeros_like(Yd)

#Entradas
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

################### ENTRENAMIENTO #####################


## Implementar el early stop



for i in range(epochs):
  print(f"Epoca {i}: ")
  print("Yobt = ", yobt)
  print("W = ", W)
  print("w0 = ", w0)
  for j in range(len(Yd)):
    z = w0*x0 + np.sum(W*X[j, :])
    yobt[j] = escalon_modifi(z)
    if yobt[j] != Yd[j]:
      ##Actualización de pesos
      w0 = w0 - lr*(yobt[j] - Yd[j])
      W = W - lr*(yobt[j] - Yd[j])*X[j, :]
      print(f"Nuevos pesos w0 = {w0} y W = {W}")

print("Yobt = ", yobt)
print("W = ", W)
print("w0 = ", w0)


##ETAPA DE OPERACIÓN
X1 = int(input("Introduce el valor de X1:"))
X2 = int(input("Introduce el valor de X2:"))

X_op = np.array([X1, X2])
z = w0*x0 + np.sum(W*X_op)
yobt_op = np.round(sigmoide(z))

print("El valor de Y es: ", yobt_op)