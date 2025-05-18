import numpy as np
import matplotlib.pyplot as plt


##definir X(vector de caracteristicas) y Y (Vector de etiquetas)
X1 = np.array ([1,2,3,4,5,6,7,8,9])
Yd = np.array ([2,5,8,10,14,5,19,20,24.5])

##Definir hiperparametros/parametros
b0 = 0.1
b1 = 0.2

## Se aconseja un learning rate entre (0,1), lr = learning rate
lr = 0.06
## Num de Muestras
m = len(X1)
## Factor de atenuacion Lambda
L = 0.000002
##Definir epocas (Epocas en IA son las iteraciones)
epochs = 100000
tolerancia = 0.0001
##Definir variables
ecm_act = 0
ECM_record = []
ecm_anterior = float('inf')  # Comienza en infinito
for i in range (epochs):
  Yobt = b0 + b1*X1

  ##Descenso de gradiente
  b0 = b0 - (lr/m)*np.sum(Yobt - Yd)
  b1 += - (lr/m)*np.dot((Yobt-Yd), X1) + 2*L*b1

  ecm_act = (1/(2*m))*np.sum((Yd-Yobt)**2) + L*b1**2
  ECM_record += [ecm_act]
  if abs(ecm_act - ecm_anterior) < tolerancia:
        print(f"Parando en la época {i}, diferencia mínima alcanzada.")
        break
  #se actualiza ecm anterior
  ecm_anterior = ecm_act

print("El valor de b0 es: ", b0)
print("El valor de b1 es: ", b1)
print("El ultimo ecm es: ", ecm_act)

# Graficar datos 
plt.scatter(X1, Yd, label='Datos reales')
plt.plot(X1, b1 *X1 + b0, color='red', label='Línea ajustada')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("Escribe la x: ")
xp = float(input())
yp=b1*xp+b0
print("valor correspondiente: ", yp)





