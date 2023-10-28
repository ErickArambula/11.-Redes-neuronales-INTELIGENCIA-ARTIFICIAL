import matplotlib.pyplot as plt
import numpy as np

# Generar datos de ejemplo
np.random.seed(0)
n = 50  # Número de puntos por clase
X = np.random.randn(n, 2)  # Clase 1
X = np.vstack([X, np.random.randn(n, 2) + [2, 2]])  # Clase 2
y = np.array([0] * n + [1] * n)  # Etiquetas de clase

# Dibujar los datos
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("Datos de ejemplo")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# Verificar la separabilidad lineal
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(X, y)

# Dibujar la línea de decisión (hiperplano)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
ax = plt.gca()
xlim = ax.get_xlim()
w = perceptron.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - (perceptron.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k--', label="Línea de Decisión")
plt.title("Separabilidad Lineal")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()

# Comprobar si el Perceptrón pudo separar los datos
if len(perceptron.classes_) == 2:
    print("Los datos son linealmente separables.")
else:
    print("Los datos no son linealmente separables.")
