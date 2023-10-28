import numpy as np

# Función para crear una matriz de pesos de una red de Hopfield
def crear_matriz_pesos(patrones):
    num_patrones, num_neuronas = patrones.shape
    pesos = np.zeros((num_neuronas, num_neuronas))

    for i in range(num_neuronas):
        for j in range(num_neuronas):
            if i != j:
                for p in range(num_patrones):
                    pesos[i, j] += patrones[p, i] * patrones[p, j]

    return pesos

# Función para recuperar un patrón en una red de Hopfield
def recuperar_patron(entrada, pesos, max_iter=10):
    num_neuronas = len(entrada)
    for _ in range(max_iter):
        for i in range(num_neuronas):
            entrada[i] = np.sign(np.dot(pesos[i, :], entrada))

    return entrada

# Patrones de ejemplo
patrones = np.array([[1, 1, -1, -1],
                    [1, -1, 1, -1]])

# Crear la matriz de pesos
pesos = crear_matriz_pesos(patrones)

# Entrada de prueba con ruido
entrada_ruidosa = np.array([-1, -1, 1, -1])

# Recuperar un patrón en la red de Hopfield
patron_recuperado = recuperar_patron(entrada_ruidosa, pesos)

print("Patrón original:", entrada_ruidosa)
print("Patrón recuperado:", patron_recuperado)
