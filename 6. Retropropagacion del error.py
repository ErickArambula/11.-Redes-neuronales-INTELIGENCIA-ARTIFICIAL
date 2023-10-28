import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrenamiento (puerta lógica XOR)
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
etiquetas = np.array([[0], [1], [1], [0]])

# Inicializar pesos de forma aleatoria
np.random.seed(1)
pesos_capa_oculta = 2 * np.random.random((2, 2)) - 1
pesos_capa_salida = 2 * np.random.random((2, 1)) - 1

# Hiperparámetros
tasa_aprendizaje = 0.1
epocas = 10000

# Entrenamiento de la red neuronal
for epoca in range(epocas):
    # Capa de entrada a capa oculta
    capa_oculta_entrada = np.dot(entradas, pesos_capa_oculta)
    capa_oculta_activacion = sigmoid(capa_oculta_entrada)

    # Capa oculta a capa de salida
    capa_salida_entrada = np.dot(capa_oculta_activacion, pesos_capa_salida)
    capa_salida_activacion = sigmoid(capa_salida_entrada)

    # Calcular el error de predicción
    error = etiquetas - capa_salida_activacion

    # Retropropagación del error
    delta = error * sigmoid_derivative(capa_salida_activacion)
    error_capa_oculta = delta.dot(pesos_capa_salida.T)
    delta_capa_oculta = error_capa_oculta * sigmoid_derivative(capa_oculta_activacion)

    # Actualizar pesos
    pesos_capa_salida += capa_oculta_activacion.T.dot(delta)
    pesos_capa_oculta += entradas.T.dot(delta_capa_oculta)

# Imprimir la salida final
print("Salida después del entrenamiento:")
print(capa_salida_activacion)
