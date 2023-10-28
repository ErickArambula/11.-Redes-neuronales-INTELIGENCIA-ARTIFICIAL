import math

# Función de activación sigmoide
def sigmoid(entrada):
    return 1 / (1 + math.exp(-entrada))

# Entradas
entrada = 2.0

# Calcular la salida de la neurona
salida = sigmoid(entrada)

# Imprimir la salida
print("Salida de la neurona:", salida)
