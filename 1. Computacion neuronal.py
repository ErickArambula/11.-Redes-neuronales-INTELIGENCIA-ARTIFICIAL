# Función de activación de una neurona
def funcion_activacion(entrada, umbral):
    if entrada >= umbral:
        return 1  # Neurona activada
    else:
        return 0  # Neurona desactivada

# Entradas para la neurona
entrada1 = 0.7
entrada2 = 0.2

# Umbral de activación
umbral = 0.5

# Calcular la salida de la neurona
salida = funcion_activacion(entrada1 + entrada2, umbral)

# Imprimir la salida
print("Salida de la neurona:", salida)
